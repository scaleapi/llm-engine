import asyncio
import os
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import aioredis
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from model_engine_server.common.config import hmi_config
from model_engine_server.common.dtos.model_endpoints import BrokerType
from model_engine_server.common.env_vars import CIRCLECI
from model_engine_server.core.auth.authentication_repository import AuthenticationRepository, User
from model_engine_server.core.auth.fake_authentication_repository import (
    FakeAuthenticationRepository,
)
from model_engine_server.core.loggers import (
    LoggerTagKey,
    LoggerTagManager,
    filename_wo_ext,
    make_logger,
)
from model_engine_server.db.base import SessionAsync, SessionReadOnlyAsync
from model_engine_server.domain.gateways import (
    CronJobGateway,
    DockerImageBatchJobGateway,
    FileStorageGateway,
    LLMArtifactGateway,
    ModelPrimitiveGateway,
    TaskQueueGateway,
)
from model_engine_server.domain.repositories import (
    DockerImageBatchJobBundleRepository,
    DockerRepository,
    LLMFineTuneEventsRepository,
    ModelBundleRepository,
    TriggerRepository,
)
from model_engine_server.domain.services import (
    BatchJobService,
    LLMFineTuningService,
    LLMModelEndpointService,
    ModelEndpointService,
)
from model_engine_server.infra.gateways import (
    CeleryTaskQueueGateway,
    FakeMonitoringMetricsGateway,
    LiveAsyncModelEndpointInferenceGateway,
    LiveBatchJobOrchestrationGateway,
    LiveBatchJobProgressGateway,
    LiveCronJobGateway,
    LiveDockerImageBatchJobGateway,
    LiveModelEndpointInfraGateway,
    LiveModelEndpointsSchemaGateway,
    LiveStreamingModelEndpointInferenceGateway,
    LiveSyncModelEndpointInferenceGateway,
    ModelEndpointInfraGateway,
    RedisInferenceAutoscalingMetricsGateway,
    S3FilesystemGateway,
    S3LLMArtifactGateway,
)
from model_engine_server.infra.gateways.fake_model_primitive_gateway import (
    FakeModelPrimitiveGateway,
)
from model_engine_server.infra.gateways.filesystem_gateway import FilesystemGateway
from model_engine_server.infra.gateways.resources.endpoint_resource_gateway import (
    EndpointResourceGateway,
)
from model_engine_server.infra.gateways.resources.fake_sqs_endpoint_resource_delegate import (
    FakeSQSEndpointResourceDelegate,
)
from model_engine_server.infra.gateways.resources.live_endpoint_resource_gateway import (
    LiveEndpointResourceGateway,
)
from model_engine_server.infra.gateways.resources.live_sqs_endpoint_resource_delegate import (
    LiveSQSEndpointResourceDelegate,
)
from model_engine_server.infra.gateways.resources.sqs_endpoint_resource_delegate import (
    SQSEndpointResourceDelegate,
)
from model_engine_server.infra.gateways.s3_file_storage_gateway import S3FileStorageGateway
from model_engine_server.infra.repositories import (
    DbBatchJobRecordRepository,
    DbDockerImageBatchJobBundleRepository,
    DbModelBundleRepository,
    DbModelEndpointRecordRepository,
    DbTriggerRepository,
    ECRDockerRepository,
    FakeDockerRepository,
    RedisModelEndpointCacheRepository,
    S3FileLLMFineTuneEventsRepository,
    S3FileLLMFineTuneRepository,
)
from model_engine_server.infra.services import (
    DockerImageBatchJobLLMFineTuningService,
    LiveBatchJobService,
    LiveModelEndpointService,
)
from model_engine_server.infra.services.live_llm_model_endpoint_service import (
    LiveLLMModelEndpointService,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_scoped_session

logger = make_logger(filename_wo_ext(__name__))

AUTH = HTTPBasic(auto_error=False)


@dataclass
class ExternalInterfaces:
    """
    Internal object used for aggregating various Gateway and Repository objects for dependency
    injection.
    """

    docker_repository: DockerRepository
    docker_image_batch_job_bundle_repository: DockerImageBatchJobBundleRepository
    model_bundle_repository: ModelBundleRepository
    trigger_repository: TriggerRepository
    model_endpoint_service: ModelEndpointService
    batch_job_service: BatchJobService
    llm_model_endpoint_service: LLMModelEndpointService
    llm_fine_tuning_service: LLMFineTuningService
    llm_fine_tune_events_repository: LLMFineTuneEventsRepository

    resource_gateway: EndpointResourceGateway
    endpoint_creation_task_queue_gateway: TaskQueueGateway
    inference_task_queue_gateway: TaskQueueGateway
    model_endpoint_infra_gateway: ModelEndpointInfraGateway
    docker_image_batch_job_gateway: DockerImageBatchJobGateway
    model_primitive_gateway: ModelPrimitiveGateway
    file_storage_gateway: FileStorageGateway
    filesystem_gateway: FilesystemGateway
    llm_artifact_gateway: LLMArtifactGateway
    cron_job_gateway: CronJobGateway


def _get_external_interfaces(
    read_only: bool, session: Callable[[], AsyncSession]
) -> ExternalInterfaces:
    """
    Dependency that returns a ExternalInterfaces object. This allows repositories to share
    sessions for the database and redis.
    """
    redis_task_queue_gateway = CeleryTaskQueueGateway(broker_type=BrokerType.REDIS)
    redis_24h_task_queue_gateway = CeleryTaskQueueGateway(broker_type=BrokerType.REDIS_24H)
    sqs_task_queue_gateway = CeleryTaskQueueGateway(broker_type=BrokerType.SQS)
    monitoring_metrics_gateway = FakeMonitoringMetricsGateway()
    model_endpoint_record_repo = DbModelEndpointRecordRepository(
        monitoring_metrics_gateway=monitoring_metrics_gateway,
        session=session,
        read_only=read_only,
    )

    sqs_delegate: SQSEndpointResourceDelegate
    if CIRCLECI:
        sqs_delegate = FakeSQSEndpointResourceDelegate()
    else:
        sqs_delegate = LiveSQSEndpointResourceDelegate(
            sqs_profile=os.getenv("SQS_PROFILE", hmi_config.sqs_profile)
        )

    inference_task_queue_gateway = (
        sqs_task_queue_gateway if not CIRCLECI else redis_24h_task_queue_gateway
    )
    resource_gateway = LiveEndpointResourceGateway(sqs_delegate=sqs_delegate)
    redis_client = aioredis.Redis(connection_pool=get_or_create_aioredis_pool())
    model_endpoint_cache_repo = RedisModelEndpointCacheRepository(
        redis_client=redis_client,
    )
    model_endpoint_infra_gateway = LiveModelEndpointInfraGateway(
        resource_gateway=resource_gateway,
        task_queue_gateway=redis_task_queue_gateway,
    )
    async_model_endpoint_inference_gateway = LiveAsyncModelEndpointInferenceGateway(
        task_queue_gateway=inference_task_queue_gateway
    )
    # In CircleCI, we cannot use asyncio because aiohttp cannot connect to the sync endpoints.
    sync_model_endpoint_inference_gateway = LiveSyncModelEndpointInferenceGateway(
        use_asyncio=(not CIRCLECI),
    )
    streaming_model_endpoint_inference_gateway = LiveStreamingModelEndpointInferenceGateway(
        use_asyncio=(not CIRCLECI),
    )
    filesystem_gateway = S3FilesystemGateway()
    llm_artifact_gateway = S3LLMArtifactGateway()
    model_endpoints_schema_gateway = LiveModelEndpointsSchemaGateway(
        filesystem_gateway=filesystem_gateway
    )
    inference_autoscaling_metrics_gateway = RedisInferenceAutoscalingMetricsGateway(
        redis_client=redis_client
    )  # we can just reuse the existing redis client, we shouldn't get key collisions because of the prefix
    model_endpoint_service = LiveModelEndpointService(
        model_endpoint_record_repository=model_endpoint_record_repo,
        model_endpoint_infra_gateway=model_endpoint_infra_gateway,
        model_endpoint_cache_repository=model_endpoint_cache_repo,
        async_model_endpoint_inference_gateway=async_model_endpoint_inference_gateway,
        streaming_model_endpoint_inference_gateway=streaming_model_endpoint_inference_gateway,
        sync_model_endpoint_inference_gateway=sync_model_endpoint_inference_gateway,
        model_endpoints_schema_gateway=model_endpoints_schema_gateway,
        inference_autoscaling_metrics_gateway=inference_autoscaling_metrics_gateway,
    )
    llm_model_endpoint_service = LiveLLMModelEndpointService(
        model_endpoint_record_repository=model_endpoint_record_repo,
        model_endpoint_service=model_endpoint_service,
    )
    model_bundle_repository = DbModelBundleRepository(session=session, read_only=read_only)
    docker_image_batch_job_bundle_repository = DbDockerImageBatchJobBundleRepository(
        session=session, read_only=read_only
    )
    batch_job_record_repository = DbBatchJobRecordRepository(session=session, read_only=read_only)
    trigger_repository = DbTriggerRepository(session=session, read_only=read_only)
    batch_job_orchestration_gateway = LiveBatchJobOrchestrationGateway()
    batch_job_progress_gateway = LiveBatchJobProgressGateway(filesystem_gateway=filesystem_gateway)
    batch_job_service = LiveBatchJobService(
        batch_job_record_repository=batch_job_record_repository,
        model_endpoint_service=model_endpoint_service,
        batch_job_orchestration_gateway=batch_job_orchestration_gateway,
        batch_job_progress_gateway=batch_job_progress_gateway,
    )

    model_primitive_gateway = FakeModelPrimitiveGateway()

    docker_image_batch_job_gateway = LiveDockerImageBatchJobGateway()
    cron_job_gateway = LiveCronJobGateway()

    llm_fine_tune_repository = S3FileLLMFineTuneRepository(
        file_path=os.getenv(
            "S3_FILE_LLM_FINE_TUNE_REPOSITORY",
            hmi_config.s3_file_llm_fine_tune_repository,
        ),
    )
    llm_fine_tune_events_repository = S3FileLLMFineTuneEventsRepository()
    llm_fine_tuning_service = DockerImageBatchJobLLMFineTuningService(
        docker_image_batch_job_gateway=docker_image_batch_job_gateway,
        docker_image_batch_job_bundle_repo=docker_image_batch_job_bundle_repository,
        llm_fine_tune_repository=llm_fine_tune_repository,
    )

    file_storage_gateway = S3FileStorageGateway()

    docker_repository = ECRDockerRepository() if not CIRCLECI else FakeDockerRepository()

    external_interfaces = ExternalInterfaces(
        docker_repository=docker_repository,
        model_bundle_repository=model_bundle_repository,
        model_endpoint_service=model_endpoint_service,
        llm_model_endpoint_service=llm_model_endpoint_service,
        batch_job_service=batch_job_service,
        resource_gateway=resource_gateway,
        endpoint_creation_task_queue_gateway=redis_task_queue_gateway,
        inference_task_queue_gateway=sqs_task_queue_gateway,
        model_endpoint_infra_gateway=model_endpoint_infra_gateway,
        model_primitive_gateway=model_primitive_gateway,
        docker_image_batch_job_bundle_repository=docker_image_batch_job_bundle_repository,
        docker_image_batch_job_gateway=docker_image_batch_job_gateway,
        llm_fine_tuning_service=llm_fine_tuning_service,
        llm_fine_tune_events_repository=llm_fine_tune_events_repository,
        file_storage_gateway=file_storage_gateway,
        filesystem_gateway=filesystem_gateway,
        llm_artifact_gateway=llm_artifact_gateway,
        trigger_repository=trigger_repository,
        cron_job_gateway=cron_job_gateway,
    )
    return external_interfaces


def get_default_external_interfaces() -> ExternalInterfaces:
    session = async_scoped_session(SessionAsync, scopefunc=asyncio.current_task)  # type: ignore
    return _get_external_interfaces(read_only=False, session=session)


def get_default_external_interfaces_read_only() -> ExternalInterfaces:
    session = async_scoped_session(  # type: ignore
        SessionReadOnlyAsync, scopefunc=asyncio.current_task  # type: ignore
    )
    return _get_external_interfaces(read_only=True, session=session)


async def get_external_interfaces():
    try:
        from plugins.dependencies import get_external_interfaces as get_custom_external_interfaces

        yield get_custom_external_interfaces()
    except ModuleNotFoundError:
        yield get_default_external_interfaces()
    finally:
        pass


async def get_external_interfaces_read_only():
    try:
        from plugins.dependencies import (
            get_external_interfaces_read_only as get_custom_external_interfaces_read_only,
        )

        yield get_custom_external_interfaces_read_only()
    except ModuleNotFoundError:
        yield get_default_external_interfaces_read_only()
    finally:
        pass


def get_auth_repository() -> Iterator[AuthenticationRepository]:
    """
    Dependency for an AuthenticationRepository. This implementation returns a fake repository.
    """
    try:
        yield FakeAuthenticationRepository()
    finally:
        pass


async def verify_authentication(
    credentials: HTTPBasicCredentials = Depends(AUTH),
    auth_repo: AuthenticationRepository = Depends(get_auth_repository),
) -> User:
    """
    Verifies the authentication headers and returns a (user_id, team_id) auth tuple. Otherwise,
    raises a 401.
    """
    username = credentials.username if credentials is not None else None
    if username is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No authentication was passed in",
            headers={"WWW-Authenticate": "Basic"},
        )

    auth = await auth_repo.get_auth_from_username_async(username=username)

    if not auth:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not authenticate user",
            headers={"WWW-Authenticate": "Basic"},
        )

    # set logger context with identity data
    LoggerTagManager.set(LoggerTagKey.USER_ID, auth.user_id)
    LoggerTagManager.set(LoggerTagKey.TEAM_ID, auth.team_id)

    return auth


_pool: Optional[aioredis.BlockingConnectionPool] = None


def get_or_create_aioredis_pool() -> aioredis.ConnectionPool:
    global _pool

    if _pool is None:
        _pool = aioredis.BlockingConnectionPool.from_url(hmi_config.cache_redis_url)
    return _pool
