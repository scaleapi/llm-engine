import asyncio
import os
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import aioredis
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy.ext.asyncio import AsyncSession, async_scoped_session

from llm_engine_server.common.config import hmi_config
from llm_engine_server.common.dtos.model_endpoints import BrokerType
from llm_engine_server.common.env_vars import CIRCLECI
from llm_engine_server.core.auth.authentication_repository import AuthenticationRepository, User
from llm_engine_server.core.auth.fake_authentication_repository import FakeAuthenticationRepository
from llm_engine_server.db.base import SessionAsync, SessionReadOnlyAsync
from llm_engine_server.domain.gateways import (
    DockerImageBatchJobGateway,
    ModelPrimitiveGateway,
    TaskQueueGateway,
)
from llm_engine_server.domain.repositories import (
    DockerImageBatchJobBundleRepository,
    DockerRepository,
    ModelBundleRepository,
)
from llm_engine_server.domain.services import (
    BatchJobService,
    LLMModelEndpointService,
    ModelEndpointService,
)
from llm_engine_server.infra.gateways import (
    CeleryTaskQueueGateway,
    FakeMonitoringMetricsGateway,
    LiveAsyncModelEndpointInferenceGateway,
    LiveBatchJobOrchestrationGateway,
    LiveBatchJobProgressGateway,
    LiveDockerImageBatchJobGateway,
    LiveModelEndpointInfraGateway,
    LiveModelEndpointsSchemaGateway,
    LiveStreamingModelEndpointInferenceGateway,
    LiveSyncModelEndpointInferenceGateway,
    ModelEndpointInfraGateway,
    S3FilesystemGateway,
)
from llm_engine_server.infra.gateways.fake_model_primitive_gateway import FakeModelPrimitiveGateway
from llm_engine_server.infra.gateways.resources.endpoint_resource_gateway import (
    EndpointResourceGateway,
)
from llm_engine_server.infra.gateways.resources.fake_sqs_endpoint_resource_delegate import (
    FakeSQSEndpointResourceDelegate,
)
from llm_engine_server.infra.gateways.resources.live_endpoint_resource_gateway import (
    LiveEndpointResourceGateway,
)
from llm_engine_server.infra.gateways.resources.live_sqs_endpoint_resource_delegate import (
    LiveSQSEndpointResourceDelegate,
)
from llm_engine_server.infra.gateways.resources.sqs_endpoint_resource_delegate import (
    SQSEndpointResourceDelegate,
)
from llm_engine_server.infra.repositories import (
    DbBatchJobRecordRepository,
    DbDockerImageBatchJobBundleRepository,
    DbModelBundleRepository,
    DbModelEndpointRecordRepository,
    ECRDockerRepository,
    RedisModelEndpointCacheRepository,
)
from llm_engine_server.infra.services import LiveBatchJobService, LiveModelEndpointService
from llm_engine_server.infra.services.live_llm_model_endpoint_service import (
    LiveLLMModelEndpointService,
)

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
    model_endpoint_service: ModelEndpointService
    batch_job_service: BatchJobService
    llm_model_endpoint_service: LLMModelEndpointService

    resource_gateway: EndpointResourceGateway
    endpoint_creation_task_queue_gateway: TaskQueueGateway
    inference_task_queue_gateway: TaskQueueGateway
    model_endpoint_infra_gateway: ModelEndpointInfraGateway
    docker_image_batch_job_gateway: DockerImageBatchJobGateway
    model_primitive_gateway: ModelPrimitiveGateway


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
    model_endpoints_schema_gateway = LiveModelEndpointsSchemaGateway(
        filesystem_gateway=filesystem_gateway
    )
    model_endpoint_service = LiveModelEndpointService(
        model_endpoint_record_repository=model_endpoint_record_repo,
        model_endpoint_infra_gateway=model_endpoint_infra_gateway,
        model_endpoint_cache_repository=model_endpoint_cache_repo,
        async_model_endpoint_inference_gateway=async_model_endpoint_inference_gateway,
        streaming_model_endpoint_inference_gateway=streaming_model_endpoint_inference_gateway,
        sync_model_endpoint_inference_gateway=sync_model_endpoint_inference_gateway,
        model_endpoints_schema_gateway=model_endpoints_schema_gateway,
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
    batch_job_orchestration_gateway = LiveBatchJobOrchestrationGateway()
    batch_job_progress_gateway = LiveBatchJobProgressGateway(filesystem_gateway=filesystem_gateway)
    batch_job_service = LiveBatchJobService(
        batch_job_record_repository=batch_job_record_repository,
        model_endpoint_service=model_endpoint_service,
        batch_job_orchestration_gateway=batch_job_orchestration_gateway,
        batch_job_progress_gateway=batch_job_progress_gateway,
    )

    model_primitive_gateway: ModelPrimitiveGateway
    model_primitive_gateway = FakeModelPrimitiveGateway()

    docker_image_batch_job_gateway = LiveDockerImageBatchJobGateway()

    external_interfaces = ExternalInterfaces(
        docker_repository=ECRDockerRepository(),
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
    )
    return external_interfaces


async def get_external_interfaces():
    try:
        session = async_scoped_session(SessionAsync, scopefunc=asyncio.current_task)
        yield _get_external_interfaces(read_only=False, session=session)
    finally:
        pass


async def get_external_interfaces_read_only():
    try:
        session = async_scoped_session(SessionReadOnlyAsync, scopefunc=asyncio.current_task)
        yield _get_external_interfaces(read_only=True, session=session)
    finally:
        pass


def get_auth_repository() -> Iterator[AuthenticationRepository]:
    """
    Dependency for an AuthenticationRepository. This implementation returns a Scale-specific repository.
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
    user_id = credentials.username if credentials is not None else None
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No user id was passed in",
            headers={"WWW-Authenticate": "Basic"},
        )

    auth = await auth_repo.get_auth_from_user_id_async(user_id=user_id)

    if not auth:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not authenticate user",
            headers={"WWW-Authenticate": "Basic"},
        )

    return auth


_pool: Optional[aioredis.BlockingConnectionPool] = None


def get_or_create_aioredis_pool() -> aioredis.ConnectionPool:
    global _pool

    if _pool is None:
        _pool = aioredis.BlockingConnectionPool.from_url(hmi_config.cache_redis_url)
    return _pool
