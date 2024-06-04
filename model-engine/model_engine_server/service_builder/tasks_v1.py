import asyncio
import os
from typing import Any, Dict

import aioredis
from celery.signals import worker_process_init
from model_engine_server.api.dependencies import get_monitoring_metrics_gateway
from model_engine_server.common.config import hmi_config
from model_engine_server.common.constants import READYZ_FPATH
from model_engine_server.common.dtos.endpoint_builder import (
    BuildEndpointRequest,
    BuildEndpointResponse,
)
from model_engine_server.common.env_vars import CIRCLECI
from model_engine_server.core.config import infra_config
from model_engine_server.core.fake_notification_gateway import FakeNotificationGateway
from model_engine_server.db.base import get_session_async_null_pool
from model_engine_server.domain.repositories import DockerRepository
from model_engine_server.infra.gateways import ABSFilesystemGateway, S3FilesystemGateway
from model_engine_server.infra.gateways.resources.asb_queue_endpoint_resource_delegate import (
    ASBQueueEndpointResourceDelegate,
)
from model_engine_server.infra.gateways.resources.fake_queue_endpoint_resource_delegate import (
    FakeQueueEndpointResourceDelegate,
)
from model_engine_server.infra.gateways.resources.k8s_endpoint_resource_delegate import (
    set_lazy_load_kubernetes_clients,
)
from model_engine_server.infra.gateways.resources.live_endpoint_resource_gateway import (
    LiveEndpointResourceGateway,
)
from model_engine_server.infra.gateways.resources.queue_endpoint_resource_delegate import (
    QueueEndpointResourceDelegate,
)
from model_engine_server.infra.gateways.resources.sqs_queue_endpoint_resource_delegate import (
    SQSQueueEndpointResourceDelegate,
)
from model_engine_server.infra.repositories import (
    ACRDockerRepository,
    DbModelEndpointRecordRepository,
    ECRDockerRepository,
    FakeDockerRepository,
    RedisFeatureFlagRepository,
    RedisModelEndpointCacheRepository,
)
from model_engine_server.infra.services import LiveEndpointBuilderService
from model_engine_server.service_builder.celery import service_builder_service

SessionAsyncNullPool = get_session_async_null_pool()

# Need to disable lazy loading of k8s clients because each event loop should contain its own k8s
# client, which constructs the aiohttp.ClientSession in the event loop.
set_lazy_load_kubernetes_clients(False)


def get_live_endpoint_builder_service(
    session: Any,
    redis: aioredis.Redis,
):
    queue_delegate: QueueEndpointResourceDelegate
    if CIRCLECI:
        queue_delegate = FakeQueueEndpointResourceDelegate()
    elif infra_config().cloud_provider == "azure":
        queue_delegate = ASBQueueEndpointResourceDelegate()
    else:
        queue_delegate = SQSQueueEndpointResourceDelegate(
            sqs_profile=os.getenv("SQS_PROFILE", hmi_config.sqs_profile)
        )
    notification_gateway = FakeNotificationGateway()
    monitoring_metrics_gateway = get_monitoring_metrics_gateway()
    docker_repository: DockerRepository
    if CIRCLECI:
        docker_repository = FakeDockerRepository()
    elif infra_config().cloud_provider == "azure":
        docker_repository = ACRDockerRepository()
    else:
        docker_repository = ECRDockerRepository()
    service = LiveEndpointBuilderService(
        docker_repository=docker_repository,
        resource_gateway=LiveEndpointResourceGateway(
            queue_delegate=queue_delegate,
        ),
        monitoring_metrics_gateway=monitoring_metrics_gateway,
        model_endpoint_record_repository=DbModelEndpointRecordRepository(
            monitoring_metrics_gateway=monitoring_metrics_gateway, session=session, read_only=False
        ),
        model_endpoint_cache_repository=RedisModelEndpointCacheRepository(redis_client=redis),
        filesystem_gateway=ABSFilesystemGateway()
        if infra_config().cloud_provider == "azure"
        else S3FilesystemGateway(),
        notification_gateway=notification_gateway,
        feature_flag_repo=RedisFeatureFlagRepository(redis_client=redis),
    )

    return service


async def _build_endpoint(
    build_endpoint_request: BuildEndpointRequest,
) -> BuildEndpointResponse:
    session = SessionAsyncNullPool
    pool = aioredis.BlockingConnectionPool.from_url(hmi_config.cache_redis_url)
    redis = aioredis.Redis(connection_pool=pool)
    service: LiveEndpointBuilderService = get_live_endpoint_builder_service(session, redis)

    response = await service.build_endpoint(build_endpoint_request)
    await redis.close()
    await pool.disconnect()
    return response


@worker_process_init.connect
def init_worker(*args, **kwargs):
    # k8s health check
    with open(READYZ_FPATH, "w") as f:
        f.write("READY")


@service_builder_service.task
def build_endpoint(build_endpoint_request_json: Dict[str, Any]) -> Dict[str, str]:
    build_endpoint_request: BuildEndpointRequest = BuildEndpointRequest.parse_obj(
        build_endpoint_request_json
    )
    result = asyncio.run(_build_endpoint(build_endpoint_request))
    return result.dict()
