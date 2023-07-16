import asyncio
import os
from typing import Any, Dict

import aioredis
from celery.signals import worker_process_init

from spellbook_serve.common.config import hmi_config
from spellbook_serve.common.constants import READYZ_FPATH
from spellbook_serve.common.dtos.endpoint_builder import BuildEndpointRequest, BuildEndpointResponse
from spellbook_serve.common.env_vars import CIRCLECI
from spellbook_serve.core.fake_notification_gateway import FakeNotificationGateway
from spellbook_serve.core.notification_gateway import NotificationGateway
from spellbook_serve.db.base import SessionAsyncNullPool
from spellbook_serve.domain.gateways.monitoring_metrics_gateway import MonitoringMetricsGateway
from spellbook_serve.infra.gateways import FakeMonitoringMetricsGateway, S3FilesystemGateway
from spellbook_serve.infra.gateways.resources.fake_sqs_endpoint_resource_delegate import (
    FakeSQSEndpointResourceDelegate,
)
from spellbook_serve.infra.gateways.resources.k8s_endpoint_resource_delegate import (
    set_lazy_load_kubernetes_clients,
)
from spellbook_serve.infra.gateways.resources.live_endpoint_resource_gateway import (
    LiveEndpointResourceGateway,
)
from spellbook_serve.infra.gateways.resources.live_sqs_endpoint_resource_delegate import (
    LiveSQSEndpointResourceDelegate,
)
from spellbook_serve.infra.gateways.resources.sqs_endpoint_resource_delegate import (
    SQSEndpointResourceDelegate,
)
from spellbook_serve.infra.repositories import (
    DbModelEndpointRecordRepository,
    ECRDockerRepository,
    RedisFeatureFlagRepository,
    RedisModelEndpointCacheRepository,
)
from spellbook_serve.infra.services import LiveEndpointBuilderService
from spellbook_serve.service_builder.celery import service_builder_service

# Need to disable lazy loading of k8s clients because each event loop should contain its own k8s
# client, which constructs the aiohttp.ClientSession in the event loop.
set_lazy_load_kubernetes_clients(False)


async def _build_endpoint(
    build_endpoint_request: BuildEndpointRequest,
) -> BuildEndpointResponse:
    session = SessionAsyncNullPool
    pool = aioredis.BlockingConnectionPool.from_url(hmi_config.cache_redis_url)
    redis = aioredis.Redis(connection_pool=pool)
    sqs_delegate: SQSEndpointResourceDelegate
    notification_gateway: NotificationGateway
    if CIRCLECI:
        sqs_delegate = FakeSQSEndpointResourceDelegate()
    else:
        sqs_delegate = LiveSQSEndpointResourceDelegate(
            sqs_profile=os.getenv("SQS_PROFILE", hmi_config.sqs_profile)
        )
    monitoring_metrics_gateway: MonitoringMetricsGateway
    monitoring_metrics_gateway = FakeMonitoringMetricsGateway()
    notification_gateway = FakeNotificationGateway()

    service = LiveEndpointBuilderService(
        docker_repository=ECRDockerRepository(),
        resource_gateway=LiveEndpointResourceGateway(sqs_delegate=sqs_delegate),
        monitoring_metrics_gateway=monitoring_metrics_gateway,
        model_endpoint_record_repository=DbModelEndpointRecordRepository(
            monitoring_metrics_gateway=monitoring_metrics_gateway, session=session, read_only=False
        ),
        model_endpoint_cache_repository=RedisModelEndpointCacheRepository(redis_client=redis),
        filesystem_gateway=S3FilesystemGateway(),
        notification_gateway=notification_gateway,
        feature_flag_repo=RedisFeatureFlagRepository(redis_client=redis),
    )
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
