# This file is the entrypoint to the k8s cacher, which reads data from the k8s apiserver
#   and sticks it inside of redis. This is to work around an issue where requests from Gateway
#   pods to the k8s apiserver seem to time out a lot.

import argparse
import asyncio
import os
import time
from typing import Any

from kubernetes import config as kube_config
from kubernetes.config.config_exception import ConfigException
from model_engine_server.api.dependencies import get_monitoring_metrics_gateway
from model_engine_server.common.config import hmi_config
from model_engine_server.common.constants import READYZ_FPATH
from model_engine_server.common.env_vars import CIRCLECI
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.db.base import get_session_async_null_pool
from model_engine_server.domain.repositories import DockerRepository
from model_engine_server.infra.gateways.resources.asb_queue_endpoint_resource_delegate import (
    ASBQueueEndpointResourceDelegate,
)
from model_engine_server.infra.gateways.resources.endpoint_resource_gateway import (
    EndpointResourceGateway,
)
from model_engine_server.infra.gateways.resources.fake_queue_endpoint_resource_delegate import (
    FakeQueueEndpointResourceDelegate,
)
from model_engine_server.infra.gateways.resources.image_cache_gateway import ImageCacheGateway
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
    ECRDockerRepository,
    FakeDockerRepository,
)
from model_engine_server.infra.repositories.db_model_endpoint_record_repository import (
    DbModelEndpointRecordRepository,
)
from model_engine_server.infra.repositories.model_endpoint_cache_repository import (
    ModelEndpointCacheRepository,
)
from model_engine_server.infra.repositories.model_endpoint_record_repository import (
    ModelEndpointRecordRepository,
)
from model_engine_server.infra.repositories.redis_model_endpoint_cache_repository import (
    RedisModelEndpointCacheRepository,
)
from model_engine_server.infra.services.image_cache_service import ImageCacheService
from model_engine_server.infra.services.model_endpoint_cache_service import (
    ModelEndpointCacheWriteService,
)

logger = make_logger(logger_name())
# This is the entrypoint to the k8s cacher

try:
    kube_config.load_incluster_config()
except ConfigException:
    kube_config.load_kube_config()


async def loop_iteration(
    cache_repo: ModelEndpointCacheRepository,
    k8s_resource_manager: EndpointResourceGateway,
    endpoint_record_repo: ModelEndpointRecordRepository,
    image_cache_gateway: ImageCacheGateway,
    docker_repository: DockerRepository,
    ttl_seconds: float,
):
    image_cache_service = ImageCacheService(
        model_endpoint_record_repository=endpoint_record_repo,
        image_cache_gateway=image_cache_gateway,
        docker_repository=docker_repository,
    )
    cache_write_service = ModelEndpointCacheWriteService(
        cache_repo, k8s_resource_manager, image_cache_service
    )
    await cache_write_service.execute(ttl_seconds=ttl_seconds)


async def main(args: Any):
    assert (
        args.ttl_seconds > 0 and args.sleep_interval_seconds > 0
    ), "TTL + polling interval must be positive"
    if args.ttl_seconds < args.sleep_interval_seconds:
        logger.warning(
            "Redis ttl is less than polling interval, cache entries will expire and cause misses"
        )
    redis_url = args.redis_url_override if args.redis_url_override else hmi_config.cache_redis_url
    logger.info(f"Using cache redis url {redis_url}")
    cache_repo = RedisModelEndpointCacheRepository(redis_info=redis_url)

    monitoring_metrics_gateway = get_monitoring_metrics_gateway()
    endpoint_record_repo = DbModelEndpointRecordRepository(
        monitoring_metrics_gateway=monitoring_metrics_gateway,
        session=get_session_async_null_pool(),
        read_only=True,
    )

    queue_delegate: QueueEndpointResourceDelegate
    if CIRCLECI:
        queue_delegate = FakeQueueEndpointResourceDelegate()
    elif infra_config().cloud_provider == "azure":
        queue_delegate = ASBQueueEndpointResourceDelegate()
    else:
        queue_delegate = SQSQueueEndpointResourceDelegate(
            sqs_profile=os.getenv("SQS_PROFILE", hmi_config.sqs_profile)
        )

    k8s_resource_manager = LiveEndpointResourceGateway(
        queue_delegate=queue_delegate,
        inference_autoscaling_metrics_gateway=None,
    )
    image_cache_gateway = ImageCacheGateway()
    docker_repo: DockerRepository
    if CIRCLECI:
        docker_repo = FakeDockerRepository()
    elif infra_config().cloud_provider == "onprem":
        docker_repo = FakeDockerRepository()
    elif infra_config().docker_repo_prefix.endswith("azurecr.io"):
        docker_repo = ACRDockerRepository()
    else:
        docker_repo = ECRDockerRepository()
    while True:
        loop_start = time.time()
        await loop_iteration(
            cache_repo,
            k8s_resource_manager,
            endpoint_record_repo,
            image_cache_gateway,
            docker_repo,
            args.ttl_seconds,
        )
        loop_end = time.time()
        loop_duration = loop_end - loop_start
        logger.info(f"Loop took {loop_duration} seconds")
        if loop_duration < args.sleep_interval_seconds:
            logger.info(
                f"Loop took {loop_duration} seconds, sleeping for "
                f"{args.sleep_interval_seconds - loop_duration} seconds"
            )
            await asyncio.sleep(args.sleep_interval_seconds - loop_duration)

        # k8s health check
        if not os.path.exists(READYZ_FPATH):
            with open(READYZ_FPATH, "w") as f:
                f.write("READY")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ttl-seconds", type=int, default=60)
    parser.add_argument("--sleep-interval-seconds", type=int, default=15)
    parser.add_argument("--redis-url-override", type=str, default=None)
    main_args = parser.parse_args()
    asyncio.run(main(main_args))
