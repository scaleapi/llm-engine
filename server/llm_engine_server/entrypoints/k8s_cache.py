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
from llm_engine_server.common.config import hmi_config
from llm_engine_server.common.constants import READYZ_FPATH
from llm_engine_server.common.env_vars import CIRCLECI
from llm_engine_server.core.loggers import filename_wo_ext, make_logger
from llm_engine_server.db.base import SessionAsyncNullPool
from llm_engine_server.domain.repositories import DockerRepository
from llm_engine_server.infra.gateways import FakeMonitoringMetricsGateway
from llm_engine_server.infra.gateways.resources.endpoint_resource_gateway import (
    EndpointResourceGateway,
)
from llm_engine_server.infra.gateways.resources.fake_sqs_endpoint_resource_delegate import (
    FakeSQSEndpointResourceDelegate,
)
from llm_engine_server.infra.gateways.resources.image_cache_gateway import ImageCacheGateway
from llm_engine_server.infra.gateways.resources.live_endpoint_resource_gateway import (
    LiveEndpointResourceGateway,
)
from llm_engine_server.infra.gateways.resources.live_sqs_endpoint_resource_delegate import (
    LiveSQSEndpointResourceDelegate,
)
from llm_engine_server.infra.gateways.resources.sqs_endpoint_resource_delegate import (
    SQSEndpointResourceDelegate,
)
from llm_engine_server.infra.repositories import ECRDockerRepository
from llm_engine_server.infra.repositories.db_model_endpoint_record_repository import (
    DbModelEndpointRecordRepository,
)
from llm_engine_server.infra.repositories.model_endpoint_cache_repository import (
    ModelEndpointCacheRepository,
)
from llm_engine_server.infra.repositories.model_endpoint_record_repository import (
    ModelEndpointRecordRepository,
)
from llm_engine_server.infra.repositories.redis_model_endpoint_cache_repository import (
    RedisModelEndpointCacheRepository,
)
from llm_engine_server.infra.services.image_cache_service import ImageCacheService
from llm_engine_server.infra.services.model_endpoint_cache_service import (
    ModelEndpointCacheWriteService,
)

logger = make_logger(filename_wo_ext(__file__))
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

    monitoring_metrics_gateway = FakeMonitoringMetricsGateway()
    endpoint_record_repo = DbModelEndpointRecordRepository(
        monitoring_metrics_gateway=monitoring_metrics_gateway,
        session=SessionAsyncNullPool,
        read_only=True,
    )
    sqs_delegate: SQSEndpointResourceDelegate
    if CIRCLECI:
        sqs_delegate = FakeSQSEndpointResourceDelegate()
    else:
        sqs_delegate = LiveSQSEndpointResourceDelegate(
            sqs_profile=os.getenv("SQS_PROFILE", hmi_config.sqs_profile)
        )

    k8s_resource_manager = LiveEndpointResourceGateway(sqs_delegate=sqs_delegate)
    image_cache_gateway = ImageCacheGateway()
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
