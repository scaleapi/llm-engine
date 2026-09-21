# Entrypoint for the endpoint garbage collector. Meant to run as a CronJob: one pass over every
# endpoint, bookkeeping in endpoint_metadata, optional deletions, one digest, exit.
#
# Deletions only happen with --delete. Without it the run still records observations and flags
# so that the unavailable-days clock advances during an observe-only rollout.

import argparse
import asyncio
import os
from contextlib import asynccontextmanager

from kubernetes import config as kube_config
from kubernetes.config.config_exception import ConfigException
from model_engine_server.api.dependencies import (
    get_external_interfaces,
    get_monitoring_metrics_gateway,
)
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.db.base import get_session_async_null_pool
from model_engine_server.infra.gateways.resources.live_endpoint_resource_gateway import (
    LiveEndpointResourceGateway,
)
from model_engine_server.infra.gateways.slack_digest_gateway import build_digest_gateway
from model_engine_server.infra.repositories.db_model_endpoint_record_repository import (
    DbModelEndpointRecordRepository,
)
from model_engine_server.infra.services.endpoint_gc_service import (
    EndpointGarbageCollectionService,
    EndpointGcConfig,
)

logger = make_logger(logger_name())

try:
    kube_config.load_incluster_config()
except ConfigException:
    kube_config.load_kube_config()


async def main(config: EndpointGcConfig) -> None:
    async with asynccontextmanager(get_external_interfaces)() as external_interfaces:
        resource_gateway = external_interfaces.resource_gateway
        if not isinstance(resource_gateway, LiveEndpointResourceGateway):
            raise TypeError("Endpoint GC needs the live resource gateway for queue activity")
        service = EndpointGarbageCollectionService(
            # ExternalInterfaces does not expose the record repository; build one on the same
            # null-pool session type the k8s cacher uses.
            model_endpoint_record_repository=DbModelEndpointRecordRepository(
                monitoring_metrics_gateway=get_monitoring_metrics_gateway(),
                session=get_session_async_null_pool(),
                read_only=False,
            ),
            resource_gateway=resource_gateway,
            queue_delegate=resource_gateway.queue_delegate,
            model_endpoint_service=external_interfaces.model_endpoint_service,
            digest_gateway=build_digest_gateway(
                bot_token=os.getenv("ENDPOINT_GC_SLACK_BOT_TOKEN"),
                channel=os.getenv("ENDPOINT_GC_SLACK_CHANNEL"),
            ),
            config=config,
        )
        report = await service.execute()
    logger.info(
        f"Endpoint GC finished: deleted={len(report.deleted)} failed={len(report.delete_failed)} "
        f"deferred={len(report.delete_deferred)} flagged={len(report.flagged_new)} "
        f"in_grace={len(report.in_grace)} observing={len(report.observing)}"
    )
    if report.delete_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unavailable-days", type=int, default=30)
    parser.add_argument("--grace-days", type=int, default=14)
    parser.add_argument("--delete-cap", type=int, default=20)
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete eligible endpoints. Without it the run only records and reports.",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            EndpointGcConfig(
                unavailable_days=args.unavailable_days,
                grace_days=args.grace_days,
                delete_cap=args.delete_cap,
                delete_enabled=args.delete,
            )
        )
    )
