# Entrypoint for the endpoint garbage collector. Meant to run as a CronJob: one pass over every
# endpoint, bookkeeping in endpoint_metadata, optional deletions, one digest, exit.
#
# Deletions only happen with --delete. Without it the run still records observations and flags
# so that the 30 day clock advances during a dry-run rollout.

import argparse
import asyncio
import os

from kubernetes import config as kube_config
from kubernetes.config.config_exception import ConfigException
from model_engine_server.api.dependencies import (
    ExternalInterfaces,
    get_default_external_interfaces,
    get_monitoring_metrics_gateway,
)
from model_engine_server.common.config import hmi_config
from model_engine_server.common.env_vars import CIRCLECI
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.db.base import get_session_async_null_pool
from model_engine_server.infra.gateways.slack_digest_gateway import build_digest_gateway
from model_engine_server.infra.gateways.sqs_queue_activity_gateway import (
    SQSQueueActivityGateway,
    UnknownQueueActivityGateway,
)
from model_engine_server.infra.repositories.db_model_endpoint_record_repository import (
    DbModelEndpointRecordRepository,
)
from model_engine_server.infra.services.endpoint_gc_service import (
    EndpointGarbageCollectionService,
    EndpointGcConfig,
    QueueActivityGateway,
)

logger = make_logger(logger_name())

try:
    kube_config.load_incluster_config()
except ConfigException:
    kube_config.load_kube_config()


def _get_external_interfaces() -> ExternalInterfaces:
    # Same plugin override as the gateway's dependency, without the FastAPI generator wrapper.
    try:
        from plugins.dependencies import get_external_interfaces as get_custom_external_interfaces

        return get_custom_external_interfaces()
    except ModuleNotFoundError:
        return get_default_external_interfaces()


def _build_queue_activity_gateway() -> QueueActivityGateway:
    if CIRCLECI or infra_config().cloud_provider != "aws":
        return UnknownQueueActivityGateway()
    return SQSQueueActivityGateway(sqs_profile=os.getenv("SQS_PROFILE", hmi_config.sqs_profile))


async def main(config: EndpointGcConfig) -> None:
    external_interfaces = _get_external_interfaces()
    record_repository = DbModelEndpointRecordRepository(
        monitoring_metrics_gateway=get_monitoring_metrics_gateway(),
        session=get_session_async_null_pool(),
        read_only=False,
    )
    try:
        service = EndpointGarbageCollectionService(
            model_endpoint_record_repository=record_repository,
            resource_gateway=external_interfaces.resource_gateway,
            model_endpoint_service=external_interfaces.model_endpoint_service,
            queue_activity_gateway=_build_queue_activity_gateway(),
            digest_gateway=build_digest_gateway(
                bot_token=os.getenv("ENDPOINT_GC_SLACK_BOT_TOKEN"),
                channel=os.getenv("ENDPOINT_GC_SLACK_CHANNEL"),
            ),
            config=config,
        )
        try:
            report = await service.execute()
        except Exception:
            logger.exception("Endpoint GC run failed before completing")
            raise SystemExit(1)
    finally:
        await external_interfaces.file_storage_gateway.close()
    logger.info(
        f"Endpoint GC finished: deleted={len(report.deleted)} failed={len(report.delete_failed)} "
        f"deferred={len(report.delete_deferred)} flagged={len(report.flagged_new)} "
        f"in_grace={len(report.in_grace)} observing={len(report.observing) + len(report.observing_new)}"
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
