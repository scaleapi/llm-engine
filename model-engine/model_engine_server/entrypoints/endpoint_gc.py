# Entrypoint for the endpoint garbage collector. Meant to run as a CronJob: one pass over every
# endpoint, bookkeeping in endpoint_metadata, optional scale-to-zero and delete actions, one
# digest, exit.
#
# Actions only happen with --act. Without it the run still records observations so that the
# clocks advance during an observe-only rollout.

import argparse
import asyncio
import dataclasses
import os
from contextlib import asynccontextmanager
from typing import List

from kubernetes import config as kube_config
from kubernetes.config.config_exception import ConfigException
from model_engine_server.api.dependencies import (
    get_external_interfaces,
    get_monitoring_metrics_gateway,
)
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.db.base import get_session_async_null_pool
from model_engine_server.domain.gateways import EndpointTrafficGateway
from model_engine_server.infra.gateways.datadog_endpoint_traffic_gateway import (
    DatadogEndpointTrafficGateway,
)
from model_engine_server.infra.gateways.prometheus_endpoint_traffic_gateway import (
    PrometheusEndpointTrafficGateway,
)
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


def _build_traffic_gateways() -> List[EndpointTrafficGateway]:
    gateways: List[EndpointTrafficGateway] = []
    prometheus = infra_config().prometheus_server_address
    if prometheus:
        gateways.append(PrometheusEndpointTrafficGateway(server_address=prometheus))
    api_key, app_key = os.getenv("DD_API_KEY"), os.getenv("DD_APP_KEY")
    if api_key and app_key:
        gateways.append(
            DatadogEndpointTrafficGateway(
                api_key=api_key,
                app_key=app_key,
                env=os.environ["DD_ENV"],
                site=os.getenv("DD_SITE", "datadoghq.com"),
            )
        )
    if not gateways:
        raise RuntimeError(
            "Endpoint GC needs at least one traffic source: infra prometheus_server_address "
            "or DD_API_KEY + DD_APP_KEY"
        )
    return gateways


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
            traffic_gateways=_build_traffic_gateways(),
            model_endpoint_service=external_interfaces.model_endpoint_service,
            digest_gateway=build_digest_gateway(
                bot_token=os.getenv("ENDPOINT_GC_SLACK_BOT_TOKEN"),
                channel=os.getenv("ENDPOINT_GC_SLACK_CHANNEL"),
            ),
            config=dataclasses.replace(
                config,
                http_scale_to_zero_supported=(
                    external_interfaces.model_endpoint_service.can_scale_http_endpoint_from_zero()
                ),
            ),
        )
        report = await service.execute()
    logger.info(
        f"Endpoint GC finished: scaled_to_zero={len(report.scaled_to_zero)} "
        f"deleted={len(report.deleted)} failed={len(report.action_failed)} "
        f"deferred={len(report.deferred)} tracking={len(report.tracking)} "
        f"sources_unknown={sorted(set(report.sources_unknown))}"
    )
    if (
        report.action_failed
        or report.check_failed
        or report.judge_failed
        or report.sources_unknown
        or not report.digest_delivered
    ):
        # A failed job is the only signal besides the digest; make outages and lost notices show.
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--broken-scale-to-zero-days", type=int, default=30)
    parser.add_argument("--broken-delete-days", type=int, default=90)
    parser.add_argument("--idle-scale-to-zero-days", type=int, default=90)
    parser.add_argument("--idle-delete-days", type=int, default=180)
    parser.add_argument("--action-cap", type=int, default=20)
    parser.add_argument(
        "--act",
        action="store_true",
        help="Scale to zero and delete eligible endpoints. Without it the run only records.",
    )
    args = parser.parse_args()
    asyncio.run(
        main(
            EndpointGcConfig(
                broken_scale_to_zero_days=args.broken_scale_to_zero_days,
                broken_delete_days=args.broken_delete_days,
                idle_scale_to_zero_days=args.idle_scale_to_zero_days,
                idle_delete_days=args.idle_delete_days,
                action_cap=args.action_cap,
                actions_enabled=args.act,
            )
        )
    )
