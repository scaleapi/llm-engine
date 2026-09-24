import asyncio
from datetime import datetime, timezone
from typing import Optional, Set

import requests
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.gateways import EndpointTrafficGateway, TrafficKey

logger = make_logger(logger_name())

# The only Istio request metric this cluster's Prometheus keeps (see its metric_relabel_configs).
# Readiness probes bypass the sidecar, so this counts real requests only.
_QUERY = (
    "sum by (destination_workload) (increase(istio_request_duration_milliseconds_count"
    '{reporter="destination", destination_workload=~"%s.*"}[%ds]))'
)


class PrometheusEndpointTrafficGateway(EndpointTrafficGateway):
    key = TrafficKey.DEPLOYMENT_NAME

    def __init__(self, server_address: str, workload_prefix: str = "launch-endpoint-id-"):
        self.server_address = server_address.rstrip("/")
        self.workload_prefix = workload_prefix

    async def active_keys(self, since: datetime) -> Optional[Set[str]]:
        window = int((datetime.now(timezone.utc) - since).total_seconds())
        try:
            response = await asyncio.to_thread(
                requests.get,
                f"{self.server_address}/api/v1/query",
                params={"query": _QUERY % (self.workload_prefix, window)},
                timeout=60,
            )
            response.raise_for_status()
            body = response.json()
        except (requests.RequestException, ValueError):
            logger.exception("Prometheus traffic query failed")
            return None
        if body.get("status") != "success":
            logger.error(f"Prometheus traffic query returned {body.get('status')}: {body}")
            return None
        active: Set[str] = set()
        for result in body["data"]["result"]:
            workload = result["metric"].get("destination_workload")
            if workload and float(result["value"][1]) > 0:
                active.add(workload)
        return active
