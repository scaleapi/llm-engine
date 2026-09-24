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
# Coverage probe: with no series at all for the prefix, the metric or the scrape is gone and an
# empty answer means "unknown", not "idle".
_COVERAGE_QUERY = (
    "count(istio_request_duration_milliseconds_count"
    '{reporter="destination", destination_workload=~"%s.*"})'
)


class PrometheusEndpointTrafficGateway(EndpointTrafficGateway):
    key = TrafficKey.DEPLOYMENT_NAME

    def __init__(self, server_address: str, workload_prefix: str = "launch-endpoint-id-"):
        self.server_address = server_address.rstrip("/")
        self.workload_prefix = workload_prefix

    async def _query(self, query: str) -> Optional[list]:
        try:
            response = await asyncio.to_thread(
                requests.get,
                f"{self.server_address}/api/v1/query",
                params={"query": query},
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
        return body["data"]["result"]

    async def active_keys(self, since: datetime) -> Optional[Set[str]]:
        coverage = await self._query(_COVERAGE_QUERY % self.workload_prefix)
        if coverage is None or not coverage or float(coverage[0]["value"][1]) <= 0:
            logger.error("Prometheus has no istio request series for endpoint workloads")
            return None
        window = int((datetime.now(timezone.utc) - since).total_seconds())
        results = await self._query(_QUERY % (self.workload_prefix, window))
        if results is None:
            return None
        active: Set[str] = set()
        for result in results:
            workload = result["metric"].get("destination_workload")
            if workload and float(result["value"][1]) > 0:
                active.add(workload)
        return active
