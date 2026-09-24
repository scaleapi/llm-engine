import asyncio
from datetime import datetime, timezone
from typing import Dict, Optional, Set
from urllib.parse import urlparse

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
# Envoy's Prometheus endpoint, on the sidecar (15090) or merged through the agent (15020).
_ISTIO_STATS_PATH = "/stats/prometheus"
# Coverage probe: with no series at all for the prefix, the metric or the scrape is gone and an
# empty answer means "unknown", not "idle".
_COVERAGE_QUERY = (
    "count(istio_request_duration_milliseconds_count"
    '{reporter="destination", destination_workload=~"%s.*"})'
)


class PrometheusEndpointTrafficGateway(EndpointTrafficGateway):
    key = TrafficKey.DEPLOYMENT_NAME
    reports_coverage = True

    def __init__(self, server_address: str, workload_prefix: str = "launch-endpoint-id-"):
        self.server_address = server_address.rstrip("/")
        self.workload_prefix = workload_prefix

    async def _get(self, path: str, params: dict) -> Optional[dict]:
        try:
            response = await asyncio.to_thread(
                requests.get, f"{self.server_address}{path}", params=params, timeout=60
            )
            response.raise_for_status()
            body = response.json()
        except (requests.RequestException, ValueError):
            logger.exception(f"Prometheus request {path} failed")
            return None
        if body.get("status") != "success":
            logger.error(f"Prometheus request {path} returned {body.get('status')}: {body}")
            return None
        return body["data"]

    async def _query(self, query: str) -> Optional[list]:
        data = await self._get("/api/v1/query", {"query": query})
        return None if data is None else data["result"]

    async def covered_keys(self) -> Optional[Set[str]]:
        """Deployments whose every ready pod has a healthy, scraped Istio sidecar target.

        Read from the discovered (pre-relabeling) pod labels of the active scrape targets, so
        it does not depend on which labels the scrape config keeps. Endpoint pods carry
        ``app=<deployment name>``. Only sidecar targets (Envoy's ``/stats/prometheus``) carry the
        request metric; an application's own ``/metrics`` target proves nothing, and one scraped
        pod says nothing about a sibling whose sidecar is not.
        """
        data = await self._get("/api/v1/targets", {"state": "active"})
        if data is None:
            return None
        ready_pods: Dict[str, Set[str]] = {}
        observed_pods: Dict[str, Set[str]] = {}
        for target in data.get("activeTargets", []):
            labels = target.get("discoveredLabels") or {}
            app = labels.get("__meta_kubernetes_pod_label_app", "")
            pod = labels.get("__meta_kubernetes_pod_name", "")
            if not pod or not app.startswith(self.workload_prefix):
                continue
            if labels.get("__meta_kubernetes_pod_ready") != "true":
                continue  # not serving: gets no requests, needs no coverage
            ready_pods.setdefault(app, set()).add(pod)
            scrape_path = urlparse(target.get("scrapeUrl", "")).path
            if target.get("health") == "up" and scrape_path.endswith(_ISTIO_STATS_PATH):
                observed_pods.setdefault(app, set()).add(pod)
        covered = {app for app, pods in ready_pods.items() if pods <= observed_pods.get(app, set())}
        if not covered:
            logger.error("Prometheus scrapes no healthy endpoint sidecar: coverage unknown")
            return None
        return covered

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
