import asyncio
from datetime import datetime, timezone
from typing import Dict, Optional, Set

from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v1.api.metrics_api import MetricsApi
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.gateways import EndpointTrafficGateway, TrafficKey

logger = make_logger(logger_name())

# Endpoint pods trace with DD_SERVICE set to the endpoint name. Health probes are traced too and
# are excluded by resource name, the same filter the serving audits settled on.
_PROBE_EXCLUSIONS = (
    "!resource_name:*health*,!resource_name:*readyz*,!resource_name:*livez*,!resource_name:*ping*"
)


class DatadogEndpointTrafficGateway(EndpointTrafficGateway):
    key = TrafficKey.ENDPOINT_NAME

    def __init__(self, api_key: str, app_key: str, env: str, site: str = "datadoghq.com"):
        self.configuration = Configuration()
        self.configuration.api_key["apiKeyAuth"] = api_key
        self.configuration.api_key["appKeyAuth"] = app_key
        self.configuration.server_variables["site"] = site
        self.env = env

    def _query(self, since: datetime):
        query = f"sum:trace.fastapi.request.hits{{env:{self.env},{_PROBE_EXCLUSIONS}}} by {{service}}.as_count()"
        with ApiClient(self.configuration) as api_client:
            return MetricsApi(api_client).query_metrics(
                _from=int(since.timestamp()),
                to=int(datetime.now(timezone.utc).timestamp()),
                query=query,
            )

    @staticmethod
    def _service_of(series) -> Optional[str]:
        for tag in series.tag_set or []:
            if tag.startswith("service:"):
                return tag[len("service:") :]
        return None

    async def active_keys(self, since: datetime) -> Optional[Set[str]]:
        history = await self.last_active_at(since)
        return None if history is None else set(history)

    async def last_active_at(self, since: datetime) -> Optional[Dict[str, datetime]]:
        try:
            response = await asyncio.to_thread(self._query, since)
        except Exception:
            logger.exception("Datadog traffic query failed")
            return None
        if response.status != "ok":
            logger.error(f"Datadog traffic query returned status {response.status}")
            return None
        last_seen: Dict[str, datetime] = {}
        for series in response.series or []:
            service = self._service_of(series)
            if not service:
                continue
            for point in series.pointlist or []:
                timestamp_ms, value = point.value
                if value and value > 0:
                    seen = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                    if service not in last_seen or seen > last_seen[service]:
                        last_seen[service] = seen
        return last_seen
