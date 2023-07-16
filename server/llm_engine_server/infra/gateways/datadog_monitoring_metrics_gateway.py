from datadog import statsd

from llm_engine_server.core.config import ml_infra_config
from llm_engine_server.domain.gateways.monitoring_metrics_gateway import MonitoringMetricsGateway


class DatadogMonitoringMetricsGateway(MonitoringMetricsGateway):
    def __init__(self):
        self.tags = [f"env:{ml_infra_config().env}"]

    def emit_attempted_build_metric(self):
        statsd.increment("scale_llm_engine_server.service_builder.attempt", tags=self.tags)

    def emit_successful_build_metric(self):
        statsd.increment("scale_llm_engine_server.service_builder.success", tags=self.tags)

    def emit_docker_failed_build_metric(self):
        statsd.increment("scale_llm_engine_server.service_builder.docker_failed", tags=self.tags)

    def emit_database_cache_hit_metric(self):
        statsd.increment("scale_llm_engine_server.database_cache.hit", tags=self.tags)

    def emit_database_cache_miss_metric(self):
        statsd.increment("scale_llm_engine_server.database_cache.miss", tags=self.tags)
