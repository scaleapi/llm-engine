from datadog import statsd
from model_engine_server.core.config import infra_config
from model_engine_server.domain.gateways.monitoring_metrics_gateway import MonitoringMetricsGateway


class DatadogMonitoringMetricsGateway(MonitoringMetricsGateway):
    def __init__(self):
        self.tags = [f"env:{infra_config().env}"]

    def emit_attempted_build_metric(self):
        statsd.increment("scale_launch.service_builder.attempt", tags=self.tags)

    def emit_successful_build_metric(self):
        statsd.increment("scale_launch.service_builder.success", tags=self.tags)

    def emit_build_time_metric(self, duration_seconds: float):
        statsd.distribution(
            "scale_launch.service_builder.endpoint_build_time", duration_seconds, tags=self.tags
        )

    def emit_image_build_cache_hit_metric(self, image_type: str):
        statsd.increment(
            f"scale_launch.service_builder.{image_type}_image_cache_hit", tags=self.tags
        )

    def emit_image_build_cache_miss_metric(self, image_type: str):
        statsd.increment(
            f"scale_launch.service_builder.{image_type}_image_cache_miss", tags=self.tags
        )

    def emit_docker_failed_build_metric(self):
        statsd.increment("scale_launch.service_builder.docker_failed", tags=self.tags)

    def emit_database_cache_hit_metric(self):
        statsd.increment("scale_launch.database_cache.hit", tags=self.tags)

    def emit_database_cache_miss_metric(self):
        statsd.increment("scale_launch.database_cache.miss", tags=self.tags)
