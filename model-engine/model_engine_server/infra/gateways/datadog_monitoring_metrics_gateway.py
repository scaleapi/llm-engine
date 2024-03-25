from typing import List, Optional

from datadog import statsd
from model_engine_server.common.dtos.llms import TokenUsage
from model_engine_server.core.config import infra_config
from model_engine_server.domain.gateways.monitoring_metrics_gateway import (
    MetricMetadata,
    MonitoringMetricsGateway,
)


def get_model_tags(model_name: Optional[str]) -> List[str]:
    """
    Returns a tag for the model name and whether it is a finetuned model
    """
    tags = []
    if model_name:
        parts = model_name.split(".")
        tags.extend([f"model_name:{parts[0]}"])
    return tags


class DatadogMonitoringMetricsGateway(MonitoringMetricsGateway):
    def __init__(self, prefix: str = "model_engine"):
        self.prefix = prefix
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

    def _format_call_tags(self, metadata: MetricMetadata) -> List[str]:
        tags = self.tags
        tags.extend(get_model_tags(metadata.model_name))
        return tags

    def emit_route_call_metric(self, route: str, metadata: MetricMetadata):
        statsd.increment(f"{self.prefix}.{route}.call", tags=self._format_call_tags(metadata))

    def emit_token_count_metrics(self, token_usage: TokenUsage, metadata: MetricMetadata):
        tags = self._format_call_tags(metadata)

        token_count_metric = f"{self.prefix}.token_count"
        statsd.increment(
            f"{token_count_metric}.prompt", (token_usage.num_prompt_tokens or 0), tags=tags
        )
        statsd.increment(
            f"{token_count_metric}.completion", (token_usage.num_completion_tokens or 0), tags=tags
        )
        statsd.increment(f"{token_count_metric}.total", token_usage.num_total_tokens, tags=tags)

        total_tokens_per_second = f"{self.prefix}.total_tokens_per_second"
        statsd.histogram(total_tokens_per_second, token_usage.total_tokens_per_second, tags=tags)

        time_to_first_token = f"{self.prefix}.time_to_first_token"
        if token_usage.time_to_first_token is not None:
            statsd.histogram(time_to_first_token, token_usage.time_to_first_token, tags=tags)

        inter_token_latency = f"{self.prefix}.inter_token_latency"
        if token_usage.inter_token_latency is not None:
            statsd.histogram(inter_token_latency, token_usage.inter_token_latency, tags=tags)
