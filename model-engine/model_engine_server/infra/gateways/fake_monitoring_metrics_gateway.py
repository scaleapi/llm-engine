from collections import defaultdict

from model_engine_server.common.dtos.llms import TokenUsage
from model_engine_server.domain.gateways.monitoring_metrics_gateway import (
    MetricMetadata,
    MonitoringMetricsGateway,
)


class FakeMonitoringMetricsGateway(MonitoringMetricsGateway):
    def __init__(self):
        self.attempted_build = 0
        self.successful_build = 0
        self.build_time_seconds = 0
        self.image_build_cache_hit = defaultdict(int)
        self.image_build_cache_miss = defaultdict(int)
        self.docker_failed_build = 0
        self.attempted_hook = defaultdict(int)
        self.successful_hook = defaultdict(int)
        self.database_cache_hit = 0
        self.database_cache_miss = 0
        self.route_call = defaultdict(int)
        self.token_count = 0
        self.total_tokens_per_second = 0
        self.sync_call_timeout = defaultdict(int)

    def reset(self):
        self.attempted_build = 0
        self.successful_build = 0
        self.build_time_seconds = 0
        self.image_build_cache_hit = defaultdict(int)
        self.image_build_cache_miss = defaultdict(int)
        self.docker_failed_build = 0
        self.attempted_hook = defaultdict(int)
        self.successful_hook = defaultdict(int)
        self.database_cache_hit = 0
        self.database_cache_miss = 0
        self.route_call = defaultdict(int)
        self.token_count = 0
        self.total_tokens_per_second = 0
        self.sync_call_timeout = defaultdict(int)

    def emit_attempted_build_metric(self):
        self.attempted_build += 1

    def emit_successful_build_metric(self):
        self.successful_build += 1

    def emit_build_time_metric(self, duration_seconds: float):
        self.build_time_seconds += duration_seconds

    def emit_image_build_cache_hit_metric(self, image_type: str):
        self.image_build_cache_hit[image_type] += 1

    def emit_image_build_cache_miss_metric(self, image_type: str):
        self.image_build_cache_miss[image_type] += 1

    def emit_docker_failed_build_metric(self):
        self.docker_failed_build += 1

    def emit_attempted_post_inference_hook(self, hook: str):
        self.attempted_hook[hook] += 1

    def emit_successful_post_inference_hook(self, hook: str):
        self.successful_hook[hook] += 1

    def emit_database_cache_hit_metric(self):
        self.database_cache_hit += 1

    def emit_database_cache_miss_metric(self):
        self.database_cache_miss += 1

    def emit_route_call_metric(self, route: str, _metadata: MetricMetadata):
        self.route_call[route] += 1

    def emit_token_count_metrics(self, token_usage: TokenUsage, _metadata: MetricMetadata):
        self.token_count += token_usage.num_total_tokens
        self.total_tokens_per_second = token_usage.total_tokens_per_second

    def emit_http_call_error_metrics(self, endpoint_name: str, error_code: int):
        self.sync_call_timeout[(endpoint_name, error_code)] += 1
