from collections import defaultdict

from llm_engine_server.domain.gateways import MonitoringMetricsGateway


class FakeMonitoringMetricsGateway(MonitoringMetricsGateway):
    def __init__(self):
        self.attempted_build = 0
        self.successful_build = 0
        self.docker_failed_build = 0
        self.attempted_hook = defaultdict(int)
        self.successful_hook = defaultdict(int)
        self.database_cache_hit = 0
        self.database_cache_miss = 0

    def reset(self):
        self.attempted_build = 0
        self.successful_build = 0
        self.docker_failed_build = 0
        self.attempted_hook = defaultdict(int)
        self.successful_hook = defaultdict(int)
        self.database_cache_hit = 0
        self.database_cache_miss = 0

    def emit_attempted_build_metric(self):
        self.attempted_build += 1

    def emit_successful_build_metric(self):
        self.successful_build += 1

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
