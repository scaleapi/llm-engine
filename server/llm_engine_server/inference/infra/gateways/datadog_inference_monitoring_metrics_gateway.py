from datadog import statsd
from llm_engine_server.inference.domain.gateways.inference_monitoring_metrics_gateway import (
    InferenceMonitoringMetricsGateway,
)


class DatadogInferenceMonitoringMetricsGateway(InferenceMonitoringMetricsGateway):
    def emit_attempted_post_inference_hook(self, hook: str):
        statsd.increment(f"scale_llm_engine_server.post_inference_hook.{hook}.attempt")

    def emit_successful_post_inference_hook(self, hook: str):
        statsd.increment(f"scale_llm_engine_server.post_inference_hook.{hook}.success")
