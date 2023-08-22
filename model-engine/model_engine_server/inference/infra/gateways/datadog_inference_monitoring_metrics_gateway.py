from datadog import statsd
from model_engine_server.inference.domain.gateways.inference_monitoring_metrics_gateway import (
    InferenceMonitoringMetricsGateway,
)


class DatadogInferenceMonitoringMetricsGateway(InferenceMonitoringMetricsGateway):
    def emit_attempted_post_inference_hook(self, hook: str):
        statsd.increment(f"scale_launch.post_inference_hook.{hook}.attempt")

    def emit_successful_post_inference_hook(self, hook: str):
        statsd.increment(f"scale_launch.post_inference_hook.{hook}.success")
