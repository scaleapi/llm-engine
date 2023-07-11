from datadog import statsd

from spellbook_serve.inference.domain.gateways.inference_monitoring_metrics_gateway import (
    InferenceMonitoringMetricsGateway,
)


class DatadogInferenceMonitoringMetricsGateway(InferenceMonitoringMetricsGateway):
    def emit_attempted_post_inference_hook(self, hook: str):
        statsd.increment(f"scale_spellbook_serve.post_inference_hook.{hook}.attempt")

    def emit_successful_post_inference_hook(self, hook: str):
        statsd.increment(f"scale_spellbook_serve.post_inference_hook.{hook}.success")
