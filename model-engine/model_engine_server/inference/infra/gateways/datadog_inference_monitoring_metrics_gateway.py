from datadog import statsd
from model_engine_server.inference.domain.gateways.inference_monitoring_metrics_gateway import (
    InferenceMonitoringMetricsGateway,
)


class DatadogInferenceMonitoringMetricsGateway(InferenceMonitoringMetricsGateway):
    def emit_attempted_post_inference_hook(self, hook: str):
        statsd.increment(f"scale_launch.post_inference_hook.{hook}.attempt")

    def emit_successful_post_inference_hook(self, hook: str):
        statsd.increment(f"scale_launch.post_inference_hook.{hook}.success")

    def emit_async_task_received_metric(self, queue_name: str):
        statsd.increment(
            "scale_launch.async_task.received.count", tags=[f"queue_name:{queue_name}"]
        )

    def emit_async_task_stuck_metric(self, queue_name: str):
        statsd.increment("scale_launch.async_task.stuck.count", tags=[f"queue_name:{queue_name}"])
