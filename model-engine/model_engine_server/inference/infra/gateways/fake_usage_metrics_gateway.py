from typing import Dict

from model_engine_server.inference.domain.gateways.usage_metrics_gateway import UsageMetricsGateway


class FakeUsageMetricsGateway(UsageMetricsGateway):
    """No-op usage metrics emitter"""

    def emit_task_call_metric(self, idempotency_token: str, tags: Dict[str, str]):
        pass
