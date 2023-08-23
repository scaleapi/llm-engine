from abc import ABC, abstractmethod
from typing import Dict


class UsageMetricsGateway(ABC):
    """
    Base class for gateway that emits usage metrics to some store of metrics, e.g. Datadog or
        Platform Money Infra.

    Inside inference/ because otherwise we import tons of stuff (in particular hmi_config) that
        isn't safe to import inside of the inference code (since it contains sensitive data)

    TODO this code (at least in its current form) should be considered temporary, it's to enable
        instantml billing
    """

    @abstractmethod
    def emit_task_call_metric(self, idempotency_token: str, tags: Dict[str, str]):
        """
        Emits the billing event to the billing queue
        Args:
            idempotency_token: Some per-request token
            tags: User-defined tags to get passed to billing. Should be for internal only.
                Right now `tags` is pretty strictly formatted,
                and reflects the scale FinancialEvent schema (see EventbridgeUsageMetricsGateway)

        """
        pass
