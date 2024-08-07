from abc import ABC, abstractmethod


class InferenceAutoscalingMetricsGateway(ABC):
    """
    Abstract Base Class for a gateway that emits autoscaling metrics for inference requests. Can be used in conjunction
    with various autoscaler resources, e.g. a Keda ScaledObject, to autoscale inference endpoints.
    """

    @abstractmethod
    async def emit_inference_autoscaling_metric(self, endpoint_id: str):
        """
        On an inference request, emit a metric
        """
        pass

    @abstractmethod
    async def emit_prewarm_metric(self, endpoint_id: str):
        """
        If you want to prewarm an endpoint, emit a metric here
        """
        pass

    @abstractmethod
    async def create_or_update_resources(self, endpoint_id: str):
        """
        Create necessary resources for autoscaling metrics
        """
        pass

    @abstractmethod
    async def delete_resources(self, endpoint_id: str):
        """
        Delete necessary resources for autoscaling metrics
        """
        pass
