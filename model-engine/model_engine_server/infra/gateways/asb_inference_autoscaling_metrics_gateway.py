import os

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.servicebus import ServiceBusClient, ServiceBusMessage
from azure.servicebus.management import ServiceBusAdministrationClient
from model_engine_server.domain.gateways.inference_autoscaling_metrics_gateway import (
    InferenceAutoscalingMetricsGateway,
)

EXPIRY_SECONDS = 60  # 1 minute; this gets added to the cooldown time present in the keda ScaledObject to get total
# scaledown time. This also needs to be larger than the keda ScaledObject's refresh rate.
PREWARM_EXPIRY_SECONDS = 60 * 60  # 1 hour


def _get_servicebus_administration_client() -> ServiceBusAdministrationClient:
    return ServiceBusAdministrationClient(
        f"{os.getenv('SERVICEBUS_NAMESPACE')}.servicebus.windows.net",
        credential=DefaultAzureCredential(),
    )


class ASBInferenceAutoscalingMetricsGateway(InferenceAutoscalingMetricsGateway):
    @staticmethod
    def _find_queue_name(endpoint_id: str):
        # Keep in line with keda scaled object yaml
        return f"launch-endpoint-autoscaling.{endpoint_id}"

    async def _emit_metric(self, endpoint_id: str, expiry_time: int):
        queue_name = self._find_queue_name(endpoint_id)

        servicebus_namespace = os.getenv("SERVICEBUS_NAMESPACE")
        if servicebus_namespace is None:
            raise ValueError("SERVICEBUS_NAMESPACE env var must be set in Azure")

        with ServiceBusClient(
            fully_qualified_namespace=f"{servicebus_namespace}.servicebus.windows.net",
            credential=DefaultAzureCredential(),
        ) as servicebus_client:
            sender = servicebus_client.get_queue_sender(queue_name=queue_name)
            with sender:
                message = ServiceBusMessage(
                    "message"
                )  # we only care about the length of the queue, not the message values
                sender.send_messages(message=message, timeout=expiry_time)

    async def emit_inference_autoscaling_metric(self, endpoint_id: str):
        await self._emit_metric(endpoint_id, EXPIRY_SECONDS)

    async def emit_prewarm_metric(self, endpoint_id: str):
        await self._emit_metric(endpoint_id, PREWARM_EXPIRY_SECONDS)

    async def create_or_update_resources(self, endpoint_id: str):
        queue_name = self._find_queue_name(endpoint_id)
        with _get_servicebus_administration_client() as client:
            try:
                client.create_queue(queue_name=queue_name)
            except ResourceExistsError:
                pass

    async def delete_resources(self, endpoint_id: str):
        queue_name = self._find_queue_name(endpoint_id)
        with _get_servicebus_administration_client() as client:
            try:
                client.delete_queue(queue_name=queue_name)
            except ResourceNotFoundError:
                pass
