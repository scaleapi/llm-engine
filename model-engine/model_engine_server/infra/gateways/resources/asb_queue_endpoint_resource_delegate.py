import os
from typing import Any, Dict

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.servicebus.management import ServiceBusAdministrationClient
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.exceptions import EndpointResourceInfraException
from model_engine_server.infra.gateways.resources.queue_endpoint_resource_delegate import (
    QueueEndpointResourceDelegate,
    QueueInfo,
)

logger = make_logger(logger_name())


def _get_servicebus_administration_client() -> ServiceBusAdministrationClient:
    return ServiceBusAdministrationClient(
        f"{os.getenv('SERVICEBUS_NAMESPACE')}.servicebus.windows.net",
        credential=DefaultAzureCredential(),
    )


class ASBQueueEndpointResourceDelegate(QueueEndpointResourceDelegate):
    """
    Using Azure Service Bus.
    """

    async def create_queue_if_not_exists(
        self,
        endpoint_id: str,
        endpoint_name: str,
        endpoint_created_by: str,
        endpoint_labels: Dict[str, Any],
    ) -> QueueInfo:
        queue_name = QueueEndpointResourceDelegate.endpoint_id_to_queue_name(endpoint_id)
        with _get_servicebus_administration_client() as client:
            try:
                client.create_queue(queue_name=queue_name)
            except ResourceExistsError:
                pass

            return QueueInfo(queue_name, None)

    async def delete_queue(self, endpoint_id: str) -> None:
        queue_name = QueueEndpointResourceDelegate.endpoint_id_to_queue_name(endpoint_id)
        with _get_servicebus_administration_client() as client:
            try:
                client.delete_queue(queue_name=queue_name)
            except ResourceNotFoundError:
                logger.info(f"Could not find ASB queue {queue_name} for endpoint {endpoint_id}")

    async def get_queue_attributes(self, endpoint_id: str) -> Dict[str, Any]:
        queue_name = QueueEndpointResourceDelegate.endpoint_id_to_queue_name(endpoint_id)
        with _get_servicebus_administration_client() as client:
            try:
                queue_attributes = client.get_queue_runtime_properties(queue_name=queue_name)
            except ResourceNotFoundError as e:
                raise EndpointResourceInfraException(
                    f"Could not find ASB queue {queue_name} for endpoint {endpoint_id}"
                ) from e

            # queue_attributes does have other fields, but we don't need them right now
            return {
                "name": queue_attributes.name,
                "total_message_count": queue_attributes.total_message_count,
                "active_message_count": queue_attributes.active_message_count,
            }
