import os
from typing import Any, Dict

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.servicebus.management import ServiceBusAdministrationClient
from model_engine_server.domain.exceptions import EndpointResourceInfraException
from model_engine_server.infra.gateways.resources.queue_endpoint_resource_delegate import (
    QueueEndpointResourceDelegate,
    QueueInfo,
)


def _get_servicebus_administration_client() -> ServiceBusAdministrationClient:
    conn_str = os.getenv("SERVICEBUS_CONNECTION_STRING")
    if conn_str is None:
        raise ValueError("SERVICEBUS_CONNECTION_STRING env var is required")
    return ServiceBusAdministrationClient.from_connection_string(conn_str=conn_str)


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
            except ResourceNotFoundError as e:
                raise EndpointResourceInfraException(
                    f"Could not find ASB queue {queue_name} for endpoint {endpoint_id}"
                ) from e

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
