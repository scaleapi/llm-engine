from abc import ABC, abstractmethod
from typing import Dict, Generic, Sequence, Tuple, TypeVar

from model_engine_server.common.dtos.model_endpoints import BrokerType
from model_engine_server.common.dtos.resource_manager import CreateOrUpdateResourcesRequest
from model_engine_server.domain.entities import (
    ModelEndpointInfraState,
    ModelEndpointRecord,
    ModelEndpointType,
)
from pydantic import BaseModel

__all__: Sequence[str] = (
    "EndpointResourceGateway",
    "QueueInfo",
    "EndpointResourceGatewayCreateOrUpdateResourcesResponse",
)


class EndpointResourceGatewayCreateOrUpdateResourcesResponse(BaseModel):
    destination: str


class QueueInfo(BaseModel):
    queue_name: str
    broker: BrokerType


Q = TypeVar("Q", bound=QueueInfo)
"""Either a QueueInfo or some specialization of it.
"""


class EndpointResourceGateway(ABC, Generic[Q]):
    @abstractmethod
    async def create_queue(
        self,
        endpoint_record: ModelEndpointRecord,
        labels: Dict[str, str],
    ) -> Q:
        """Creates an asynchronous queue and returns its uniquely identifying information.

        This method MUST be used within the `create_or_update_resources` method.
        """
        raise NotImplementedError

    @abstractmethod
    async def create_or_update_resources(
        self, request: CreateOrUpdateResourcesRequest
    ) -> EndpointResourceGatewayCreateOrUpdateResourcesResponse:
        """
        Creates the infrastructure resources for the endpoint given the request.

        Args:
            request: Specifies the parameters of the infrastructure resources to create.

        Returns: EndpointResourceGatewayCreateOrUpdateResourcesResponse
        """

    @abstractmethod
    async def get_resources(
        self, endpoint_id: str, deployment_name: str, endpoint_type: ModelEndpointType
    ) -> ModelEndpointInfraState:
        """
        Retrieves the infrastructure state for the given endpoint.

        Args:
            endpoint_id: The ID of the endpoint.
            deployment_name: (deprecated) The name of the deployment.
            endpoint_type: The type of the endpoint (sync or async).

        Returns: A domain entity object containing the infrastructure state.
        """

    @abstractmethod
    async def delete_resources(
        self, endpoint_id: str, deployment_name: str, endpoint_type: ModelEndpointType
    ) -> bool:
        """
        Deletes the infrastructure state for the given endpoint.

        Args:
            endpoint_id: The ID of the endpoint.
            deployment_name: (deprecated) The name of the deployment.
            endpoint_type: The type of the endpoint (sync or async).

        Returns: Whether the resources were successfully deleted.
        """

    @abstractmethod
    async def get_all_resources(
        self,
    ) -> Dict[str, Tuple[bool, ModelEndpointInfraState]]:
        """
        Retrieves the infrastructure state for all the endpoints.
        Returns: A Dict where the key is an endpoint_id, and the value is a Tuple with the following components:
            -bool: whether the key actually is an endpoint_id, or if it's a literal k8s deployment name.
            XXX This part is a temporary hack until we attach endpoint_id labels on all k8s resources. Note that unlike
            with point reads/writes, a get_all_resources call scans all k8s objects anyway, after which we can do a
            filtering on the endpoint_id label. Therefore, we're not beholden to the naming scheme of the actual
            k8s objects for this scenario.
            -ModelEndpointInfraState: The endpoint infra state.
        """
