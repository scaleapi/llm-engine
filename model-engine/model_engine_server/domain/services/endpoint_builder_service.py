from abc import ABC, abstractmethod

from model_engine_server.common.dtos.endpoint_builder import (
    BuildEndpointRequest,
    BuildEndpointResponse,
)


class EndpointBuilderService(ABC):
    """
    Base class for the Endpoint Builder Service
    """

    @abstractmethod
    async def build_endpoint(self, builder_params: BuildEndpointRequest) -> BuildEndpointResponse:
        """
        Builds an endpoint.

        Args:
            builder_params: the parameters to use for building the endpoint.

        Returns:
            Response for the endpoint builder result.
        """
