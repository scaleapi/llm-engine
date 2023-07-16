# Represents high-level CRUD operations for model endpoints.
from abc import ABC, abstractmethod
from typing import List, Optional

from llm_engine_server.common.dtos.model_endpoints import ModelEndpointOrderBy
from llm_engine_server.domain.entities import ModelEndpoint


class LLMModelEndpointService(ABC):
    """
    Base class for LLM Model Endpoint services.
    """

    @abstractmethod
    async def list_llm_model_endpoints(
        self,
        owner: Optional[str],
        name: Optional[str],
        order_by: Optional[ModelEndpointOrderBy],
    ) -> List[ModelEndpoint]:
        """
        Lists LLM model endpoints.
        Args:
            owner: The user ID of the owner of the endpoints.
            name: An optional name of the endpoint used for filtering endpoints.
            order_by: The ordering to output the Model Endpoints.
        Returns:
            A Model Endpoint Record domain entity, or None if not found.
        """

    @abstractmethod
    async def get_llm_model_endpoint(self, model_endpoint_name: str) -> Optional[ModelEndpoint]:
        """
        Gets an LLM model endpoint.
        Args:
            model_endpoint_name: The name of the model endpoint.
        Returns:
            A Model Endpoint domain entity, or None if not found.
        """
