from abc import abstractmethod
from typing import Sequence

from spellbook_serve.domain.entities import ModelEndpointRecord, ModelEndpointsSchema


class ModelEndpointsSchemaGateway:
    """Base class for Model Endpoints Schema gateways."""

    @abstractmethod
    def get_model_endpoints_schema(
        self,
        model_endpoint_records: Sequence[ModelEndpointRecord],
    ) -> ModelEndpointsSchema:
        """Get the OpenAPI schema for the model endpoints.

        Args:
            model_endpoint_records: The model endpoint records.

        Returns:
            ModelEndpointsSchema: The OpenAPI schema for the model endpoints.
        """
