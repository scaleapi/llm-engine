from abc import ABC, abstractmethod

from spellbook_serve.common.dtos.tasks import (
    EndpointPredictV1Request,
    SyncEndpointPredictV1Response,
)


class SyncModelEndpointInferenceGateway(ABC):
    """
    Base class for synchronous inference endpoints.
    Note that this is distinct from the ModelEndpoint class, which is a domain entity object that
    corresponds to CRUD operations on Endpoints. This class hierarchy is where the actual inference
    requests get sent to.
    """

    @abstractmethod
    async def predict(
        self, topic: str, predict_request: EndpointPredictV1Request
    ) -> SyncEndpointPredictV1Response:
        """
        Runs a prediction request and returns a response.

        Raises:
            TooManyRequestsException: If the upstream HTTP service raised 429 errors.
            UpstreamServiceError: If the upstream HTTP service raised an error.
        """
