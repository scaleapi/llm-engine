from abc import ABC, abstractmethod
from typing import AsyncIterable, Optional

from model_engine_server.common.dtos.tasks import (
    SyncEndpointPredictV1Request,
    SyncEndpointPredictV1Response,
)


class StreamingModelEndpointInferenceGateway(ABC):
    """
    Base class for synchronous inference endpoints.
    Note that this is distinct from the ModelEndpoint class, which is a domain entity object that
    corresponds to CRUD operations on Endpoints. This class hierarchy is where the actual inference
    requests get sent to.
    """

    @abstractmethod
    def streaming_predict(
        self,
        topic: str,
        predict_request: SyncEndpointPredictV1Request,
        manually_resolve_dns: bool,
        readable_endpoint_name: Optional[str],
    ) -> AsyncIterable[SyncEndpointPredictV1Response]:
        """
        Runs a prediction request and returns a streaming response.

        Raises:
            TooManyRequestsException: If the upstream HTTP service raised 429 errors.
            UpstreamServiceError: If the upstream HTTP service raised an error.
        """
