from abc import ABC, abstractmethod

from llm_engine_server.common.constants import DEFAULT_CELERY_TASK_NAME
from llm_engine_server.common.dtos.tasks import (
    CreateAsyncTaskV1Response,
    EndpointPredictV1Request,
    GetAsyncTaskV1Response,
)


class AsyncModelEndpointInferenceGateway(ABC):
    """
    Base class for asynchronous inference endpoints.
    Note that this is distinct from the ModelEndpoint class, which is a domain entity object that
    corresponds to CRUD operations on Endpoints. This class hierarchy is where the actual inference
    requests get sent to.
    """

    @abstractmethod
    def create_task(
        self,
        topic: str,
        predict_request: EndpointPredictV1Request,
        task_timeout_seconds: int,
        *,
        task_name: str = DEFAULT_CELERY_TASK_NAME,
    ) -> CreateAsyncTaskV1Response:
        """
        Runs a prediction request and returns a response.
        """

    @abstractmethod
    def get_task(self, task_id: str) -> GetAsyncTaskV1Response:
        """
        Gets the status of a prediction request.
        """
