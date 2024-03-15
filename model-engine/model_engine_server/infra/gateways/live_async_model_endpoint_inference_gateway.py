import json
from datetime import datetime

from model_engine_server.common.constants import DEFAULT_CELERY_TASK_NAME
from model_engine_server.common.dtos.tasks import (
    CreateAsyncTaskV1Response,
    EndpointPredictV1Request,
    GetAsyncTaskV1Response,
)
from model_engine_server.domain.gateways.async_model_endpoint_inference_gateway import (
    AsyncModelEndpointInferenceGateway,
)
from model_engine_server.domain.gateways.task_queue_gateway import TaskQueueGateway


class LiveAsyncModelEndpointInferenceGateway(AsyncModelEndpointInferenceGateway):
    """
    Concrete implementation for an AsyncModelEndpointInferenceGateway.

    This particular implementation utilizes a TaskQueueGateway.
    """

    def __init__(self, task_queue_gateway: TaskQueueGateway):
        self.task_queue_gateway = task_queue_gateway

    def create_task(
        self,
        topic: str,
        predict_request: EndpointPredictV1Request,
        task_timeout_seconds: int,
        *,
        task_name: str = DEFAULT_CELERY_TASK_NAME,
    ) -> CreateAsyncTaskV1Response:
        # Use json.loads instead of predict_request.dict() because we have overridden the '__root__'
        # key in some fields, and __root__ overriding only reflects in the json() output.
        predict_args = json.loads(predict_request.json())

        send_task_response = self.task_queue_gateway.send_task(
            task_name=task_name,
            queue_name=topic,
            args=[predict_args, datetime.now(), predict_request.return_pickled],
            expires=task_timeout_seconds,
        )
        return CreateAsyncTaskV1Response(task_id=send_task_response.task_id)

    def get_task(self, task_id: str) -> GetAsyncTaskV1Response:
        # TODO: Deconstruct instead of wrapping?
        get_task_response = self.task_queue_gateway.get_task(task_id=task_id)
        return get_task_response
