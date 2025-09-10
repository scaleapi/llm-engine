import json
from typing import Any, Dict, List, Optional

import botocore
from model_engine_server.common.dtos.model_endpoints import BrokerType
from model_engine_server.common.dtos.tasks import (
    CreateAsyncTaskV1Response,
    GetAsyncTaskV1Response,
    TaskStatus,
)
from model_engine_server.core.celery import TaskVisibility, celery_app, get_default_backend_protocol
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.core.tracing.tracing_gateway import TracingGateway
from model_engine_server.domain.exceptions import InvalidRequestException
from model_engine_server.domain.gateways.task_queue_gateway import TaskQueueGateway

logger = make_logger(logger_name())

backend_protocol = get_default_backend_protocol()

celery_redis = celery_app(
    None,
    s3_bucket=infra_config().s3_bucket,
    broker_type=str(BrokerType.REDIS.value),
    backend_protocol=backend_protocol,
)
celery_redis_24h = celery_app(
    None,
    s3_bucket=infra_config().s3_bucket,
    broker_type=str(BrokerType.REDIS.value),
    task_visibility=TaskVisibility.VISIBILITY_24H,
    backend_protocol=backend_protocol,
)
celery_sqs = celery_app(
    None,
    s3_bucket=infra_config().s3_bucket,
    broker_type=str(BrokerType.SQS.value),
    backend_protocol=backend_protocol,
)
celery_servicebus = celery_app(
    None, broker_type=str(BrokerType.SERVICEBUS.value), backend_protocol=backend_protocol
)


class CeleryTaskQueueGateway(TaskQueueGateway):
    def __init__(self, broker_type: BrokerType, tracing_gateway: TracingGateway):
        self.broker_type = broker_type
        assert self.broker_type in [
            BrokerType.SQS,
            BrokerType.REDIS,
            BrokerType.REDIS_24H,
            BrokerType.SERVICEBUS,
        ]
        self.tracing_gateway = tracing_gateway

    def _get_celery_dest(self):
        if self.broker_type == BrokerType.SQS:
            return celery_sqs
        elif self.broker_type == BrokerType.REDIS_24H:
            return celery_redis_24h
        elif self.broker_type == BrokerType.REDIS:
            return celery_redis
        else:
            return celery_servicebus

    def send_task(
        self,
        task_name: str,
        queue_name: str,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        expires: Optional[int] = None,
    ) -> CreateAsyncTaskV1Response:
        # Used for both endpoint infra creation and async tasks
        celery_dest = self._get_celery_dest()
        kwargs = kwargs or {}
        with self.tracing_gateway.create_span("send_task_to_queue") as span:
            kwargs.update(self.tracing_gateway.encode_trace_kwargs())
            try:
                res = celery_dest.send_task(
                    name=task_name,
                    args=args,
                    kwargs=kwargs,
                    queue=queue_name,
                )
                span.input = {
                    "queue_name": queue_name,
                    "args": json.loads(json.dumps(args, indent=4, sort_keys=True, default=str)),
                    "task_id": res.id,
                    "task_name": task_name,
                }
                span.output = {"task_id": res.id}
            except botocore.exceptions.ClientError as e:
                logger.exception(f"Error sending task to queue {queue_name}: {e}")
                raise InvalidRequestException(f"Error sending celery task: {e}")
            logger.info(
                f"Task {res.id} sent to queue {queue_name} from gateway"
            )  # pragma: no cover
            return CreateAsyncTaskV1Response(task_id=res.id)

    def get_task(self, task_id: str) -> GetAsyncTaskV1Response:
        # Only used for async tasks
        celery_dest = self._get_celery_dest()
        res = celery_dest.AsyncResult(task_id)
        response_state = res.state
        if response_state == "SUCCESS":
            # No longer wrapping things in the result itself, since the DTO already has a 'result' key:
            # result_dict = (
            #    response_result if type(response_result) is dict else {"result": response_result}
            # )
            status_code = None
            result = res.result
            if isinstance(result, dict) and "status_code" in result:
                # Filter out status code from result if it was added by the forwarder
                # This is admittedly kinda hacky and would technically introduce an edge case
                # if we ever decide not to have async tasks wrap response.
                status_code = result["status_code"]
                del result["status_code"]
            return GetAsyncTaskV1Response(
                task_id=task_id,
                status=TaskStatus.SUCCESS,
                result=result,
                status_code=status_code,
            )

        elif response_state == "FAILURE":
            return GetAsyncTaskV1Response(
                task_id=task_id,
                status=TaskStatus.FAILURE,
                traceback=res.traceback,
                status_code=None,  # probably
            )
        elif response_state == "RETRY":
            # Backwards compatibility, otherwise we'd need to add "RETRY" to the clients
            response_state = "PENDING"

        try:
            task_status = TaskStatus(response_state)
            return GetAsyncTaskV1Response(task_id=task_id, status=task_status)
        except ValueError:
            logger.info(f"Task {task_id} has an unknown state: <{response_state}> ")
            return GetAsyncTaskV1Response(task_id=task_id, status=TaskStatus.UNDEFINED)
