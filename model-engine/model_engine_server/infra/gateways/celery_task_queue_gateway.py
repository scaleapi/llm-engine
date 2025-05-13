from typing import Any, Dict, List, Optional

import botocore
from model_engine_server.common.dtos.model_endpoints import BrokerType
from model_engine_server.common.dtos.tasks import (
    CreateAsyncTaskV1Response,
    GetAsyncTaskV1Response,
    TaskStatus,
)
from model_engine_server.core.celery import TaskVisibility, celery_app
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.exceptions import InvalidRequestException
from model_engine_server.domain.gateways.task_queue_gateway import TaskQueueGateway

logger = make_logger(logger_name())


def get_backend_protocol():
    cloud_provider = infra_config().cloud_provider
    if cloud_provider == "azure":
        return "abs"
    elif cloud_provider == "aws":
        return "s3"
    elif cloud_provider == "gcp":
        return "gcppubsub"
    else:
        return "s3"  # TODO: I feel like we should raise an error here.


backend_protocol = get_backend_protocol()

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
    None,
    broker_type=str(BrokerType.SERVICEBUS.value),
    backend_protocol=backend_protocol,
    # TODO: check how Azure uses s3
)

# XXX: check the next line
celery_gcppubsub = celery_app(
    None,
    broker_type=str(BrokerType.GCPPUBSUB.value),
    backend_protocol=backend_protocol,
)


class CeleryTaskQueueGateway(TaskQueueGateway):
    def __init__(self, broker_type: BrokerType):
        self.broker_type = broker_type
        assert (
            self.broker_type
            in [  # TODO: why do have this assert? this is the same as the enum -- is it so we remember to implement it here?
                BrokerType.SQS,
                BrokerType.REDIS,
                BrokerType.REDIS_24H,
                BrokerType.SERVICEBUS,
                BrokerType.GCPPUBSUB,
            ]
        )

    def _get_celery_dest(self):
        if self.broker_type == BrokerType.SQS:
            return celery_sqs
        elif self.broker_type == BrokerType.REDIS_24H:
            return celery_redis_24h
        elif self.broker_type == BrokerType.REDIS:
            return celery_redis
        elif self.broker_type == BrokerType.GCPPUBSUB:
            return celery_gcppubsub
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

        try:
            res = celery_dest.send_task(
                name=task_name,
                args=args,
                kwargs=kwargs,
                queue=queue_name,
            )
        except botocore.exceptions.ClientError as e:
            logger.exception(f"Error sending task to queue {queue_name}: {e}")
            raise InvalidRequestException(f"Error sending celery task: {e}")
        logger.info(f"Task {res.id} sent to queue {queue_name} from gateway")  # pragma: no cover
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
            if type(result) is dict and "status_code" in result:
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
