from typing import Any, Dict, List, Optional

from spellbook_serve.common.dtos.model_endpoints import BrokerType
from spellbook_serve.common.dtos.tasks import (
    CreateAsyncTaskV1Response,
    GetAsyncTaskV1Response,
    TaskStatus,
)
from spellbook_serve.core.celery import TaskVisibility, celery_app
from spellbook_serve.core.config import ml_infra_config
from spellbook_serve.core.loggers import filename_wo_ext, make_logger
from spellbook_serve.domain.gateways.task_queue_gateway import TaskQueueGateway

logger = make_logger(filename_wo_ext(__file__))

celery_redis = celery_app(
    None,
    s3_bucket=ml_infra_config().s3_bucket,
    broker_type=str(BrokerType.REDIS.value),
)
celery_redis_24h = celery_app(
    None,
    s3_bucket=ml_infra_config().s3_bucket,
    broker_type=str(BrokerType.REDIS.value),
    task_visibility=TaskVisibility.VISIBILITY_24H,
)
celery_sqs = celery_app(
    None, s3_bucket=ml_infra_config().s3_bucket, broker_type=str(BrokerType.SQS.value)
)


class CeleryTaskQueueGateway(TaskQueueGateway):
    def __init__(self, broker_type: BrokerType):
        self.broker_type = broker_type
        assert self.broker_type in [BrokerType.SQS, BrokerType.REDIS, BrokerType.REDIS_24H]

    def _get_celery_dest(self):
        if self.broker_type == BrokerType.SQS:
            return celery_sqs
        elif self.broker_type == BrokerType.REDIS_24H:
            return celery_redis_24h
        else:  # self.broker_type == BrokerType.REDIS
            return celery_redis

    def send_task(
        self,
        task_name: str,
        queue_name: str,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        expires: Optional[int] = None,
    ) -> CreateAsyncTaskV1Response:
        celery_dest = self._get_celery_dest()
        logger.info(f"Sending task {task_name} with args {args} kwargs {kwargs} to queue {queue_name}")
        res = celery_dest.send_task(
            name=task_name,
            args=args,
            kwargs=kwargs,
            queue=queue_name,
        )
        logger.info(f"Response from sending task {task_name}")
        return CreateAsyncTaskV1Response(task_id=res.id)

    def get_task(self, task_id: str) -> GetAsyncTaskV1Response:
        celery_dest = self._get_celery_dest()
        res = celery_dest.AsyncResult(task_id)
        response_state = res.state
        if response_state == "SUCCESS":
            # No longer wrapping things in the result itself, since the DTO already has a 'result' key:
            # result_dict = (
            #    response_result if type(response_result) is dict else {"result": response_result}
            # )
            return GetAsyncTaskV1Response(
                task_id=task_id, status=TaskStatus.SUCCESS, result=res.result
            )

        elif response_state == "FAILURE":
            return GetAsyncTaskV1Response(
                task_id=task_id,
                status=TaskStatus.FAILURE,
                traceback=res.traceback,
            )

        try:
            task_status = TaskStatus(response_state)
            return GetAsyncTaskV1Response(task_id=task_id, status=task_status)
        except ValueError:
            return GetAsyncTaskV1Response(task_id=task_id, status=TaskStatus.UNDEFINED)
