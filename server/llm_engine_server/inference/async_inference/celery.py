import os

from llm_engine_server.common.dtos.model_endpoints import BrokerType
from llm_engine_server.core.celery import TaskVisibility, celery_app
from llm_engine_server.inference.common import unset_sensitive_envvars

unset_sensitive_envvars()
broker_type_str = os.getenv("BROKER_TYPE")
broker_type = BrokerType(broker_type_str)
s3_bucket: str = os.environ.get("CELERY_S3_BUCKET")  # type: ignore
celery_kwargs = dict(
    name="llm_engine_server.inference.async_inference",
    modules=["llm_engine_server.inference.async_inference.tasks"],
    aws_role=os.environ["AWS_PROFILE"],
    s3_bucket=s3_bucket,
    # s3_base_path = TODO get from env var/config
    task_reject_on_worker_lost=False,
    worker_proc_alive_timeout=1500,
    broker_type=broker_type_str,
    task_visibility=TaskVisibility.VISIBILITY_24H,  # We're using SQS so this only changes task_time_limit
)
if broker_type == BrokerType.SQS:
    queue_name = os.getenv("SQS_QUEUE_NAME")
    queue_url = os.getenv("SQS_QUEUE_URL")
    celery_kwargs.update(
        dict(broker_transport_options={"predefined_queues": {queue_name: {"url": queue_url}}})
    )


async_inference_service = celery_app(**celery_kwargs)  # type: ignore

if __name__ == "__main__":
    async_inference_service.start()
