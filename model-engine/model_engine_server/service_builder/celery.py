from model_engine_server.common.dtos.model_endpoints import BrokerType
from model_engine_server.common.env_vars import CIRCLECI
from model_engine_server.core.celery import celery_app
from model_engine_server.core.config import infra_config


# TODO: this is copied from celery_task_queue_gateway.py
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


service_builder_broker_type: str
# TODO: this seems redundant? we definitely have other code doing this
if CIRCLECI:
    service_builder_broker_type = str(BrokerType.REDIS.value)
elif infra_config().cloud_provider == "azure":
    service_builder_broker_type = str(BrokerType.SERVICEBUS.value)
elif infra_config().cloud_provider == "gcp":
    service_builder_broker_type = str(BrokerType.GCPPUBSUB.value)
else:
    service_builder_broker_type = str(BrokerType.SQS.value)

service_builder_service = celery_app(
    name="model_engine_server.service_builder",
    modules=[
        "model_engine_server.service_builder.tasks_v1",
    ],
    s3_bucket=infra_config().s3_bucket,
    broker_type=service_builder_broker_type,
    backend_protocol=get_backend_protocol(),  # TODO: similarly, this has a big overlap with celery_task_queue_gateway.py
)

if __name__ == "__main__":
    service_builder_service.start()
