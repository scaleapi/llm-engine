from model_engine_server.common.dtos.model_endpoints import BrokerType
from model_engine_server.common.env_vars import CIRCLECI
from model_engine_server.core.celery import celery_app
from model_engine_server.core.config import infra_config

service_builder_broker_type: str
if CIRCLECI:
    service_builder_broker_type = str(BrokerType.REDIS.value)
elif infra_config().cloud_provider == "azure":
    service_builder_broker_type = str(BrokerType.SERVICEBUS.value)
else:
    service_builder_broker_type = str(BrokerType.REDIS.value)

service_builder_service = celery_app(
    name="model_engine_server.service_builder",
    modules=[
        "model_engine_server.service_builder.tasks_v1",
    ],
    s3_bucket=infra_config().s3_bucket,
    broker_type=service_builder_broker_type,
    backend_protocol="abs" if infra_config().cloud_provider == "azure" else "s3",
)

if __name__ == "__main__":
    service_builder_service.start()
