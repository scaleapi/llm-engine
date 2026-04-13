from model_engine_server.common.dtos.model_endpoints import BrokerType
from model_engine_server.common.env_vars import CIRCLECI
from model_engine_server.core.celery import celery_app
from model_engine_server.core.config import infra_config


def get_broker_type(cloud_provider: str, is_ci: bool, force_redis: bool) -> str:
    if is_ci or force_redis:
        return str(BrokerType.REDIS.value)
    elif cloud_provider == "azure":
        return str(BrokerType.SERVICEBUS.value)
    elif cloud_provider == "gcp":
        return str(BrokerType.REDIS.value)
    else:
        return str(BrokerType.SQS.value)


service_builder_broker_type = get_broker_type(
    cloud_provider=infra_config().cloud_provider,
    is_ci=bool(CIRCLECI),
    force_redis=bool(infra_config().celery_broker_type_redis),
)

service_builder_service = celery_app(
    name="model_engine_server.service_builder",
    modules=[
        "model_engine_server.service_builder.tasks_v1",
    ],
    s3_bucket=infra_config().s3_bucket,
    broker_type=service_builder_broker_type,
    backend_protocol=(
        "abs"
        if infra_config().cloud_provider == "azure"
        else ("redis" if infra_config().cloud_provider == "gcp" else "s3")
    ),
    # Add detailed task tracking for debugging
    task_track_started=True,
    task_remote_tracebacks=True,
    # Reduce time limits to catch hanging tasks faster
    task_time_limit=1800,  # 30 minutes hard limit
    task_soft_time_limit=1500,  # 25 minutes soft limit
)

if __name__ == "__main__":
    service_builder_service.start()
