from model_engine_server.core.celery import celery_app
from model_engine_server.core.config import infra_config

service_builder_service = celery_app(
    name="model_engine_server.service_builder",
    modules=[
        "model_engine_server.service_builder.tasks_v1",
    ],
    s3_bucket=infra_config().s3_bucket,
)

if __name__ == "__main__":
    service_builder_service.start()
