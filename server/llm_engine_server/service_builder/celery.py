from llm_engine_server.core.celery import celery_app
from llm_engine_server.core.config import ml_infra_config

service_builder_service = celery_app(
    name="llm_engine_server.service_builder",
    modules=[
        "llm_engine_server.service_builder.tasks_v1",
    ],
    s3_bucket=ml_infra_config().s3_bucket,
)

if __name__ == "__main__":
    service_builder_service.start()
