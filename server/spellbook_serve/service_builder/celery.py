from spellbook_serve.core.celery import celery_app
from spellbook_serve.core.config import ml_infra_config

service_builder_service = celery_app(
    name="spellbook_serve.service_builder",
    modules=[
        "spellbook_serve.service_builder.tasks_v1",
    ],
    s3_bucket=ml_infra_config().s3_bucket,
)

if __name__ == "__main__":
    service_builder_service.start()
