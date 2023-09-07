from .rest_api_utils import (
    CREATE_BATCH_JOB_REQUEST,
    CREATE_DOCKER_IMAGE_BATCH_JOB_BUNDLE_REQUEST,
    CREATE_DOCKER_IMAGE_BATCH_JOB_REQUEST,
    USER_ID_0,
    cancel_batch_job,
    create_batch_job,
    create_docker_image_batch_job,
    get_or_create_docker_image_batch_job_bundle,
)
from .test_bundles import model_bundles  # noqa


def test_di_batch_jobs(model_bundles) -> None:  # noqa
    get_or_create_docker_image_batch_job_bundle(
        CREATE_DOCKER_IMAGE_BATCH_JOB_BUNDLE_REQUEST, USER_ID_0
    )
    create_docker_image_batch_job(CREATE_DOCKER_IMAGE_BATCH_JOB_REQUEST, USER_ID_0)

    batch_job_id = create_batch_job(CREATE_BATCH_JOB_REQUEST, USER_ID_0)["job_id"]

    # TODO: assert that batch job actually succeeds.

    cancel_response = cancel_batch_job(batch_job_id, USER_ID_0)
    assert cancel_response["success"]
