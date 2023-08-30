from .rest_api_utils import (
    CREATE_DOCKER_IMAGE_BATCH_JOB_BUNDLE_REQUEST,
    CREATE_BATCH_JOB_REQUEST,
    CREATE_DOCKER_IMAGE_BATCH_JOB_REQUEST,
    USER_ID_0,
    get_or_create_docker_image_batch_job_bundle,
    create_batch_job,
    create_docker_image_batch_job,
)

def test_di_batch_jobs() -> None:
    get_or_create_docker_image_batch_job_bundle(CREATE_DOCKER_IMAGE_BATCH_JOB_BUNDLE_REQUEST, USER_ID_0)

    create_batch_job(CREATE_BATCH_JOB_REQUEST, USER_ID_0)

    create_docker_image_batch_job(CREATE_DOCKER_IMAGE_BATCH_JOB_REQUEST, USER_ID_0)

    # TODO: assert that batch job actually succeeds.
