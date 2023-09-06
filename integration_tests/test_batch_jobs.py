from .rest_api_utils import (
    CREATE_BATCH_JOB_REQUEST,
    CREATE_DOCKER_IMAGE_BATCH_JOB_BUNDLE_REQUEST,
    CREATE_DOCKER_IMAGE_BATCH_JOB_REQUEST,
    USER_ID_0,
    create_batch_job,
    create_docker_image_batch_job,
    get_or_create_docker_image_batch_job_bundle,
)
from .test_bundles import model_bundles  # noqa


def test_di_batch_jobs(model_bundles) -> None:  # noqa
    get_or_create_docker_image_batch_job_bundle(
        CREATE_DOCKER_IMAGE_BATCH_JOB_BUNDLE_REQUEST, USER_ID_0
    )

    create_batch_job(CREATE_BATCH_JOB_REQUEST, USER_ID_0)

    create_docker_image_batch_job(CREATE_DOCKER_IMAGE_BATCH_JOB_REQUEST, USER_ID_0)

    # TODO: assert that batch job actually succeeds.
