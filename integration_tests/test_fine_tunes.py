import json
import os

import boto3
import smart_open

from .rest_api_utils import (
    CREATE_FINE_TUNE_DI_BATCH_JOB_BUNDLE_REQUEST,
    CREATE_FINE_TUNE_REQUEST,
    USER_ID_0,
    USER_ID_1,
    cancel_fine_tune_by_id,
    create_fine_tune,
    get_fine_tune_by_id,
    get_or_create_docker_image_batch_job_bundle,
    list_fine_tunes,
)


def test_fine_tunes() -> None:
    di_batch_job_id = get_or_create_docker_image_batch_job_bundle(
        CREATE_FINE_TUNE_DI_BATCH_JOB_BUNDLE_REQUEST, USER_ID_0
    )["id"]
    data = {
        "test_base_model-lora": {
            "docker_image_batch_job_bundle_id": di_batch_job_id,
            "launch_bundle_config": {},
            "launch_endpoint_config": {},
            "default_hparams": {},
            "required_params": [],
        }
    }

    session = boto3.Session(profile_name=os.getenv("S3_WRITE_AWS_PROFILE"))
    client = session.client("s3")
    with smart_open.open(
        "s3://model-engine-integration-tests/fine_tune_repository/circleci",
        "w",
        transport_params={"client": client},
    ) as f:
        json.dump(data, f)

    create_response = create_fine_tune(CREATE_FINE_TUNE_REQUEST, USER_ID_0)
    fine_tune_id = create_response["id"]

    get_response = get_fine_tune_by_id(fine_tune_id, USER_ID_0)
    assert get_response["id"] == fine_tune_id

    list_response_0_before = list_fine_tunes(USER_ID_0)
    num_jobs = len(list_response_0_before["jobs"])
    assert num_jobs >= 1

    list_response_1 = list_fine_tunes(USER_ID_1)
    assert len(list_response_1["jobs"]) == 0

    cancel_response = cancel_fine_tune_by_id(fine_tune_id, USER_ID_0)
    assert cancel_response["success"]

    list_response_0_after = list_fine_tunes(USER_ID_0)
    assert len(list_response_0_after["jobs"]) == num_jobs - 1
