import json
import os

import boto3
import smart_open

from .rest_api_utils import (
    CREATE_FINE_TUNE_DI_BATCH_JOB_BUNDLE_REQUEST,
    CREATE_FINE_TUNE_REQUEST,
    USER_ID_0,
    cancel_fine_tune_by_id,
    create_docker_image_batch_job_bundle,
    create_fine_tune,
    get_fine_tune_by_id,
    list_fine_tunes,
)


def test_fine_tunes() -> None:
    di_batch_job = create_docker_image_batch_job_bundle(
        CREATE_FINE_TUNE_DI_BATCH_JOB_BUNDLE_REQUEST, USER_ID_0
    )
    print(f"{di_batch_job=}")
    di_batch_job_id = di_batch_job["docker_image_batch_job_bundle_id"]
    print(f"{di_batch_job_id=}")
    data = {
        "test_base_model-lora": {
            "docker_image_batch_job_bundle_id": di_batch_job_id,
            "launch_bundle_config": {},
            "launch_endpoint_config": {},
            "default_hparams": {},
            "required_params": [],
        }
    }

    if os.getenv("CIRCLECI") == "true":
        session = boto3.Session()
        aws_s3_bucket = os.getenv("CIRCLECI_AWS_S3_BUCKET")
        client = session.client("s3")
        with smart_open.open(
            f"s3://{aws_s3_bucket}/fine_tune_repository",
            "w",
            transport_params={"client": client},
        ) as f:
            json.dump(data, f)

    create_response = create_fine_tune(CREATE_FINE_TUNE_REQUEST, USER_ID_0)
    print(f"{create_response=}")
    fine_tune_id = create_response["id"]
    print(f"{fine_tune_id=}")
    get_response = get_fine_tune_by_id(fine_tune_id, USER_ID_0)
    while get_response["status"] not in ["SUCCESS", "FAILURE"]:
        print(f"{get_response=}")
        get_response = get_fine_tune_by_id(fine_tune_id, USER_ID_0)
    assert get_response["id"] == fine_tune_id
    assert get_response["status"] == "SUCCESS"

    list_response_0_before = list_fine_tunes(USER_ID_0)
    print(f"{list_response_0_before=}")
    num_jobs = len(list_response_0_before["jobs"])
    assert num_jobs >= 1

    cancel_response = cancel_fine_tune_by_id(fine_tune_id, USER_ID_0)
    print(f"{cancel_response=}")

    list_response_0_after = list_fine_tunes(USER_ID_0)
    print(f"{list_response_0_after=}")
    assert len(list_response_0_after["jobs"]) == num_jobs - 1
