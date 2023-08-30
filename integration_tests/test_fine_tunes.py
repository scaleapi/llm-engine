from .rest_api_utils import (
    CREATE_FINE_TUNE_REQUEST,
    CREATE_FINE_TUNE_DI_BATCH_JOB_BUNDLE_REQUEST,
    USER_ID_0,
    USER_ID_1,
    create_fine_tune,
    get_fine_tune_by_id,
    list_fine_tunes,
    cancel_fine_tune_by_id,
    get_or_create_docker_image_batch_job_bundle,
)

def test_fine_tunes() -> None:
    # di_batch_job_id = get_or_create_docker_image_batch_job_bundle(CREATE_FINE_TUNE_DI_BATCH_JOB_BUNDLE_REQUEST, USER_ID_0)["id"]

    create_response = create_fine_tune(CREATE_FINE_TUNE_REQUEST, USER_ID_0)
    fine_tune_id = create_response["fine_tune_id"]

    get_response = get_fine_tune_by_id(fine_tune_id, USER_ID_0)
    assert get_response["fine_tune_id"] == fine_tune_id

    list_response_0_before = list_fine_tunes(USER_ID_0)
    num_jobs = len(list_response_0_before["jobs"])
    assert num_jobs >= 1

    list_response_1 = list_fine_tunes(USER_ID_1)
    assert len(list_response_1["jobs"]) == 0

    cancel_response = cancel_fine_tune_by_id(fine_tune_id, USER_ID_0)
    assert cancel_response["success"]

    list_response_0_after = list_fine_tunes(USER_ID_0)
    assert len(list_response_0_after["jobs"]) == num_jobs - 1
    