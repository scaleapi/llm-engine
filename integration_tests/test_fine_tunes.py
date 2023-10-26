"""
from .rest_api_utils import (  # CREATE_FINE_TUNE_DI_BATCH_JOB_BUNDLE_REQUEST, CREATE_FINE_TUNE_REQUEST, USER_ID_0, cancel_fine_tune_by_id, create_docker_image_batch_job_bundle, create_fine_tune, get_fine_tune_by_id,
    USER_ID_0,
    cancel_fine_tune_by_id,
    list_fine_tunes,
)
"""


def test_fine_tunes() -> None:
    # TODO: get this test to work (move LLM fine tune repository to database rather than in S3)

    # di_batch_job_id = create_docker_image_batch_job_bundle(
    #     CREATE_FINE_TUNE_DI_BATCH_JOB_BUNDLE_REQUEST, USER_ID_0
    # )["docker_image_batch_job_bundle_id"]

    # create_response = create_fine_tune(CREATE_FINE_TUNE_REQUEST, USER_ID_0)
    # fine_tune_id = create_response["id"]

    # get_response = get_fine_tune_by_id(fine_tune_id, USER_ID_0)
    # assert get_response["id"] == fine_tune_id

    # list_response_0_before = list_fine_tunes(USER_ID_0)
    # num_jobs = len(list_response_0_before["jobs"])
    # assert num_jobs >= 1

    # list_response_1 = list_fine_tunes(USER_ID_0)
    # assert len(list_response_1["jobs"]) == 0
    """
    if len(list_response_1) > 0:
        for resp in list_response_1:
            print(f"responses list is {list_response_1}")
            print(f"response is {resp}")
            try:
                fine_tune_id = resp.id
                cancel_response = cancel_fine_tune_by_id(fine_tune_id, USER_ID_0)
                assert cancel_response["success"]
            except:
                pass
    """

    # list_response_0_after = list_fine_tunes(USER_ID_0)
    # assert len(list_response_0_after["jobs"]) == num_jobs - 1
