import asyncio
import os

import pytest

from .rest_api_utils import (
    CREATE_LLM_MODEL_ENDPOINT_REQUEST,
    LLM_PAYLOADS_WITH_EXPECTED_RESPONSES,
    USER_ID_0,
    create_llm_model_endpoint,
    create_llm_streaming_tasks,
    create_llm_sync_tasks,
    delete_llm_model_endpoint,
    ensure_launch_gateway_healthy,
    ensure_llm_task_response_is_correct,
    ensure_llm_task_stream_response_is_correct,
    ensure_n_ready_private_llm_endpoints_short,
    ensure_nonzero_available_llm_workers,
)

TEST_INFERENCE_FRAMEWORK = os.environ.get("TEST_INFERENCE_FRAMEWORK", None)
TEST_INFERENCE_FRAMEWORK_IMAGE_TAG = os.environ.get("TEST_INFERENCE_FRAMEWORK_IMAGE_TAG", None)
print(f"TEST_INFERENCE_FRAMEWORK={TEST_INFERENCE_FRAMEWORK}")


@pytest.mark.skipif(
    (not TEST_INFERENCE_FRAMEWORK) or (not TEST_INFERENCE_FRAMEWORK_IMAGE_TAG),
    reason="Skip unless running inference framework tests",
)
def test_completions(capsys):
    ensure_launch_gateway_healthy()
    with capsys.disabled():
        try:
            user = USER_ID_0
            create_endpoint_request = CREATE_LLM_MODEL_ENDPOINT_REQUEST

            print(f"Creating {create_endpoint_request['name']} model endpoint...")
            create_llm_model_endpoint(
                create_endpoint_request,
                user,
                TEST_INFERENCE_FRAMEWORK,
                TEST_INFERENCE_FRAMEWORK_IMAGE_TAG,
            )
            ensure_n_ready_private_llm_endpoints_short(1, user)
            ensure_nonzero_available_llm_workers(create_endpoint_request["name"], user)

            for (
                completions_payload,
                required_output_fields,
                response_text_regex,
            ) in LLM_PAYLOADS_WITH_EXPECTED_RESPONSES:
                print(
                    f"Sending sync tasks to {create_endpoint_request['name']} for user {user}, {completions_payload=}..."
                )
                try:
                    task_responses = asyncio.run(
                        create_llm_sync_tasks(
                            create_endpoint_request["name"],
                            [completions_payload],
                            user,
                        )
                    )
                    for response in task_responses:
                        ensure_llm_task_response_is_correct(
                            response, required_output_fields, response_text_regex
                        )
                except Exception as e:
                    if hasattr(e, "response") and e.response.status_code // 100 == 4:
                        print(f"Got 4xx status code for {completions_payload=}, which is expected")
                    else:
                        raise e

            for (
                completions_payload,
                required_output_fields,
                response_text_regex,
            ) in LLM_PAYLOADS_WITH_EXPECTED_RESPONSES:
                print(
                    f"Sending streaming tasks to {create_endpoint_request['name']} for user {user}, {completions_payload=}..."
                )
                try:
                    task_responses = asyncio.run(
                        create_llm_streaming_tasks(
                            create_endpoint_request["name"],
                            [completions_payload],
                            user,
                        )
                    )
                    for response in task_responses:
                        ensure_llm_task_stream_response_is_correct(
                            response, required_output_fields, response_text_regex
                        )
                except Exception as e:
                    if hasattr(e, "response") and e.response.status_code // 100 == 4:
                        print(f"Got 4xx status code for {completions_payload=}, which is expected")
                    else:
                        raise e
        finally:
            delete_llm_model_endpoint(create_endpoint_request["name"], user)
