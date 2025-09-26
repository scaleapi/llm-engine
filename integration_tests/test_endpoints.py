import asyncio
import time

import pytest
from tenacity import RetryError, retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from .rest_api_utils import (
    CREATE_ASYNC_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE,
    CREATE_ASYNC_MODEL_ENDPOINT_REQUEST_SIMPLE,
    CREATE_SYNC_MODEL_ENDPOINT_REQUEST_SIMPLE,
    CREATE_SYNC_STREAMING_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE,
    INFERENCE_PAYLOAD,
    INFERENCE_PAYLOAD_RETURN_PICKLED_FALSE,
    INFERENCE_PAYLOAD_RETURN_PICKLED_TRUE,
    UPDATE_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE,
    UPDATE_MODEL_ENDPOINT_REQUEST_SIMPLE,
    USER_ID_0,
    create_async_tasks,
    create_model_endpoint,
    create_streaming_tasks,
    create_sync_tasks,
    delete_existing_endpoints,
    delete_model_endpoint,
    ensure_all_async_tasks_success,
    ensure_gateway_ready,
    ensure_inference_task_response_is_correct,
    ensure_n_ready_endpoints_long,
    ensure_n_ready_endpoints_short,
    ensure_nonzero_available_workers,
    get_model_endpoint,
    update_model_endpoint,
)


@pytest.fixture(autouse=True)
def delete_endpoints(capsys):
    try:
        ensure_gateway_ready()
        delete_existing_endpoints()
    except Exception:
        with capsys.disabled():
            print("Endpoint deletion failed")


@retry(stop=stop_after_attempt(3), wait=wait_fixed(10), retry=retry_if_exception_type(RetryError))
def ensure_async_inference_works(user, create_endpoint_request, inference_payload, return_pickled):
    print(
        f"Sending async tasks to {create_endpoint_request['name']} for user {user}, {inference_payload=}, {return_pickled=} ..."
    )
    task_ids = asyncio.run(
        create_async_tasks(
            create_endpoint_request["name"],
            [inference_payload] * 3,
            user,
        )
    )
    print("Retrieving async task results...")
    ensure_nonzero_available_workers(create_endpoint_request["name"], user)
    ensure_all_async_tasks_success(task_ids, user, return_pickled)


@retry(stop=stop_after_attempt(3), wait=wait_fixed(20))
def ensure_endpoint_updated(create_endpoint_request, update_endpoint_request, user):
    endpoint = get_model_endpoint(create_endpoint_request["name"], user)
    assert endpoint["resource_state"]["cpus"] == update_endpoint_request["cpus"]
    assert endpoint["resource_state"]["memory"] == update_endpoint_request["memory"]
    assert endpoint["deployment_state"]["max_workers"] == update_endpoint_request["max_workers"]


@pytest.mark.parametrize(
    "create_endpoint_request,update_endpoint_request,inference_requests",
    [
        (
            CREATE_ASYNC_MODEL_ENDPOINT_REQUEST_SIMPLE,
            UPDATE_MODEL_ENDPOINT_REQUEST_SIMPLE,
            [
                (INFERENCE_PAYLOAD_RETURN_PICKLED_TRUE, True),
                (INFERENCE_PAYLOAD_RETURN_PICKLED_FALSE, False),
            ],
        ),
        (
            CREATE_ASYNC_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE,
            UPDATE_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE,
            [(INFERENCE_PAYLOAD, False)],
        ),
    ],
)
def test_async_model_endpoint(
    capsys, create_endpoint_request, update_endpoint_request, inference_requests
):
    with capsys.disabled():
        try:
            user = USER_ID_0
            print(f"Creating {create_endpoint_request['name']} model endpoint...")
            create_model_endpoint(create_endpoint_request, user)
            ensure_n_ready_endpoints_long(1, user)

            print(f"Updating {create_endpoint_request['name']} model endpoint...")
            update_model_endpoint(
                create_endpoint_request["name"],
                update_endpoint_request,
                user,
            )
            # Let the cache update
            time.sleep(60)
            # Endpoint builds should be cached now.
            ensure_n_ready_endpoints_short(1, user)

            print("Checking endpoint state...")
            ensure_endpoint_updated(create_endpoint_request, update_endpoint_request, user)

            time.sleep(20)

            for inference_payload, return_pickled in inference_requests:
                ensure_async_inference_works(
                    user, create_endpoint_request, inference_payload, return_pickled
                )
        finally:
            delete_model_endpoint(create_endpoint_request["name"], user)


def test_sync_model_endpoint(capsys):
    with capsys.disabled():
        try:
            user = USER_ID_0
            create_endpoint_request = CREATE_SYNC_MODEL_ENDPOINT_REQUEST_SIMPLE
            update_endpoint_request = UPDATE_MODEL_ENDPOINT_REQUEST_SIMPLE
            inference_requests = [
                (INFERENCE_PAYLOAD_RETURN_PICKLED_TRUE, True),
                (INFERENCE_PAYLOAD_RETURN_PICKLED_FALSE, False),
            ]

            print(f"Creating {create_endpoint_request['name']} model endpoint...")
            create_model_endpoint(create_endpoint_request, user)
            ensure_n_ready_endpoints_short(1, user)

            print(f"Updating {create_endpoint_request['name']} model endpoint...")
            update_model_endpoint(
                create_endpoint_request["name"],
                update_endpoint_request,
                user,
            )
            # Let the cache update
            time.sleep(30)
            # Endpoint builds should be cached now.
            ensure_n_ready_endpoints_short(1, user)
            ensure_nonzero_available_workers(create_endpoint_request["name"], user)

            print("Checking endpoint state...")
            endpoint = get_model_endpoint(create_endpoint_request["name"], user)
            assert endpoint["resource_state"]["cpus"] == update_endpoint_request["cpus"]
            assert endpoint["resource_state"]["memory"] == update_endpoint_request["memory"]
            assert (
                endpoint["deployment_state"]["max_workers"]
                == update_endpoint_request["max_workers"]
            )

            time.sleep(10)

            for inference_payload, return_pickled in inference_requests:
                print(
                    f"Sending sync tasks to {create_endpoint_request['name']} for user {user}, {inference_payload=}, {return_pickled=} ..."
                )
                task_responses = asyncio.run(
                    create_sync_tasks(
                        create_endpoint_request["name"],
                        [inference_payload],
                        user,
                    )
                )
                for response in task_responses:
                    ensure_inference_task_response_is_correct(response, return_pickled)
        finally:
            delete_model_endpoint(create_endpoint_request["name"], user)


def test_sync_streaming_model_endpoint(capsys):
    with capsys.disabled():
        try:
            user = USER_ID_0
            create_endpoint_request = CREATE_SYNC_STREAMING_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE
            update_endpoint_request = UPDATE_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE

            print(f"Creating {create_endpoint_request['name']} model endpoint...")
            create_model_endpoint(create_endpoint_request, user)
            ensure_n_ready_endpoints_short(1, user)

            print(f"Updating {create_endpoint_request['name']} model endpoint...")
            update_model_endpoint(
                create_endpoint_request["name"],
                update_endpoint_request,
                user,
            )
            # Let the cache update
            time.sleep(30)
            # Endpoint builds should be cached now.
            ensure_n_ready_endpoints_short(1, user)
            ensure_nonzero_available_workers(create_endpoint_request["name"], user)

            print("Checking endpoint state...")
            endpoint = get_model_endpoint(create_endpoint_request["name"], user)
            assert endpoint["resource_state"]["cpus"] == update_endpoint_request["cpus"]
            assert endpoint["resource_state"]["memory"] == update_endpoint_request["memory"]
            assert (
                endpoint["deployment_state"]["max_workers"]
                == update_endpoint_request["max_workers"]
            )

            time.sleep(5)

            print(f"Sending sync tasks to {create_endpoint_request['name']} for user {user} ...")
            task_responses = asyncio.run(
                create_sync_tasks(
                    create_endpoint_request["name"],
                    [INFERENCE_PAYLOAD] * 3,
                    user,
                )
            )
            for response in task_responses:
                ensure_inference_task_response_is_correct(response, False)

            print(
                f"Sending streaming tasks to {create_endpoint_request['name']} for user {user} ..."
            )
            task_responses = asyncio.run(
                create_streaming_tasks(
                    create_endpoint_request["name"],
                    [INFERENCE_PAYLOAD] * 5,
                    user,
                )
            )
            for response in task_responses:
                assert (
                    response.strip()
                    == 'data: {"status":"SUCCESS","result":{"result":{"y":1}},"traceback":null,"status_code":200}'
                )
        finally:
            delete_model_endpoint(create_endpoint_request["name"], user)


@pytest.mark.skipif(
    reason="Need to update the following test to hit remote service to be integration test"
)
def test_models_tokenizers() -> None:
    from model_engine_server.infra.gateways.s3_llm_artifact_gateway import S3LLMArtifactGateway
    from model_engine_server.infra.repositories import LiveTokenizerRepository
    from model_engine_server.infra.repositories.live_tokenizer_repository import (
        SUPPORTED_MODELS_INFO,
    )

    llm_artifact_gateway = S3LLMArtifactGateway()
    tokenizer_repository = LiveTokenizerRepository(llm_artifact_gateway=llm_artifact_gateway)
    for model_name in SUPPORTED_MODELS_INFO:
        tokenizer_repository.load_tokenizer(model_name)
