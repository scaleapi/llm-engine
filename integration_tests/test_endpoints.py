import asyncio
import time
import pytest

from .rest_api_utils import (
    CREATE_ASYNC_MODEL_ENDPOINT_REQUEST_SIMPLE,
    CREATE_ASYNC_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE,
    CREATE_SYNC_MODEL_ENDPOINT_REQUEST_SIMPLE,
    CREATE_SYNC_STREAMING_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE,
    UPDATE_MODEL_ENDPOINT_REQUEST_SIMPLE,
    UPDATE_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE,
    INFERENCE_PAYLOAD,
    INFERENCE_PAYLOAD_RETURN_PICKLED_TRUE,
    INFERENCE_PAYLOAD_RETURN_PICKLED_FALSE,
    USER_ID_0,
    USER_ID_1,
    ensure_gateway_ready,
    delete_existing_endpoints,
    create_model_endpoint,
    get_model_endpoint,
    update_model_endpoint,
    delete_model_endpoint,
    ensure_n_ready_endpoints_long,
    ensure_n_ready_endpoints_short,
    create_async_tasks,
    create_sync_tasks,
    create_streaming_tasks,
    ensure_nonzero_available_workers,
    ensure_all_async_tasks_success,
    ensure_inference_task_response_is_correct,
)

@pytest.fixture(autouse=True)
def delete_endpoints():
    ensure_gateway_ready()
    delete_existing_endpoints()


@pytest.mark.parametrize("user", [USER_ID_0, USER_ID_1])
@pytest.mark.parametrize(
    "create_endpoint_request,update_endpoint_request,inference_requests",
    [
        (CREATE_ASYNC_MODEL_ENDPOINT_REQUEST_SIMPLE, UPDATE_MODEL_ENDPOINT_REQUEST_SIMPLE, [(INFERENCE_PAYLOAD_RETURN_PICKLED_TRUE, True), (INFERENCE_PAYLOAD_RETURN_PICKLED_FALSE, False)]),
        (CREATE_ASYNC_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE, UPDATE_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE, [(INFERENCE_PAYLOAD, False)]),
    ],
)
def test_async_model_endpoint(capsys, user, create_endpoint_request, update_endpoint_request, inference_requests):
    with capsys.disabled():
        try:
            print(f"Creating {create_endpoint_request['name']} model endpoint...")
            create_model_endpoint(create_endpoint_request, user)
            ensure_n_ready_endpoints_long(1, user)

            print(f"Updating {create_endpoint_request['name']} model endpoint...")
            update_model_endpoint(
                create_endpoint_request["name"],
                update_endpoint_request,
                user,
            )
            # Endpoint builds should be cached now.
            ensure_n_ready_endpoints_short(1, user)

            # Let the cache update
            # time.sleep(30)
            # endpoint = get_model_endpoint(create_endpoint_request["name"], user)
            # assert endpoint["resource_state"]["cpus"] == update_endpoint_request["cpus"]
            # assert endpoint["resource_state"]["memory"] == update_endpoint_request["memory"]
            # assert endpoint["deployment_state"]["max_workers"] == update_endpoint_request["max_workers"]

            time.sleep(5)

            for inference_payload, return_pickled in inference_requests:
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
        finally:
            delete_model_endpoint(create_endpoint_request["name"], user)


@pytest.mark.parametrize("user", [USER_ID_0, USER_ID_1])
def test_sync_model_endpoint(capsys, user):
    with capsys.disabled():
        try:
            create_endpoint_request = CREATE_SYNC_MODEL_ENDPOINT_REQUEST_SIMPLE
            update_endpoint_request = UPDATE_MODEL_ENDPOINT_REQUEST_SIMPLE
            inference_requests = [(INFERENCE_PAYLOAD_RETURN_PICKLED_TRUE, True), (INFERENCE_PAYLOAD_RETURN_PICKLED_FALSE, False)]

            print(f"Creating {create_endpoint_request['name']} model endpoint...")
            create_model_endpoint(create_endpoint_request, user)
            ensure_n_ready_endpoints_short(1, user)

            print(f"Updating {create_endpoint_request['name']} model endpoint...")
            update_model_endpoint(
                create_endpoint_request["name"],
                update_endpoint_request,
                user,
            )
            # Endpoint builds should be cached now.
            ensure_n_ready_endpoints_short(1, user)
            ensure_nonzero_available_workers(create_endpoint_request["name"], user)

            # Let the cache update
            # time.sleep(30)
            # endpoint = get_model_endpoint(create_endpoint_request["name"], user)
            # assert endpoint["resource_state"]["cpus"] == update_endpoint_request["cpus"]
            # assert endpoint["resource_state"]["memory"] == update_endpoint_request["memory"]
            # assert endpoint["deployment_state"]["max_workers"] == update_endpoint_request["max_workers"]

            time.sleep(5)

            for inference_payload, return_pickled in inference_requests:
                print(
                    f"Sending sync tasks to {create_endpoint_request['name']} for user {user}, {inference_payload=}, {return_pickled=} ..."
                )
                task_responses = asyncio.run(
                    create_sync_tasks(
                        create_endpoint_request["name"],
                        [inference_payload] * 3,
                        user,
                    )
                )
                for response in task_responses:
                    ensure_inference_task_response_is_correct(response, return_pickled)
        finally:
            delete_model_endpoint(create_endpoint_request["name"], user)


@pytest.mark.parametrize("user", [USER_ID_0, USER_ID_1])
def test_sync_streaming_model_endpoint(capsys, user):
    with capsys.disabled():
        try:
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
            # Endpoint builds should be cached now.
            ensure_n_ready_endpoints_short(1, user)
            ensure_nonzero_available_workers(create_endpoint_request["name"], user)

            # Let the cache update
            # time.sleep(30)
            # endpoint = get_model_endpoint(create_endpoint_request["name"], user)
            # assert endpoint["resource_state"]["cpus"] == update_endpoint_request["cpus"]
            # assert endpoint["resource_state"]["memory"] == update_endpoint_request["memory"]
            # assert endpoint["deployment_state"]["max_workers"] == update_endpoint_request["max_workers"]

            time.sleep(5)

            print(
                f"Sending sync tasks to {create_endpoint_request['name']} for user {user} ..."
            )
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
                    == 'data: {"status": "SUCCESS", "result": {"result": {"y": 1}}, "traceback": null}'
                )
        finally:
            delete_model_endpoint(create_endpoint_request["name"], user)
