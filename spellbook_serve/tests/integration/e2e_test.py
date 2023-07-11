# Tests the gateway and service builder version 0 for an end-to-end workflow.
#
# This includes creating bundles, building and updating endpoints, sending requests to the
# endpoints, and deleting the endpoints and bundles.
#
# This test expects a local Launch gateway service listening at http://localhost:5000.
#
# The Launch client is not used to keep this test free of external dependencies.

import argparse
import asyncio
import os
from typing import Any, Dict, List

import aiohttp
import requests
from tenacity import retry, stop_after_attempt, wait_fixed

from spellbook_serve.core.config import ml_infra_config

_DEFAULT_BASE_PATH = "http://localhost:5001"
BASE_PATH = os.environ.get("BASE_PATH", _DEFAULT_BASE_PATH)
print(f"Integration tests using gateway {BASE_PATH=}")
DEFAULT_NETWORK_TIMEOUT_SEC = 10

if BASE_PATH == _DEFAULT_BASE_PATH:
    # Generate some fake 24-character user IDs (Scale user IDs are 24 chars).
    # We don't want different people to get user ID collisions but at the same time we want people to
    # consistently use the same user IDs so that they can clean up their extra endpoints.
    USER_PREFIX = os.getenv("SERVICE_IDENTIFIER", "test")[:8]
    USER_ID_0 = USER_PREFIX + "0" * (24 - len(USER_PREFIX))
    USER_ID_1 = USER_PREFIX + "1" * (24 - len(USER_PREFIX))
else:
    USER_ID_0 = USER_ID_1 = "62bc820451dbea002b1c5421"
    print(f"Non-local mode! Using USERs == {USER_ID_0}")

if (OVERRIDE_USER_ID := os.environ.get("OVERRIDE_USER_ID", None)) is not None:
    print(f"Overriding user ID for testing with env var {OVERRIDE_USER_ID=}")
    USER_ID_0 = USER_ID_1 = OVERRIDE_USER_ID

CREATE_MODEL_BUNDLE_REQUEST_SIMPLE = {
    "packaging_type": "cloudpickle",
    "bundle_name": "model_bundle_simple",
    "location": "s3://scale-ml/model_bundles/61a67d767bce560024c7eb96/c0c198c3-941a-427e-8f54-db31690001b4",
    "bundle_metadata": {
        "load_predict_fn": """def returns_returns_1(x):
    def returns_1(y):
        return 1
    return returns_1""",
    },
    "requirements": [],
    "env_params": {
        "framework_type": "pytorch",
        "pytorch_image_tag": "1.7.1-cuda11.0-cudnn8-runtime",
    },
}

CREATE_MODEL_BUNDLE_REQUEST_CUSTOM_IMAGE = {
    "packaging_type": "zip",
    "bundle_name": "model_bundle_custom_image",
    "location": "s3://scale-ml/scale-launch/model_bundles/seamonkey/2c965c70-bab4-4d8f-b364-3fc46ecab568",
    "bundle_metadata": {
        "load_predict_fn_module_path": "seamonkey.models.model_top_k_matches.load_seamonkey_predict_top_k_matches_fn",
        "load_model_fn_module_path": "seamonkey.models.model_top_k_matches.load_seamonkey_model_top_k_matches_fn",
    },
    "requirements": [
        "attrdict==2.0.1",
        "# Install torch and torchvision",
        "--find-links https://download.pytorch.org/whl/torch_stable.html",
        "# WARNING: MUST KEEP CUDA VERSIONS IN SYNC            !!!",
        "--extra-index-url https://download.pytorch.org/whl/cu113",
        "torch==1.10.0+cu113; sys_platform == 'linux' ",
        "torchvision==0.11.0+cu113; sys_platform == 'linux'",
        "# WARNING: YOU MUST MAKE SURE THIS VALUE IS THE SAME   !!!",
        "#          AS THE ONE IN INSTALL.SH AND Dockerfile.gpu !!!",
        "torch==1.10.0; sys_platform == 'darwin' ",
        "torchvision==0.11.0; sys_platform == 'darwin'",
        "",
        "# Install other dependencies",
        "Jinja2==3.0.3",
        "pydantic~=1.8.2",
        "ftfy==6.1.1",
        "regex==2022.3.15",
        "timm==0.5.4",
        "opencv-python==4.6.0.66",
        "Pillow==9.1.1",
        "smart_open==5.2.1",
        "s3fs",
        "git+https://github.com/openai/CLIP.git@b46f5ac7587d2e1862f8b7b1573179d80dcdd620",
    ],
    "env_params": {
        "framework_type": "custom_base_image",
        "image_tag": "64a006a64d2b1747d7332addcfdcbd505ba8b509-gpu",
        "ecr_repo": "seamonkey",
    },
    "app_config": {
        "faiss_index": "s3://scale-ml/temp/seamonkey-reference-set/faiss_flat_256_DIM_2022-11-23-14.faiss",
        "metadata_pkl": "s3://scale-ml/temp/seamonkey-reference-set/metadata_file_256_DIM_2022-11-23-14.pkl",
        "reference_set_version": "810dd15931d8737bba9f690138d44efc",
        "threshold": 0.925,
        "logging_level": "INFO",
    },
}

AWS_ROLE = "default" if os.getenv("CIRCLECI") else "ml-worker"
S3_BUCKET = ml_infra_config().s3_bucket

CREATE_MODEL_ENDPOINT_REQUEST_SIMPLE = {
    "bundle_name": "model_bundle_simple",
    "endpoint_name": "model-endpoint-simple",
    "endpoint_type": "async",
    "cpus": "0.5",
    "memory": "500Mi",
    "min_workers": 1,
    "max_workers": 1,
    "gpus": 0,
    "per_worker": 1,
    "aws_role": AWS_ROLE,
    "results_s3_bucket": S3_BUCKET,
    "labels": {"team": "infra", "product": "launch"},
    "post_inference_hooks": [],
}

CREATE_MODEL_ENDPOINT_REQUEST_CUSTOM_IMAGE = {
    "bundle_name": "model_bundle_custom_image",
    "post_inference_hooks": [],
    "endpoint_type": "async",
    "endpoint_name": "model-endpoint-custom-image",
    "cpus": "3",
    "gpu_type": "nvidia-tesla-t4",
    "gpus": 1,
    "memory": "12Gi",
    "min_workers": 0,
    "max_workers": 1,
    "per_worker": 1,
    "aws_role": AWS_ROLE,
    "results_s3_bucket": S3_BUCKET,
    "labels": {"team": "seamonkey", "product": "seamonkey"},
}

UPDATE_MODEL_ENDPOINT_REQUEST_SIMPLE = {
    "bundle_name": "model_bundle_simple",
    "cpus": "0.75",
    "max_workers": 2,
}

UPDATE_MODEL_ENDPOINT_REQUEST_CUSTOM_IMAGE = {
    "bundle_name": "model_bundle_custom_image",
    "cpus": "2",
    "max_workers": 2,
}

CREATE_ASYNC_TASK = {
    "args": {"y": 1},
    "return_pickled": True,
}


def create_model_bundle(
    create_model_bundle_request: Dict[str, Any], user_id: str
) -> Dict[str, Any]:
    response = requests.post(
        f"{BASE_PATH}/model_bundle",
        json=create_model_bundle_request,
        headers={"Content-Type": "application/json"},
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def delete_model_bundle(bundle_name: str, user_id: str) -> Dict[str, Any]:
    response = requests.delete(
        f"{BASE_PATH}/model_bundle/{bundle_name}",
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


async def create_async_task(
    endpoint_name: str,
    create_async_task_request: Dict[str, Any],
    user_id: str,
    session: aiohttp.ClientSession,
) -> str:
    async with session.post(
        f"{BASE_PATH}/task_async/{endpoint_name}",
        json=create_async_task_request,
        headers={"Content-Type": "application/json"},
        auth=aiohttp.BasicAuth(user_id),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    ) as response:
        return (await response.json())["task_id"]


async def create_async_tasks(
    endpoint_name: str, create_async_task_requests: List[Dict[str, Any]], user_id: str
):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for create_async_task_request in create_async_task_requests:
            task = create_async_task(endpoint_name, create_async_task_request, user_id, session)
            tasks.append(asyncio.create_task(task))

        result = await asyncio.gather(*tasks)
        return result


async def get_async_task(
    endpoint_name: str, task_id: str, user_id: str, session: aiohttp.ClientSession
) -> Dict[str, Any]:
    async with session.get(
        f"{BASE_PATH}/endpoints/{endpoint_name}/task_async/{task_id}",
        auth=aiohttp.BasicAuth(user_id),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    ) as response:
        return await response.json()


async def get_async_tasks(endpoint_name: str, task_ids: List[str], user_id: str):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for task_id in task_ids:
            task = get_async_task(endpoint_name, task_id, user_id, session)
            tasks.append(asyncio.create_task(task))

        result = await asyncio.gather(*tasks)
        return result


def create_model_endpoint(
    create_model_endpoint_request: Dict[str, Any], user_id: str
) -> Dict[str, Any]:
    response = requests.post(
        f"{BASE_PATH}/endpoints",
        json=create_model_endpoint_request,
        headers={"Content-Type": "application/json"},
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


def update_model_endpoint(
    endpoint_name: str, update_model_endpoint_request: Dict[str, Any], user_id: str
) -> Dict[str, Any]:
    response = requests.put(
        f"{BASE_PATH}/endpoints/{endpoint_name}",
        json=update_model_endpoint_request,
        headers={"Content-Type": "application/json"},
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def get_model_endpoint(endpoint_name: str, user_id: str) -> Dict[str, Any]:
    response = requests.get(
        f"{BASE_PATH}/endpoints/{endpoint_name}",
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


def delete_model_endpoint(endpoint_name: str, user_id: str) -> Dict[str, Any]:
    response = requests.delete(
        f"{BASE_PATH}/endpoints/{endpoint_name}",
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def list_model_bundles(user_id: str) -> List[Dict[str, Any]]:
    response = requests.get(
        f"{BASE_PATH}/model_bundle",
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()["bundles"]


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def list_model_endpoints(user_id: str) -> List[Dict[str, Any]]:
    response = requests.get(
        f"{BASE_PATH}/endpoints",
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()["endpoints"]


# Wait 20 minutes (1200 seconds) for endpoints to build.
@retry(stop=stop_after_attempt(120), wait=wait_fixed(10))
def ensure_n_ready_endpoints_long(n: int, user_id: str):
    endpoints = list_model_endpoints(user_id)
    ready_endpoints = [endpoint for endpoint in endpoints if endpoint["status"] == "READY"]
    print(
        f"User {user_id} Current num endpoints: {len(endpoints)}, num ready endpoints: {len(ready_endpoints)}"
    )
    assert len(ready_endpoints) >= n


# Wait 2 minutes (120 seconds) for endpoints to build.
@retry(stop=stop_after_attempt(12), wait=wait_fixed(10))
def ensure_n_ready_endpoints_short(n: int, user_id: str):
    endpoints = list_model_endpoints(user_id)
    ready_endpoints = [endpoint for endpoint in endpoints if endpoint["status"] == "READY"]
    print(
        f"User {user_id} Current num endpoints: {len(endpoints)}, num ready endpoints: {len(ready_endpoints)}"
    )
    assert len(ready_endpoints) >= n


def delete_all_bundles_and_endpoints(user_id):
    endpoints = list_model_endpoints(user_id)
    for endpoint in endpoints:
        response = delete_model_endpoint(endpoint["name"], user_id)
        assert response["deleted"] == "true"
    bundles = list_model_bundles(user_id)
    for bundle in bundles:
        response = delete_model_bundle(bundle["name"], user_id)
        assert response["deleted"] == "true"


# Wait up to 3 minutes (300 seconds) for the gateway to be ready.
@retry(stop=stop_after_attempt(30), wait=wait_fixed(10))
def ensure_gateway_ready():
    response = requests.get(f"{BASE_PATH}/healthz")
    assert response.ok


# Wait up to 3 minutes (180 seconds) for the pods to spin up.
@retry(stop=stop_after_attempt(60), wait=wait_fixed(3))
def ensure_nonzero_available_workers(endpoint_name: str, user_id: str):
    simple_endpoint = get_model_endpoint(endpoint_name, user_id)
    assert simple_endpoint.get("worker_settings", {}).get("available_workers", 0)


# Wait up to 30 seconds for the tasks to be returned.
@retry(stop=stop_after_attempt(30), wait=wait_fixed(1))
def ensure_all_tasks_success(endpoint_name: str, task_ids: List[str], user_id: str):
    responses = asyncio.run(get_async_tasks(endpoint_name, task_ids, user_id))
    for response in responses:
        assert response["state"] == "SUCCESS"


def e2e_test(should_assert_async_tasks_success: bool):
    ensure_gateway_ready()

    print(f"Deleting all bundles and endpoints for user with ID {USER_ID_0}")
    delete_all_bundles_and_endpoints(USER_ID_0)
    print(f"Deleting all bundles and endpoints for user with ID {USER_ID_1}")
    delete_all_bundles_and_endpoints(USER_ID_1)

    print("Creating model bundles...")
    create_model_bundle(CREATE_MODEL_BUNDLE_REQUEST_SIMPLE, USER_ID_0)
    create_model_bundle(CREATE_MODEL_BUNDLE_REQUEST_SIMPLE, USER_ID_1)
    create_model_bundle(CREATE_MODEL_BUNDLE_REQUEST_CUSTOM_IMAGE, USER_ID_0)
    create_model_bundle(CREATE_MODEL_BUNDLE_REQUEST_CUSTOM_IMAGE, USER_ID_1)

    print("Creating model endpoints...")
    create_model_endpoint(CREATE_MODEL_ENDPOINT_REQUEST_SIMPLE, USER_ID_0)
    create_model_endpoint(CREATE_MODEL_ENDPOINT_REQUEST_CUSTOM_IMAGE, USER_ID_0)
    ensure_n_ready_endpoints_long(2, USER_ID_0)

    create_model_endpoint(CREATE_MODEL_ENDPOINT_REQUEST_SIMPLE, USER_ID_1)
    create_model_endpoint(CREATE_MODEL_ENDPOINT_REQUEST_CUSTOM_IMAGE, USER_ID_1)
    # Endpoint builds should be cached now.
    ensure_n_ready_endpoints_short(2, USER_ID_1)

    print("Updating model endpoints...")
    update_model_endpoint("model-endpoint-simple", UPDATE_MODEL_ENDPOINT_REQUEST_SIMPLE, USER_ID_0)
    update_model_endpoint("model-endpoint-simple", UPDATE_MODEL_ENDPOINT_REQUEST_SIMPLE, USER_ID_1)
    update_model_endpoint(
        "model-endpoint-custom-image",
        UPDATE_MODEL_ENDPOINT_REQUEST_CUSTOM_IMAGE,
        USER_ID_0,
    )
    update_model_endpoint(
        "model-endpoint-custom-image",
        UPDATE_MODEL_ENDPOINT_REQUEST_CUSTOM_IMAGE,
        USER_ID_1,
    )

    print("Waiting for updated model endpoints to build...")
    # Endpoint builds should be cached now.
    ensure_n_ready_endpoints_short(2, USER_ID_0)
    ensure_n_ready_endpoints_short(2, USER_ID_1)

    print("Sending async tasks...")
    create_async_task_requests = [CREATE_ASYNC_TASK] * 5
    task_ids_0 = asyncio.run(
        create_async_tasks("model-endpoint-simple", create_async_task_requests, USER_ID_0)
    )
    task_ids_1 = asyncio.run(
        create_async_tasks("model-endpoint-simple", create_async_task_requests, USER_ID_1)
    )

    # Note that ideally we would like to retrieve the results of these async tasks but unfortunately
    # in CircleCI we cannot do so because the deployments created by the service builder would need
    # to have AWS auth as well, which requires further work to add in AWS access keys.
    print("Retrieving async task results...")
    asyncio.run(get_async_tasks("model-endpoint-simple", task_ids_0, USER_ID_0))
    if should_assert_async_tasks_success:
        ensure_nonzero_available_workers("model-endpoint-simple", USER_ID_0)
        ensure_all_tasks_success("model-endpoint-simple", task_ids_0, USER_ID_0)
        ensure_nonzero_available_workers("model-endpoint-simple", USER_ID_1)
        ensure_all_tasks_success("model-endpoint-simple", task_ids_1, USER_ID_1)

    print("Deleting model endpoints...")
    delete_model_endpoint("model-endpoint-simple", USER_ID_0)
    delete_model_endpoint("model-endpoint-simple", USER_ID_1)
    delete_model_endpoint("model-endpoint-custom-image", USER_ID_0)
    delete_model_endpoint("model-endpoint-custom-image", USER_ID_1)

    print("Deleting model bundles...")
    delete_model_bundle("model_bundle_simple", USER_ID_0)
    delete_model_bundle("model_bundle_simple", USER_ID_1)
    delete_model_bundle("model_bundle_custom_image", USER_ID_0)
    delete_model_bundle("model_bundle_custom_image", USER_ID_1)

    print("All tests passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-async-tasks",
        "-c",
        action="store_true",
        help="If set, asserts that the async tasks return SUCCESS as their status.",
    )
    args, unknown = parser.parse_known_args()
    e2e_test(args.check_async_tasks)
