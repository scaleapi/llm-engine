# Tests the gateway and service builder version 1 for an end-to-end workflow.
#
# This includes creating bundles, building and updating endpoints, sending requests to the
# endpoints, and deleting the endpoints.
#
# This test expects a local Launch gateway service listening at http://localhost:5001.
#
# The Launch client is not used to keep this test free of external dependencies.

import argparse
import asyncio
import json
import os
import time
from typing import Any, Dict, List, Sequence

import aiohttp
import aioredis
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from spellbook_serve.common.constants import (
    FEATURE_FLAG_USE_MULTI_CONTAINER_ARCHITECTURE_FOR_ARTIFACTLIKE_BUNDLE,
)
from spellbook_serve.common.dtos.tasks import TaskStatus
from spellbook_serve.core.config import ml_infra_config

_DEFAULT_BASE_PATH = "http://localhost:5001"
BASE_PATH = os.environ.get("BASE_PATH", _DEFAULT_BASE_PATH)
print(f"Integration tests using gateway {BASE_PATH=}")
DEFAULT_NETWORK_TIMEOUT_SEC = 10
REDIS_URI = "redis://localhost:6379/15"

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

DEFAULT_USERS: Sequence[str] = (
    USER_ID_0,
    USER_ID_1,
)

S3_BUCKET = ml_infra_config().s3_bucket


CREATE_MODEL_BUNDLE_REQUEST_SIMPLE = {
    "packaging_type": "cloudpickle",
    "name": "model_bundle_simple",
    "location": f"s3://{S3_BUCKET}/scale-launch/model_bundles/646c085a6b38de00808148bb/84de9d96-6140-497b-be36-dcc8dd1dd9c4",
    "metadata": {
        "load_predict_fn": """def my_load_predict_fn(model):
    def echo(**keyword_args):
        return model(**keyword_args)
    return echo""",
        "load_model_fn": """def my_load_model_fn():
    def my_model(**keyword_args):
        return {k: v for k, v in keyword_args.items()}
    return my_model""",
    },
    "requirements": [],
    "env_params": {
        "framework_type": "pytorch",
        "pytorch_image_tag": "1.7.1-cuda11.0-cudnn8-runtime",
    },
}

# Taken from https://github.com/scaleapi/models/blob/feb133e55a4373587a8996a945ab20445688811d/autoflag/deploy/linter_loss/service_config.py
CREATE_MODEL_BUNDLE_REQUEST_CUSTOM_IMAGE = {
    "packaging_type": "zip",
    "name": "model_bundle_custom_image",
    "location": f"s3://{S3_BUCKET}/scale-launch/model_bundles/63bf16d877db8a663d044aa3/b7ccd939-e593-4f9f-89bf-0a7b9cc39df1",
    "metadata": {
        "load_predict_fn_module_path": "autoflag.deploy.linter_loss.load_predict_fn.load_predict_fn",
        "load_model_fn_module_path": "autoflag.deploy.linter_loss.load_model_fn.load_model_fn",
    },
    "requirements": [
        "awscli==1.25.46",
        "scale-nucleus==0.14.10",
        "opencv-python==4.6.0.66",
        "pycocotools==2.0.4",
        "pytorch-lightning==1.6.4",
        "pywise==0.4.0",
        "scikit-image==0.19.3",
        "torchdata==0.4.0",
        "fsspec[s3]==2022.5.0",
    ],
    "env_params": {
        "framework_type": "custom_base_image",
        "ecr_repo": "autoflag-gpu",
        "image_tag": "d6a18829c8b4dd6e47d34ed0d1c32e9474ce2bb7",
    },
    "app_config": {"linter_loss_name": "estimated-risk-semseg"},
}

CREATE_MODEL_BUNDLE_REQUEST_RUNNABLE_IMAGE = {
    "name": "model_bundle_runnable_image",
    "schema_location": f"s3://{S3_BUCKET}/scale-launch/model_bundles/63bf16d877db8a663d044aa3/b7ccd939-e593-4f9f-89bf-0a7b9cc39df1",
    "metadata": {
        "test_key": "test_value",
    },
    "flavor": {
        "flavor": "runnable_image",
        "repository": "launch/gateway",
        "tag": "fb01f90a15f2826792d75c0ae0eaefa4215eb975",
        "command": [
            "dumb-init",
            "--",
            "ddtrace-run",
            "run-service",
            "--config",
            "/workspace/std-ml-srv/tests/resources/example_echo_service_configuration.yaml",
            "--concurrency",
            "1",
            "--http",
            "production",
            "--port",
            "5005",
        ],
        "env": {
            "TEST_KEY": "test_value",
            # infra configs are mounted here
            "ML_INFRA_SERVICES_CONFIG_PATH": "/workspace/ml_infra_core/spellbook_serve.core/spellbook_serve.core/configs/circleci.yaml",
            "HTTP_HOST": "0.0.0.0",  # Hack for waitress to work in minikube
        },
        "protocol": "http",
        "readiness_initial_delay_seconds": 20,
    },
}

CREATE_MODEL_BUNDLE_REQUEST_STREAMING_IMAGE = {
    "name": "model_bundle_streaming_image",
    "schema_location": f"s3://{S3_BUCKET}/scale-launch/model_bundles/63bf16d877db8a663d044aa3/b7ccd939-e593-4f9f-89bf-0a7b9cc39df1",
    "metadata": {
        "test_key": "test_value",
    },
    "flavor": {
        "flavor": "streaming_enhanced_runnable_image",
        "repository": "launch/gateway",
        "tag": "fb01f90a15f2826792d75c0ae0eaefa4215eb975",
        "command": [
            "dumb-init",
            "--",
            "ddtrace-run",
            "run-service",
            "--config",
            "/workspace/std-ml-srv/tests/resources/example_echo_service_configuration.yaml",
            "--concurrency",
            "1",
            "--http",
            "production",
            "--port",
            "5005",
        ],
        "streaming_command": [
            "dumb-init",
            "--",
            "ddtrace-run",
            "run-streamer",
            "--config",
            "/workspace/std-ml-srv/tests/resources/example_echo_streaming_service_configuration.yaml",
            "--concurrency",
            "1",
            "--http",
            "production",
            "--port",
            "5005",
        ],
        "env": {
            "TEST_KEY": "test_value",
            "ML_INFRA_SERVICES_CONFIG_PATH": "/workspace/ml_infra_core/spellbook_serve.core/spellbook_serve.core/configs/circleci.yaml",
            # infra configs are mounted here
            "HTTP_HOST": "0.0.0.0",  # Hack for uvicorn to work in minikube
        },
        "protocol": "http",
        "readiness_initial_delay_seconds": 20,
    },
}

CREATE_MODEL_BUNDLE_REQUEST_SYNC_STREAMING_IMAGE = {
    "name": "model_bundle_sync_streaming_image",
    "schema_location": f"s3://{S3_BUCKET}/scale-launch/model_bundles/63bf16d877db8a663d044aa3/b7ccd939-e593-4f9f-89bf-0a7b9cc39df1",
    "metadata": {
        "test_key": "test_value",
    },
    "flavor": {
        "flavor": "streaming_enhanced_runnable_image",
        "repository": "launch/gateway",
        "tag": "fb01f90a15f2826792d75c0ae0eaefa4215eb975",
        "command": [
            "dumb-init",
            "--",
            "ddtrace-run",
            "python",
            "-m",
            "ml_serve.echo_server",
            "--port",
            "5005",
        ],
        "streaming_command": [
            "dumb-init",
            "--",
            "ddtrace-run",
            "python",
            "-m",
            "ml_serve.echo_server",
            "--port",
            "5005",
        ],
        "env": {
            "TEST_KEY": "test_value",
            "ML_INFRA_SERVICES_CONFIG_PATH": "/workspace/ml_infra_core/spellbook_serve.core/spellbook_serve.core/configs/circleci.yaml",
            # infra configs are mounted here
            "HTTP_HOST": "0.0.0.0",  # Hack for uvicorn to work in minikube
        },
        "protocol": "http",
        "readiness_initial_delay_seconds": 20,
    },
}

CREATE_ASYNC_MODEL_ENDPOINT_REQUEST_SIMPLE = {
    "bundle_name": "model_bundle_simple",
    "name": "model-endpoint-simple-async",
    "endpoint_type": "async",
    "cpus": "0.5",
    "memory": "500Mi",
    "min_workers": 1,
    "max_workers": 1,
    "gpus": 0,
    "per_worker": 1,
    "labels": {"team": "infra", "product": "launch"},
    "metadata": {},
}

CREATE_SYNC_MODEL_ENDPOINT_REQUEST_SIMPLE = CREATE_ASYNC_MODEL_ENDPOINT_REQUEST_SIMPLE.copy()
CREATE_SYNC_MODEL_ENDPOINT_REQUEST_SIMPLE["name"] = "model-endpoint-simple-sync"
CREATE_SYNC_MODEL_ENDPOINT_REQUEST_SIMPLE["endpoint_type"] = "sync"

CREATE_ASYNC_MODEL_ENDPOINT_REQUEST_CUSTOM_IMAGE = {
    "bundle_name": "model_bundle_custom_image",
    "name": "model-endpoint-custom-image-async",
    "post_inference_hooks": [],
    "endpoint_type": "async",
    "cpus": "3",
    "gpu_type": "nvidia-tesla-t4",
    "gpus": 1,
    "memory": "6Gi",
    "optimize_costs": True,
    "min_workers": 0,
    "max_workers": 1,
    "per_worker": 1,
    "labels": {"team": "infra", "product": "launch"},
    "metadata": {"key": "value"},
}

CREATE_SYNC_MODEL_ENDPOINT_REQUEST_CUSTOM_IMAGE = (
    CREATE_ASYNC_MODEL_ENDPOINT_REQUEST_CUSTOM_IMAGE.copy()
)
CREATE_SYNC_MODEL_ENDPOINT_REQUEST_CUSTOM_IMAGE["name"] = "model-endpoint-custom-image-sync"
CREATE_SYNC_MODEL_ENDPOINT_REQUEST_CUSTOM_IMAGE["endpoint_type"] = "sync"
CREATE_SYNC_MODEL_ENDPOINT_REQUEST_CUSTOM_IMAGE["min_workers"] = 1

CREATE_ASYNC_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE = {
    "bundle_name": "model_bundle_runnable_image",
    "name": "model-endpoint-runnable-image-async",
    "post_inference_hooks": [],
    "endpoint_type": "async",
    "cpus": "1",
    "gpus": 0,
    "memory": "1Gi",
    "optimize_costs": True,
    "min_workers": 1,
    "max_workers": 1,
    "per_worker": 1,
    "labels": {"team": "infra", "product": "launch"},
    "metadata": {"key": "value"},
}

CREATE_SYNC_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE = (
    CREATE_ASYNC_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE.copy()
)
CREATE_SYNC_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE["name"] = "model-endpoint-runnable-image-sync"
CREATE_SYNC_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE["endpoint_type"] = "sync"
CREATE_SYNC_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE["min_workers"] = 1

CREATE_STREAMING_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE = (
    CREATE_ASYNC_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE.copy()
)
CREATE_STREAMING_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE[
    "name"
] = "model-endpoint-runnable-image-streaming"
CREATE_STREAMING_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE[
    "bundle_name"
] = "model_bundle_streaming_image"
CREATE_STREAMING_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE["endpoint_type"] = "streaming"
CREATE_STREAMING_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE["min_workers"] = 1

CREATE_SYNC_STREAMING_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE = (
    CREATE_STREAMING_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE.copy()
)
SYNC_STREAMING_MODEL_ENDPOINT_NAME = "model-endpoint-runnable-image-sync-streaming"
CREATE_SYNC_STREAMING_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE[
    "name"
] = SYNC_STREAMING_MODEL_ENDPOINT_NAME
CREATE_SYNC_STREAMING_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE[
    "bundle_name"
] = "model_bundle_sync_streaming_image"

UPDATE_MODEL_ENDPOINT_REQUEST_SIMPLE = {
    "bundle_name": "model_bundle_simple",
    "cpus": "0.25",
    "max_workers": 2,
}

UPDATE_MODEL_ENDPOINT_REQUEST_CUSTOM_IMAGE = {
    "bundle_name": "model_bundle_custom_image",
    "cpus": "1",
    "max_workers": 2,
}

UPDATE_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE = {
    "bundle_name": "model_bundle_runnable_image",
    "cpus": "2",
    "max_workers": 2,
}

UPDATE_MODEL_ENDPOINT_REQUEST_STREAMING_IMAGE = {
    "bundle_name": "model_bundle_streaming_image",
    "cpus": "2",
    "max_workers": 2,
}

INFERENCE_PAYLOAD: Dict[str, Any] = {
    "args": {"y": 1},
    "url": None,
}

INFERENCE_PAYLOAD_RETURN_PICKLED_FALSE: Dict[str, Any] = INFERENCE_PAYLOAD.copy()
INFERENCE_PAYLOAD_RETURN_PICKLED_FALSE["return_pickled"] = False

INFERENCE_PAYLOAD_RETURN_PICKLED_TRUE: Dict[str, Any] = INFERENCE_PAYLOAD.copy()
INFERENCE_PAYLOAD_RETURN_PICKLED_TRUE["return_pickled"] = True

CREATE_BATCH_JOB_REQUEST: Dict[str, Any] = {
    "bundle_name": "model_bundle_simple",
    "input_path": f"s3://{S3_BUCKET}/launch/batch-jobs/inputs/test-input.json",
    "serialization_format": "JSON",
    "labels": {"team": "infra", "product": "launch"},
    "resource_requests": {
        "memory": "500Mi",
        "max_workers": 1,
        "gpus": 0,
    },
}

CREATE_DOCKER_IMAGE_BATCH_JOB_BUNDLE_REQUEST: Dict[str, Any] = {
    "name": "di_batch_job_bundle_1",
    "image_repository": "launch/gateway",
    "image_tag": "fb01f90a15f2826792d75c0ae0eaefa4215eb975",
    "command": ["jq", ".", "/launch_mount_location/file"],
    "env": {"ENV1": "VAL1"},
    "mount_location": "/launch_mount_location/file",
    "resource_requests": {
        "cpus": 0.1,
        "memory": "10Mi",
    },
}

CREATE_DOCKER_IMAGE_BATCH_JOB_REQUEST: Dict[str, Any] = {
    "docker_image_batch_job_bundle_name": "di_batch_job_bundle_1",
    "job_config": {"data": {"to": "mount"}},
    "labels": {"team": "infra", "product": "testing"},
    "resource_requests": {"cpus": 0.15, "memory": "15Mi"},
}


def create_model_bundle(
    create_model_bundle_request: Dict[str, Any], user_id: str, version: str
) -> Dict[str, Any]:
    response = requests.post(
        f"{BASE_PATH}/{version}/model-bundles",
        json=create_model_bundle_request,
        headers={"Content-Type": "application/json"},
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def get_latest_model_bundle(model_name: str, user_id: str, version: str) -> Dict[str, Any]:
    response = requests.get(
        f"{BASE_PATH}/{version}/model-bundles/latest?model_name={model_name}",
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


def get_or_create_model_bundle(
    create_model_bundle_request: Dict[str, Any], user_id: str, version: str
) -> Dict[str, Any]:
    # In v1, we will no longer have the uniqueness constraint of (name, created_by) but right now
    # for backwards compatibility, such a constraint exists. As a result, we use this get-or-create
    # method as a temporary workaround since v1 will not support bundle deletion initially.
    try:
        return get_latest_model_bundle(create_model_bundle_request["name"], user_id, version)
    except:
        return create_model_bundle(create_model_bundle_request, user_id, version)


def replace_model_bundle_name_with_id(request: Dict[str, Any], user_id: str, version):
    if "bundle_name" in request:
        model_bundle = get_latest_model_bundle(request["bundle_name"], user_id, version)
        request["model_bundle_id"] = model_bundle["id"]
        del request["bundle_name"]


def create_model_endpoint(
    create_model_endpoint_request: Dict[str, Any], user_id: str
) -> Dict[str, Any]:
    create_model_endpoint_request = create_model_endpoint_request.copy()
    replace_model_bundle_name_with_id(create_model_endpoint_request, user_id, "v1")
    response = requests.post(
        f"{BASE_PATH}/v1/model-endpoints",
        json=create_model_endpoint_request,
        headers={"Content-Type": "application/json"},
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


def create_batch_job(create_batch_job_request: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    create_batch_job_request = create_batch_job_request.copy()
    replace_model_bundle_name_with_id(create_batch_job_request, user_id, "v2")
    response = requests.post(
        f"{BASE_PATH}/v1/batch-jobs",
        json=create_batch_job_request,
        headers={"Content-Type": "application/json"},
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


def create_docker_image_batch_job_bundle(
    create_docker_image_batch_job_bundle_request: Dict[str, Any], user_id: str
) -> Dict[str, Any]:
    response = requests.post(
        f"{BASE_PATH}/v1/docker-image-batch-job-bundles",
        json=create_docker_image_batch_job_bundle_request,
        headers={"Content-Type": "application/json"},
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


def get_latest_docker_image_batch_job_bundle(bundle_name: str, user_id: str) -> Dict[str, Any]:
    response = requests.get(
        f"{BASE_PATH}/v1/docker-image-batch-job-bundles/latest?bundle_name={bundle_name}",
        headers={"Content-Type": "application/json"},
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


def get_or_create_docker_image_batch_job_bundle(
    create_docker_image_batch_job_bundle_request: Dict[str, Any], user_id: str
):
    try:
        return get_latest_docker_image_batch_job_bundle(
            create_docker_image_batch_job_bundle_request["name"], user_id
        )
    except:
        return create_docker_image_batch_job_bundle(
            create_docker_image_batch_job_bundle_request, user_id
        )


def get_docker_image_batch_job_bundle_by_id(
    docker_image_batch_job_bundle_id: str, user_id: str
) -> Dict[str, Any]:
    response = requests.get(
        f"{BASE_PATH}/v1/docker-image-batch-job-bundles/{docker_image_batch_job_bundle_id}",
        headers={"Content-Type": "application/json"},
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


def create_docker_image_batch_job(
    create_docker_image_batch_job_request: Dict[str, Any], user_id: str
) -> Dict[str, Any]:
    response = requests.post(
        f"{BASE_PATH}/v1/docker-image-batch-jobs",
        json=create_docker_image_batch_job_request,
        headers={"Content-Type": "application/json"},
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


def get_docker_image_batch_job(batch_job_id: str, user_id: str) -> Dict[str, Any]:
    response = requests.get(
        f"{BASE_PATH}/v1/docker-image-batch-jobs/{batch_job_id}",
        headers={"Content-Type": "application/json"},
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


@retry(stop=stop_after_attempt(6), wait=wait_fixed(1))
def get_model_endpoint(name: str, user_id: str) -> Dict[str, Any]:
    response = requests.get(
        f"{BASE_PATH}/v1/model-endpoints?name={name}",
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()["model_endpoints"][0]


def update_model_endpoint(
    endpoint_name: str, update_model_endpoint_request: Dict[str, Any], user_id: str
) -> Dict[str, Any]:
    update_model_endpoint_request = update_model_endpoint_request.copy()
    replace_model_bundle_name_with_id(update_model_endpoint_request, user_id, "v2")
    endpoint = get_model_endpoint(endpoint_name, user_id)
    response = requests.put(
        f"{BASE_PATH}/v1/model-endpoints/{endpoint['id']}",
        json=update_model_endpoint_request,
        headers={"Content-Type": "application/json"},
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


def delete_model_endpoint(endpoint_name: str, user_id: str) -> Dict[str, Any]:
    endpoint = get_model_endpoint(endpoint_name, user_id)
    response = requests.delete(
        f"{BASE_PATH}/v1/model-endpoints/{endpoint['id']}",
        headers={"Content-Type": "application/json"},
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def list_model_endpoints(user_id: str) -> List[Dict[str, Any]]:
    response = requests.get(
        f"{BASE_PATH}/v1/model-endpoints",
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()["model_endpoints"]


async def create_async_task(
    model_endpoint_id: str,
    create_async_task_request: Dict[str, Any],
    user_id: str,
    session: aiohttp.ClientSession,
) -> str:
    async with session.post(
        f"{BASE_PATH}/v1/async-tasks?model_endpoint_id={model_endpoint_id}",
        json=create_async_task_request,
        headers={"Content-Type": "application/json"},
        auth=aiohttp.BasicAuth(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    ) as response:
        return (await response.json())["task_id"]


async def create_async_tasks(
    endpoint_name: str, create_async_task_requests: List[Dict[str, Any]], user_id: str
) -> List[Any]:
    endpoint = get_model_endpoint(endpoint_name, user_id)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for create_async_task_request in create_async_task_requests:
            task = create_async_task(endpoint["id"], create_async_task_request, user_id, session)
            tasks.append(asyncio.create_task(task))

        result = await asyncio.gather(*tasks)
        return result  # type: ignore


async def create_sync_task(
    model_endpoint_id: str,
    create_sync_task_request: Dict[str, Any],
    user_id: str,
    session: aiohttp.ClientSession,
) -> str:
    async with session.post(
        f"{BASE_PATH}/v1/sync-tasks?model_endpoint_id={model_endpoint_id}",
        json=create_sync_task_request,
        headers={"Content-Type": "application/json"},
        auth=aiohttp.BasicAuth(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    ) as response:
        assert response.status == 200, (await response.read()).decode()
        return await response.json()


async def create_streaming_task(
    model_endpoint_id: str,
    create_streaming_task_request: Dict[str, Any],
    user_id: str,
    session: aiohttp.ClientSession,
) -> str:
    async with session.post(
        f"{BASE_PATH}/v1/streaming-tasks?model_endpoint_id={model_endpoint_id}",
        json=create_streaming_task_request,
        headers={"Content-Type": "application/json"},
        auth=aiohttp.BasicAuth(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    ) as response:
        assert response.status == 200, (await response.read()).decode()
        return (await response.read()).decode()


async def create_sync_tasks(
    endpoint_name: str, create_sync_task_requests: List[Dict[str, Any]], user_id: str
) -> List[Any]:
    endpoint = get_model_endpoint(endpoint_name, user_id)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for create_sync_task_request in create_sync_task_requests:
            task = create_sync_task(endpoint["id"], create_sync_task_request, user_id, session)
            tasks.append(asyncio.create_task(task))

        result = await asyncio.gather(*tasks)
        return result  # type: ignore


async def create_streaming_tasks(
    endpoint_name: str, create_streaming_task_requests: List[Dict[str, Any]], user_id: str
) -> List[Any]:
    endpoint = get_model_endpoint(endpoint_name, user_id)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for create_streaming_task_request in create_streaming_task_requests:
            task = create_streaming_task(
                endpoint["id"], create_streaming_task_request, user_id, session
            )
            tasks.append(asyncio.create_task(task))

        result = await asyncio.gather(*tasks)
        return result  # type: ignore


async def get_async_task(
    task_id: str, user_id: str, session: aiohttp.ClientSession
) -> Dict[str, Any]:
    async with session.get(
        f"{BASE_PATH}/v1/async-tasks/{task_id}",
        auth=aiohttp.BasicAuth(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    ) as response:
        return await response.json()


async def get_async_tasks(task_ids: List[str], user_id: str) -> List[Dict[str, Any]]:
    async with aiohttp.ClientSession() as session:
        tasks = []
        for task_id in task_ids:
            task = get_async_task(task_id, user_id, session)
            tasks.append(asyncio.create_task(task))

        result = await asyncio.gather(*tasks)
        return result  # type: ignore


# Wait 25 minutes (1500 seconds) for endpoints to build.
@retry(stop=stop_after_attempt(25), wait=wait_fixed(60))
def ensure_n_ready_endpoints_long(n: int, user_id: str):
    endpoints = list_model_endpoints(user_id)
    ready_endpoints = [endpoint for endpoint in endpoints if endpoint["status"] == "READY"]
    print(
        f"User {user_id} Current num endpoints: {len(endpoints)}, num ready endpoints: {len(ready_endpoints)}"
    )
    assert (
        len(ready_endpoints) >= n
    ), f"Expected {n} ready endpoints, got {len(ready_endpoints)}. Look through endpoint builder for errors."


# Wait 2 minutes (120 seconds) for endpoints to build.
@retry(stop=stop_after_attempt(12), wait=wait_fixed(10))
def ensure_n_ready_endpoints_short(n: int, user_id: str):
    endpoints = list_model_endpoints(user_id)
    ready_endpoints = [endpoint for endpoint in endpoints if endpoint["status"] == "READY"]
    print(
        f"User {user_id} Current num endpoints: {len(endpoints)}, num ready endpoints: {len(ready_endpoints)}"
    )
    assert len(ready_endpoints) >= n


def delete_all_endpoints(user_id):
    endpoints = list_model_endpoints(user_id)
    for i, endpoint in enumerate(endpoints):
        response = delete_model_endpoint(endpoint["name"], user_id)
        assert response["deleted"]
        print(f"[{i + 1}/{len(endpoints)}] Deleted {endpoint=}")


# Wait up to 5 minutes (300 seconds) for the gateway to be ready.
@retry(stop=stop_after_attempt(30), wait=wait_fixed(10))
def ensure_gateway_ready():
    response = requests.get(f"{BASE_PATH}/healthz")
    assert response.ok


# Wait up to 10 minutes (600 seconds) for the pods to spin up.
@retry(stop=stop_after_attempt(200), wait=wait_fixed(3))
def ensure_nonzero_available_workers(endpoint_name: str, user_id: str):
    simple_endpoint = get_model_endpoint(endpoint_name, user_id)
    assert simple_endpoint.get("deployment_state", {}).get("available_workers", 0)


def ensure_inference_task_response_is_correct(response: Dict[str, Any], return_pickled: bool):
    print(response)
    assert response["status"] == "SUCCESS"
    assert response["traceback"] is None
    if return_pickled:
        assert response["result"]["result_url"].startswith("s3://")
    else:
        assert response["result"] == {"result": '{"y": 1}'}


# Wait up to 30 seconds for the tasks to be returned.
@retry(
    stop=stop_after_attempt(30), wait=wait_fixed(1), retry=retry_if_exception_type(AssertionError)
)
def ensure_all_async_tasks_success(task_ids: List[str], user_id: str, return_pickled: bool):
    responses = asyncio.run(get_async_tasks(task_ids, user_id))
    for response in responses:
        if response["status"] not in (TaskStatus.PENDING, TaskStatus.SUCCESS, TaskStatus.STARTED):
            print(response)
            raise ValueError("Task failed!")
        ensure_inference_task_response_is_correct(response, return_pickled)


def delete_existing_endpoints(users: Sequence[str] = DEFAULT_USERS) -> None:
    if len(users) == 0:
        raise ValueError("Must supply at least one user!")

    # list all endpoints before attempting to delete them
    print(f"[{len({users})} ] Listing all user endpoints... ({users})")
    all_endpoint_info = []
    for i, u in enumerate(users):
        u_endpoints = list_model_endpoints(u)
        all_endpoint_info.append(u_endpoints)
        k8s_endpoint_names = [
            f"launch-endpoint-id-{endpoint['id'].replace('_', '-')}" for endpoint in u_endpoints
        ]
        print(
            f"[{i + 1}/{len(users)}] {len(u_endpoints)} endpoints for user {u}: {k8s_endpoint_names}"
        )

    # delete the endpoints: if this fails, manually remove the dangling k8s deployments
    # and delete the user's endpoints from the spellbook_serve.endpoints table
    # i.e. by default this is running the following SQL:
    #
    # >>>> delete from spellbook_serve.endpoints where created_by in (
    # >>>>       'test00000000000000000000',
    # >>>>       'test11111111111111111111',
    # >>>> )
    #
    time.sleep(15)  # need to sleep to allow the cache to refresh
    print(f"[{len({users})}] Deleting all user endpoints...")
    try:
        for i, u in enumerate(users):
            print(f"[{i + 1}/{len(users)}] Deleting all endpoints for user with ID {u}")
            delete_all_endpoints(u)
    except Exception:  # noqa
        try:
            j: str = json.dumps(all_endpoint_info, indent=2)
        except Exception as j_error:  # noqa
            j = f"[FAILED TO JSON ENCODE {j_error}]\n{all_endpoint_info}"
        barrier: str = "-" * 80
        print(f"ERROR! Deletion failed. All endpoint information:\n{barrier}\n{j}\n{barrier}")
        raise


def e2e_test_batch_jobs(users: Sequence[str] = DEFAULT_USERS) -> None:
    print("[Batch] Creating batch job bundles...")
    for u in users:
        get_or_create_docker_image_batch_job_bundle(CREATE_DOCKER_IMAGE_BATCH_JOB_BUNDLE_REQUEST, u)

    create_batch_job(CREATE_BATCH_JOB_REQUEST, users[0])  # USER_ID_0 by default

    create_docker_image_batch_job(CREATE_DOCKER_IMAGE_BATCH_JOB_REQUEST, users[0])

    # TODO: assert that batch job actually succeeds.


def e2e_test_async_endpoints(
    should_assert_task_success: bool, users: Sequence[str] = DEFAULT_USERS
) -> None:
    # Assumes that delete_endpoints went first, and so endpoints are already deleted prior to entering.
    if len(users) == 0:
        raise ValueError("Must supply at least one user!")

    try:
        print("[Async] Creating model bundles...")
        for u in users:
            get_or_create_model_bundle(CREATE_MODEL_BUNDLE_REQUEST_SIMPLE, u, "v1")
            get_or_create_model_bundle(CREATE_MODEL_BUNDLE_REQUEST_CUSTOM_IMAGE, u, "v1")
            get_or_create_model_bundle(CREATE_MODEL_BUNDLE_REQUEST_RUNNABLE_IMAGE, u, "v2")

        print("[Async] Creating model endpoints...")
        for u in users:
            create_model_endpoint(CREATE_ASYNC_MODEL_ENDPOINT_REQUEST_SIMPLE, u)
            create_model_endpoint(CREATE_ASYNC_MODEL_ENDPOINT_REQUEST_CUSTOM_IMAGE, u)
            create_model_endpoint(CREATE_ASYNC_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE, u)
            ensure_n_ready_endpoints_long(1, u)
            ensure_n_ready_endpoints_long(2, u)
            ensure_n_ready_endpoints_long(3, u)

        print("[Async] Updating model endpoints...")
        for u in users:
            update_model_endpoint(
                "model-endpoint-simple-async",
                UPDATE_MODEL_ENDPOINT_REQUEST_SIMPLE,
                u,
            )
            update_model_endpoint(
                "model-endpoint-custom-image-async",
                UPDATE_MODEL_ENDPOINT_REQUEST_CUSTOM_IMAGE,
                u,
            )
            update_model_endpoint(
                "model-endpoint-runnable-image-async",
                UPDATE_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE,
                u,
            )

        print("[Async] Waiting for updated model endpoints to build...")
        # Endpoint builds should be cached now.
        for u in users:
            ensure_n_ready_endpoints_short(3, u)

        if should_assert_task_success:
            time.sleep(5)

            for u in users:
                # Sending inference tasks to "model-endpoint-simple-async"
                for inference_payload, return_pickled in [
                    (INFERENCE_PAYLOAD_RETURN_PICKLED_TRUE, True),
                    (INFERENCE_PAYLOAD_RETURN_PICKLED_FALSE, False),
                ]:
                    print(
                        f"[Async] Sending async tasks to model-endpoint-simple-async for user {u}, {inference_payload=}, {return_pickled=} ..."
                    )
                    task_ids = asyncio.run(
                        create_async_tasks(
                            "model-endpoint-simple-async",
                            [inference_payload] * 3,
                            u,
                        )
                    )
                    print("[Async] Retrieving async task results...")
                    ensure_nonzero_available_workers("model-endpoint-simple-async", u)
                    ensure_all_async_tasks_success(task_ids, u, return_pickled)

                # Sending inference tasks to "model-endpoint-runnable-image-async"
                print(
                    f"[Async] Sending async tasks to model-endpoint-runnable-image-async for user {u} ..."
                )
                task_ids = asyncio.run(
                    create_async_tasks(
                        "model-endpoint-runnable-image-async",
                        [INFERENCE_PAYLOAD] * 3,
                        u,
                    )
                )
                print("[Async] Retrieving async task results...")
                ensure_nonzero_available_workers("model-endpoint-runnable-image-async", u)
                ensure_all_async_tasks_success(task_ids, u, return_pickled=False)
        else:
            print("[Async] not sending requests to endpoints")
    finally:
        print("[Async] Deleting model endpoints...")
        for u in users:
            delete_model_endpoint("model-endpoint-simple-async", u)
            delete_model_endpoint("model-endpoint-custom-image-async", u)
            delete_model_endpoint("model-endpoint-runnable-image-async", u)


def e2e_test_sync_endpoints(
    should_assert_task_success: bool,
    users: Sequence[str] = DEFAULT_USERS,
) -> None:
    # Assumes that delete_endpoints went first, and so endpoints are already deleted prior to entering.
    if len(users) == 0:
        raise ValueError("Must supply at least one user!")

    try:
        print("[Sync] Creating model bundles...")
        for u in users:
            get_or_create_model_bundle(CREATE_MODEL_BUNDLE_REQUEST_SIMPLE, u, "v1")
            get_or_create_model_bundle(CREATE_MODEL_BUNDLE_REQUEST_CUSTOM_IMAGE, u, "v1")
            get_or_create_model_bundle(CREATE_MODEL_BUNDLE_REQUEST_RUNNABLE_IMAGE, u, "v2")

        print("[Sync] Creating model endpoints...")
        for u in users:
            create_model_endpoint(CREATE_SYNC_MODEL_ENDPOINT_REQUEST_SIMPLE, u)
            create_model_endpoint(CREATE_SYNC_MODEL_ENDPOINT_REQUEST_CUSTOM_IMAGE, u)
            create_model_endpoint(CREATE_SYNC_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE, u)

        print("[Sync] Waiting for model endpoints to build...")
        # Endpoint builds should be cached now.
        for u in users:
            ensure_n_ready_endpoints_short(3, u)

        print("[Sync] Updating model endpoints...")
        for u in users:
            update_model_endpoint(
                "model-endpoint-simple-sync",
                UPDATE_MODEL_ENDPOINT_REQUEST_SIMPLE,
                u,
            )
            update_model_endpoint(
                "model-endpoint-custom-image-sync",
                UPDATE_MODEL_ENDPOINT_REQUEST_CUSTOM_IMAGE,
                u,
            )
            update_model_endpoint(
                "model-endpoint-runnable-image-sync",
                UPDATE_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE,
                u,
            )
        print("[Sync] Waiting for updated model endpoints to build...")
        # Endpoint builds should be cached now.
        for u in users:
            ensure_n_ready_endpoints_short(3, u)

        if should_assert_task_success:
            # For sync endpoints, need to wait for pods to spin up first before sending requests.
            print("[Sync] Waiting for pods to spin up...")
            for u in users:
                ensure_nonzero_available_workers("model-endpoint-simple-sync", u)
                ensure_nonzero_available_workers("model-endpoint-runnable-image-sync", u)

            time.sleep(5)

            for u in users:
                print("-" * 80)
                print(f"[Sync] user '{u}' responses:")
                # Sending inference tasks to "model-endpoint-simple-sync"
                for inference_payload, return_pickled in [
                    (INFERENCE_PAYLOAD_RETURN_PICKLED_TRUE, True),
                    (INFERENCE_PAYLOAD_RETURN_PICKLED_FALSE, False),
                ]:
                    print(
                        f"[Sync] Sending sync tasks to model-endpoint-simple-sync for user {u}, {inference_payload=}, {return_pickled=} ..."
                    )
                    task_responses = asyncio.run(
                        create_sync_tasks(
                            "model-endpoint-simple-sync",
                            [inference_payload] * 3,
                            u,
                        )
                    )
                    for response in task_responses:
                        ensure_inference_task_response_is_correct(response, return_pickled)

                # Sending inference tasks to "model-endpoint-runnable-image-sync"
                print("-" * 80)
                print(
                    f"[Sync] Sending sync tasks to model-endpoint-runnable-image-sync for user {u} ..."
                )
                task_responses = asyncio.run(
                    create_sync_tasks(
                        "model-endpoint-runnable-image-sync",
                        [INFERENCE_PAYLOAD] * 3,
                        u,
                    )
                )
                for response in task_responses:
                    ensure_inference_task_response_is_correct(response, return_pickled=False)

                print("-" * 40)
        else:
            print("[Sync] not sending requests to endpoints")
    finally:
        print("[Sync] Deleting model endpoints...")
        for u in users:
            delete_model_endpoint("model-endpoint-simple-sync", u)
            delete_model_endpoint("model-endpoint-custom-image-sync", u)
            delete_model_endpoint("model-endpoint-runnable-image-sync", u)


def e2e_test_streaming_endpoints(
    should_assert_task_success: bool,
    users: Sequence[str] = DEFAULT_USERS,
) -> None:
    # Assumes that delete_endpoints went first, and so endpoints are already deleted prior to entering.
    if len(users) == 0:
        raise ValueError("Must supply at least one user!")

    try:
        print("[Streaming] Creating model bundles...")
        for u in users:
            get_or_create_model_bundle(CREATE_MODEL_BUNDLE_REQUEST_STREAMING_IMAGE, u, "v2")

        print("[Streaming] Creating model endpoints...")
        for u in users:
            create_model_endpoint(CREATE_STREAMING_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE, u)
        print("[Streaming] Waiting for model endpoints to build...")
        # Endpoint builds should be cached now.
        for u in users:
            ensure_n_ready_endpoints_short(1, u)

        print("[Streaming] Updating model endpoints...")
        for u in users:
            update_model_endpoint(
                "model-endpoint-runnable-image-streaming",
                UPDATE_MODEL_ENDPOINT_REQUEST_STREAMING_IMAGE,
                u,
            )
        print("[Streaming] Waiting for updated model endpoints to build...")
        # Endpoint builds should be cached now.
        for u in users:
            ensure_n_ready_endpoints_short(1, u)

        if should_assert_task_success:
            # For streaming endpoints, need to wait for pods to spin up first before sending requests.
            print("[Streaming] Waiting for pods to spin up...")
            for u in users:
                ensure_nonzero_available_workers("model-endpoint-runnable-image-streaming", u)

            time.sleep(5)
            print("[Streaming] Sending streaming tasks...")
            create_streaming_task_requests = [INFERENCE_PAYLOAD] * 5
            print("-" * 80)

            for u in users:
                task_responses = []
                # TODO: add back in runnable image test once we can get it to work in Circle CI.
                for endpoint_name in [
                    "model-endpoint-runnable-image-streaming",
                ]:
                    task_responses.extend(
                        asyncio.run(
                            create_streaming_tasks(
                                endpoint_name,
                                create_streaming_task_requests,
                                u,
                            )
                        )
                    )
                print(f"[Streaming] user '{u}' responses ({len(task_responses)}):")
                for response in task_responses:
                    print(response)
                    print(type(response))
                    assert (
                        response.strip()
                        == 'data: {"status": "SUCCESS", "result": {"result": {"y": 1}}, "traceback": null}'
                    )
                print("-" * 40)

        else:
            print("[Streaming] not sending requests to endpoints")
    finally:
        print("[Streaming] Deleting model endpoints...")
        for u in users:
            delete_model_endpoint("model-endpoint-runnable-image-streaming", u)


def e2e_test_sync_streaming_endpoints(
    should_assert_task_success: bool,
    users: Sequence[str] = DEFAULT_USERS,
) -> None:
    # Assumes that delete_endpoints went first, and so endpoints are already deleted prior to entering.
    if len(users) == 0:
        raise ValueError("Must supply at least one user!")

    try:
        print("[Sync Streaming] Creating model bundles...")
        for u in users:
            get_or_create_model_bundle(CREATE_MODEL_BUNDLE_REQUEST_SYNC_STREAMING_IMAGE, u, "v2")

        print("[Sync Streaming] Creating model endpoints...")
        for u in users:
            create_model_endpoint(CREATE_SYNC_STREAMING_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE, u)
        print("[Sync Streaming] Waiting for model endpoints to build...")
        # Endpoint builds should be cached now.
        for u in users:
            ensure_n_ready_endpoints_short(1, u)

        if should_assert_task_success:
            # For streaming endpoints, need to wait for pods to spin up first before sending requests.
            print("[Sync Streaming] Waiting for pods to spin up...")
            for u in users:
                ensure_nonzero_available_workers(SYNC_STREAMING_MODEL_ENDPOINT_NAME, u)

            time.sleep(5)
            print("[Sync Streaming] Sending streaming tasks...")
            create_streaming_task_requests = [INFERENCE_PAYLOAD] * 5
            print("-" * 80)

            for u in users:
                task_responses = []
                # TODO: add back in runnable image test once we can get it to work in Circle CI.
                for endpoint_name in [
                    SYNC_STREAMING_MODEL_ENDPOINT_NAME,
                ]:
                    task_responses.extend(
                        asyncio.run(
                            create_streaming_tasks(
                                endpoint_name,
                                create_streaming_task_requests,
                                u,
                            )
                        )
                    )
                print(f"[Sync Streaming] user '{u}' responses ({len(task_responses)}):")
                for response in task_responses:
                    print(response)
                    print(type(response))
                    assert (
                        response.strip()
                        == 'data: {"status": "SUCCESS", "result": {"result": {"y": 1}}, "traceback": null}'
                    )
                print("-" * 40)

            print("[Sync Streaming] Sending sync tasks...")
            print("-" * 80)

            for u in users:
                task_responses = []
                # TODO: add back in runnable image test once we can get it to work in Circle CI.
                for endpoint_name in [
                    SYNC_STREAMING_MODEL_ENDPOINT_NAME,
                ]:
                    task_responses.extend(
                        asyncio.run(
                            create_sync_tasks(
                                endpoint_name,
                                create_streaming_task_requests,
                                u,
                            )
                        )
                    )

                print("-" * 80)
                for response in task_responses:
                    ensure_inference_task_response_is_correct(response, return_pickled=False)

                print("-" * 40)
        else:
            print("[Sync Streaming] not sending requests to endpoints")
    finally:
        print("[Sync Streaming] Deleting model endpoints...")
        for u in users:
            delete_model_endpoint(SYNC_STREAMING_MODEL_ENDPOINT_NAME, u)


async def write_lira_feature_flag(value: bool):
    redis = aioredis.from_url(REDIS_URI)
    await redis.set(
        f"launch-feature-flag:{FEATURE_FLAG_USE_MULTI_CONTAINER_ARCHITECTURE_FOR_ARTIFACTLIKE_BUNDLE}",
        str(value),
    )
    await redis.close()


def e2e_test(
    should_assert_task_success: bool,
    users: Sequence[str] = DEFAULT_USERS,
) -> None:
    print("Checking if gateway is ready for tests...")
    ensure_gateway_ready()
    delete_existing_endpoints(users=users)

    asyncio.run(write_lira_feature_flag(True))
    e2e_test_async_endpoints(should_assert_task_success, users=users)
    e2e_test_sync_endpoints(should_assert_task_success, users=users)

    asyncio.run(write_lira_feature_flag(False))
    e2e_test_async_endpoints(should_assert_task_success, users=users)
    e2e_test_sync_endpoints(should_assert_task_success, users=users)

    e2e_test_streaming_endpoints(should_assert_task_success, users=users)
    e2e_test_sync_streaming_endpoints(should_assert_task_success, users=users)
    e2e_test_batch_jobs(users=users)
    print("All tests passed!")


def entrypoint() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-async-tasks",
        "-c",
        action="store_true",
        help="If set, asserts that the async tasks return SUCCESS as their status.",
    )
    args, unknown = parser.parse_known_args()
    e2e_test(
        args.check_async_tasks,
        users=DEFAULT_USERS,
    )


if __name__ == "__main__":
    entrypoint()
