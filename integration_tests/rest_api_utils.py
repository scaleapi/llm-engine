import asyncio
import inspect
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence

import aiohttp
import requests
from model_engine_server.common.dtos.tasks import TaskStatus
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

_DEFAULT_BASE_PATH = "http://localhost:5001"
BASE_PATH = os.environ.get("BASE_PATH", _DEFAULT_BASE_PATH)
print(f"Integration tests using gateway {BASE_PATH=}")
DEFAULT_NETWORK_TIMEOUT_SEC = 10
LONG_NETWORK_TIMEOUT_SEC = 30

# add suffix to avoid name collisions
SERVICE_IDENTIFIER = os.environ.get("SERVICE_IDENTIFIER", "")


def format_name(name: str) -> str:
    return f"{name}-{SERVICE_IDENTIFIER}" if SERVICE_IDENTIFIER else name


# Use the scale-launch-integration-tests id
USER_ID_0 = os.getenv("TEST_USER_ID", "fakeuser")

DEFAULT_USERS: Sequence[str] = (USER_ID_0,)  # type: ignore


def echo_load_predict_fn(model):
    def echo(**keyword_args):
        return model(**keyword_args)

    return echo


def echo_load_model_fn():
    def my_model(**keyword_args):
        return {k: v for k, v in keyword_args.items()}

    return my_model


CREATE_MODEL_BUNDLE_REQUEST_SIMPLE = {
    "name": "model_bundle_simple",
    "schema_location": "s3://model-engine-integration-tests/model_bundles/echo_schemas",
    "metadata": {
        "test_key": "test_value",
    },
    "flavor": {
        "flavor": "cloudpickle_artifact",
        "load_predict_fn": inspect.getsource(echo_load_predict_fn),
        "load_model_fn": inspect.getsource(echo_load_model_fn),
        "framework": {
            "framework_type": "pytorch",
            "pytorch_image_tag": "1.7.1-cuda11.0-cudnn8-runtime",
        },
        "requirements": [
            "cloudpickle==2.1.0",
            "pyyaml==6.0",
            "pydantic==2.8.2",
            "fastapi==0.110.0",
        ],
        "location": "s3://model-engine-integration-tests/model_bundles/echo_bundle",
    },
}

CREATE_MODEL_BUNDLE_REQUEST_RUNNABLE_IMAGE = {
    "name": "model_bundle_runnable_image",
    "schema_location": "s3://model-engine-integration-tests/model_bundles/echo_schemas",
    "metadata": {
        "test_key": "test_value",
    },
    "flavor": {
        "flavor": "streaming_enhanced_runnable_image",
        "repository": "model-engine",
        "tag": "2c1951dfff7159d7d29dd13b4f888e8355f8d51e",
        "command": [
            "dumb-init",
            "--",
            "ddtrace-run",
            "python",
            "-m",
            "model_engine_server.inference.forwarding.echo_server",
            "--port",
            "5005",
        ],
        "streaming_command": [
            "dumb-init",
            "--",
            "ddtrace-run",
            "python",
            "-m",
            "model_engine_server.inference.forwarding.echo_server",
            "--port",
            "5005",
        ],
        "env": {
            "TEST_KEY": "test_value",
            "ML_INFRA_SERVICES_CONFIG_PATH": "/workspace/model-engine/model_engine_server/core/configs/default.yaml",
            # infra configs are mounted here
            "HTTP_HOST": "0.0.0.0",  # Hack for uvicorn to work in minikube
        },
        "protocol": "http",
        "readiness_initial_delay_seconds": 20,
    },
}

CREATE_ASYNC_MODEL_ENDPOINT_REQUEST_SIMPLE = {
    "bundle_name": "model_bundle_simple",
    "name": format_name("model-endpoint-simple-async"),
    "endpoint_type": "async",
    "cpus": "0.5",
    "memory": "500Mi",
    "storage": "1Gi",
    "min_workers": 1,
    "max_workers": 1,
    "gpus": 0,
    "per_worker": 1,
    "labels": {"team": "infra", "product": "launch"},
    "metadata": {},
}

CREATE_SYNC_MODEL_ENDPOINT_REQUEST_SIMPLE = CREATE_ASYNC_MODEL_ENDPOINT_REQUEST_SIMPLE.copy()
CREATE_SYNC_MODEL_ENDPOINT_REQUEST_SIMPLE["name"] = format_name("model-endpoint-simple-sync")
CREATE_SYNC_MODEL_ENDPOINT_REQUEST_SIMPLE["endpoint_type"] = "sync"

CREATE_ASYNC_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE = {
    "bundle_name": "model_bundle_runnable_image",
    "name": format_name("model-endpoint-runnable-async"),
    "post_inference_hooks": [],
    "endpoint_type": "async",
    "cpus": "1",
    "gpus": 0,
    "memory": "1Gi",
    "storage": "2Gi",
    "optimize_costs": False,
    "min_workers": 1,
    "max_workers": 1,
    "per_worker": 1,
    "labels": {"team": "infra", "product": "launch"},
    "metadata": {"key": "value"},
}

CREATE_SYNC_STREAMING_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE = (
    CREATE_ASYNC_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE.copy()
)
CREATE_SYNC_STREAMING_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE["name"] = format_name(
    "model-endpoint-runnable-sync-streaming"
)
CREATE_SYNC_STREAMING_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE["endpoint_type"] = "streaming"

UPDATE_MODEL_ENDPOINT_REQUEST_SIMPLE = {
    "bundle_name": "model_bundle_simple",
    "cpus": "1",
    "memory": "1Gi",
    "max_workers": 2,
}

UPDATE_MODEL_ENDPOINT_REQUEST_RUNNABLE_IMAGE = {
    "bundle_name": "model_bundle_runnable_image",
    "cpus": "2",
    "memory": "2Gi",
    "max_workers": 2,
}

INFERENCE_PAYLOAD: Dict[str, Any] = {
    "args": {"y": 1},
    "url": None,
}

CREATE_LLM_MODEL_ENDPOINT_REQUEST: Dict[str, Any] = {
    "name": format_name("llama-2-7b-test"),
    "model_name": "llama-2-7b-chat",
    "source": "hugging_face",
    "inference_framework": "vllm",
    "inference_framework_image_tag": "latest",
    "endpoint_type": "streaming",
    "cpus": 20,
    "gpus": 1,
    "memory": "20Gi",
    "gpu_type": "nvidia-hopper-h100-1g20gb",
    "storage": "40Gi",
    "optimize_costs": False,
    "min_workers": 1,
    "max_workers": 1,
    "per_worker": 1,
    "labels": {"team": "infra", "product": "launch"},
    "metadata": {"key": "value"},
    "public_inference": False,
}


INFERENCE_PAYLOAD_RETURN_PICKLED_FALSE: Dict[str, Any] = INFERENCE_PAYLOAD.copy()
INFERENCE_PAYLOAD_RETURN_PICKLED_FALSE["return_pickled"] = False

INFERENCE_PAYLOAD_RETURN_PICKLED_TRUE: Dict[str, Any] = INFERENCE_PAYLOAD.copy()
INFERENCE_PAYLOAD_RETURN_PICKLED_TRUE["return_pickled"] = True

LLM_PAYLOAD: Dict[str, Any] = {
    "prompt": "Hello, my name is",
    "max_new_tokens": 10,
    "temperature": 0.2,
}

LLM_PAYLOAD_WITH_STOP_SEQUENCE: Dict[str, Any] = LLM_PAYLOAD.copy()
LLM_PAYLOAD_WITH_STOP_SEQUENCE["stop_sequences"] = ["\n"]

LLM_PAYLOAD_WITH_PRESENCE_PENALTY: Dict[str, Any] = LLM_PAYLOAD.copy()
LLM_PAYLOAD_WITH_PRESENCE_PENALTY["presence_penalty"] = 0.5

LLM_PAYLOAD_WITH_FREQUENCY_PENALTY: Dict[str, Any] = LLM_PAYLOAD.copy()
LLM_PAYLOAD_WITH_FREQUENCY_PENALTY["frequency_penalty"] = 0.5

LLM_PAYLOAD_WITH_TOP_K: Dict[str, Any] = LLM_PAYLOAD.copy()
LLM_PAYLOAD_WITH_TOP_K["top_k"] = 10

LLM_PAYLOAD_WITH_TOP_P: Dict[str, Any] = LLM_PAYLOAD.copy()
LLM_PAYLOAD_WITH_TOP_P["top_p"] = 0.5

LLM_PAYLOAD_WITH_INCLUDE_STOP_STR_IN_OUTPUT: Dict[str, Any] = LLM_PAYLOAD.copy()
LLM_PAYLOAD_WITH_INCLUDE_STOP_STR_IN_OUTPUT["include_stop_str_in_output"] = True

LLM_PAYLOAD_WITH_GUIDED_JSON: Dict[str, Any] = LLM_PAYLOAD.copy()
LLM_PAYLOAD_WITH_GUIDED_JSON["guided_json"] = {
    "properties": {"myString": {"type": "string"}},
    "required": ["myString"],
}

LLM_PAYLOAD_WITH_GUIDED_REGEX: Dict[str, Any] = LLM_PAYLOAD.copy()
LLM_PAYLOAD_WITH_GUIDED_REGEX["guided_regex"] = "Sean.*"

LLM_PAYLOAD_WITH_GUIDED_CHOICE: Dict[str, Any] = LLM_PAYLOAD.copy()
LLM_PAYLOAD_WITH_GUIDED_CHOICE["guided_choice"] = ["dog", "cat"]

LLM_PAYLOAD_WITH_GUIDED_GRAMMAR: Dict[str, Any] = LLM_PAYLOAD.copy()
LLM_PAYLOAD_WITH_GUIDED_GRAMMAR["guided_grammar"] = 'start: "John"'

LLM_PAYLOADS_WITH_EXPECTED_RESPONSES = [
    (LLM_PAYLOAD, None, None),
    (LLM_PAYLOAD_WITH_STOP_SEQUENCE, None, None),
    (LLM_PAYLOAD_WITH_PRESENCE_PENALTY, None, None),
    (LLM_PAYLOAD_WITH_FREQUENCY_PENALTY, None, None),
    (LLM_PAYLOAD_WITH_TOP_K, None, None),
    (LLM_PAYLOAD_WITH_TOP_P, None, None),
    (LLM_PAYLOAD_WITH_INCLUDE_STOP_STR_IN_OUTPUT, ["tokens"], None),
    (LLM_PAYLOAD_WITH_GUIDED_JSON, None, None),
    (LLM_PAYLOAD_WITH_GUIDED_REGEX, None, "Sean.*"),
    (LLM_PAYLOAD_WITH_GUIDED_CHOICE, None, "dog|cat"),
    (LLM_PAYLOAD_WITH_GUIDED_GRAMMAR, None, "John"),
]

CREATE_BATCH_JOB_REQUEST: Dict[str, Any] = {
    "bundle_name": "model_bundle_simple",
    "input_path": "TBA",
    "serialization_format": "JSON",
    "labels": {"team": "infra", "product": "launch"},
    "resource_requests": {
        "memory": "500Mi",
        "max_workers": 1,
        "gpus": 0,
    },
}

CREATE_DOCKER_IMAGE_BATCH_JOB_BUNDLE_REQUEST: Dict[str, Any] = {
    "name": format_name("di_batch_job_bundle_1"),
    "image_repository": "model-engine",
    "image_tag": "2c1951dfff7159d7d29dd13b4f888e8355f8d51e",
    "command": ["jq", ".", "/launch_mount_location/file"],
    "env": {"ENV1": "VAL1"},
    "mount_location": "/launch_mount_location/file",
    "resource_requests": {
        "cpus": 0.1,
        "memory": "10Mi",
    },
}

CREATE_DOCKER_IMAGE_BATCH_JOB_REQUEST: Dict[str, Any] = {
    "docker_image_batch_job_bundle_name": format_name("di_batch_job_bundle_1"),
    "job_config": {"data": {"to": "mount"}},
    "labels": {"team": "infra", "product": "testing"},
    "resource_requests": {"cpus": 0.15, "memory": "15Mi"},
}

CREATE_FINE_TUNE_DI_BATCH_JOB_BUNDLE_REQUEST: Dict[str, Any] = {
    "name": format_name("fine_tune_di_batch_job_bundle_1"),
    "image_repository": "model-engine",
    "image_tag": "2c1951dfff7159d7d29dd13b4f888e8355f8d51e",
    "command": ["cat", "/launch_mount_location/file"],
    "env": {"ENV1": "VAL1"},
    "mount_location": "/launch_mount_location/file",
    "resource_requests": {
        "cpus": 0.1,
        "memory": "10Mi",
    },
    "public": True,
}

CREATE_FINE_TUNE_REQUEST: Dict[str, Any] = {
    "model": "test_base_model",
    "training_file": "s3://model-engine-integration-tests/fine_tune_files/run_through_walls.csv",
    "validation_file": None,
    # "fine_tuning_method": "test_fine_tuning_method",  # ignored until we change it
    "hyperparameters": {},
}


@retry(stop=stop_after_attempt(300), wait=wait_fixed(2))
def ensure_launch_gateway_healthy():
    assert requests.get(f"{BASE_PATH}/healthz").status_code == 200


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
    except:  # noqa: E722
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


def cancel_batch_job(batch_job_id: str, user_id: str) -> Dict[str, Any]:
    response = requests.put(
        f"{BASE_PATH}/v1/batch-jobs/{batch_job_id}",
        json={"cancel": True},
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
    except:  # noqa: E722
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


def create_fine_tune(create_fine_tune_request: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    response = requests.post(
        f"{BASE_PATH}/v1/llm/fine-tunes",
        json=create_fine_tune_request,
        headers={"Content-Type": "application/json"},
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


def get_fine_tune_by_id(fine_tune_id: str, user_id: str) -> Dict[str, Any]:
    response = requests.get(
        f"{BASE_PATH}/v1/llm/fine-tunes/{fine_tune_id}",
        headers={"Content-Type": "application/json"},
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


def list_fine_tunes(user_id: str) -> Dict[str, Any]:
    response = requests.get(
        f"{BASE_PATH}/v1/llm/fine-tunes",
        headers={"Content-Type": "application/json"},
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


def cancel_fine_tune_by_id(fine_tune_id: str, user_id: str) -> Dict[str, Any]:
    response = requests.put(
        f"{BASE_PATH}/v1/llm/fine-tunes/{fine_tune_id}/cancel",
        headers={"Content-Type": "application/json"},
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


def upload_file(file, user_id: str) -> Dict[str, Any]:
    files = {"file": file}
    response = requests.post(
        f"{BASE_PATH}/v1/files",
        files=files,
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


def get_file_by_id(file_id: str, user_id: str) -> Dict[str, Any]:
    response = requests.get(
        f"{BASE_PATH}/v1/files/{file_id}",
        headers={"Content-Type": "application/json"},
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


def list_files(user_id: str) -> Dict[str, Any]:
    response = requests.get(
        f"{BASE_PATH}/v1/files",
        headers={"Content-Type": "application/json"},
        auth=(user_id, ""),
        timeout=30,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


def delete_file_by_id(file_id: str, user_id: str) -> Dict[str, Any]:
    response = requests.delete(
        f"{BASE_PATH}/v1/files/{file_id}",
        headers={"Content-Type": "application/json"},
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


def get_file_content_by_id(file_id: str, user_id: str) -> Dict[str, Any]:
    response = requests.get(
        f"{BASE_PATH}/v1/files/{file_id}/content",
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


@retry(stop=stop_after_attempt(6), wait=wait_fixed(1))
def get_llm_model_endpoint(name: str, user_id: str) -> Dict[str, Any]:
    response = requests.get(
        f"{BASE_PATH}/v1/llm/model-endpoints/{name}",
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


@retry(stop=stop_after_attempt(3), wait=wait_fixed(20))
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


def delete_llm_model_endpoint(endpoint_name: str, user_id: str) -> Dict[str, Any]:
    response = requests.delete(
        f"{BASE_PATH}/v1/llm/model-endpoints/{endpoint_name}",
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


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def list_llm_model_endpoints(user_id: str) -> List[Dict[str, Any]]:
    response = requests.get(
        f"{BASE_PATH}/v1/llm/model-endpoints",
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()["model_endpoints"]


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def create_llm_model_endpoint(
    create_llm_model_endpoint_request: Dict[str, Any],
    user_id: str,
    inference_framework: Optional[str],
    inference_framework_image_tag: Optional[str],
) -> Dict[str, Any]:
    create_model_endpoint_request = create_llm_model_endpoint_request.copy()
    if inference_framework:
        create_model_endpoint_request["inference_framework"] = inference_framework
    if inference_framework_image_tag:
        create_model_endpoint_request["inference_framework_image_tag"] = (
            inference_framework_image_tag
        )
    response = requests.post(
        f"{BASE_PATH}/v1/llm/model-endpoints",
        json=create_model_endpoint_request,
        headers={"Content-Type": "application/json"},
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


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


async def create_llm_sync_task(
    model_endpoint_name: str,
    create_sync_task_request: Dict[str, Any],
    user_id: str,
    session: aiohttp.ClientSession,
) -> str:
    async with session.post(
        f"{BASE_PATH}/v1/llm/completions-sync?model_endpoint_name={model_endpoint_name}",
        json=create_sync_task_request,
        headers={"Content-Type": "application/json"},
        auth=aiohttp.BasicAuth(user_id, ""),
        timeout=LONG_NETWORK_TIMEOUT_SEC,
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


async def create_llm_streaming_task(
    model_endpoint_name: str,
    create_streaming_task_request: Dict[str, Any],
    user_id: str,
    session: aiohttp.ClientSession,
) -> str:
    async with session.post(
        f"{BASE_PATH}/v1/llm/completions-stream?model_endpoint_name={model_endpoint_name}",
        json=create_streaming_task_request,
        headers={"Content-Type": "application/json"},
        auth=aiohttp.BasicAuth(user_id, ""),
        timeout=LONG_NETWORK_TIMEOUT_SEC,
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


async def create_llm_sync_tasks(
    endpoint_name: str, create_sync_task_requests: List[Dict[str, Any]], user_id: str
) -> List[Any]:
    async with aiohttp.ClientSession() as session:
        tasks = []
        for create_sync_task_request in create_sync_task_requests:
            task = create_llm_sync_task(endpoint_name, create_sync_task_request, user_id, session)
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


async def create_llm_streaming_tasks(
    endpoint_name: str, create_streaming_task_requests: List[Dict[str, Any]], user_id: str
) -> List[Any]:
    async with aiohttp.ClientSession() as session:
        tasks = []
        for create_streaming_task_request in create_streaming_task_requests:
            task = create_llm_streaming_task(
                endpoint_name, create_streaming_task_request, user_id, session
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


# Wait 2 minutes (120 seconds) for endpoints to build.
@retry(stop=stop_after_attempt(12), wait=wait_fixed(10))
def ensure_n_ready_private_llm_endpoints_short(n: int, user_id: str):
    endpoints = list_llm_model_endpoints(user_id)
    private_endpoints = [
        endpoint for endpoint in endpoints if not endpoint["spec"]["public_inference"]
    ]
    ready_endpoints = [endpoint for endpoint in private_endpoints if endpoint["status"] == "READY"]
    print(
        f"User {user_id} Current num endpoints: {len(private_endpoints)}, num ready endpoints: {len(ready_endpoints)}"
    )
    assert (
        len(ready_endpoints) >= n
    ), f"Expected {n} ready endpoints, got {len(ready_endpoints)}. Look through endpoint builder for errors."


def delete_all_endpoints(user_id: str, delete_suffix_only: bool):
    endpoints = list_model_endpoints(user_id)
    for i, endpoint in enumerate(endpoints):
        if (
            delete_suffix_only
            and SERVICE_IDENTIFIER
            and not endpoint["name"].endswith(SERVICE_IDENTIFIER)
        ):
            continue

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


# Wait up to 20 minutes (1200 seconds) for the pods to spin up.
@retry(stop=stop_after_attempt(120), wait=wait_fixed(10))
def ensure_nonzero_available_llm_workers(endpoint_name: str, user_id: str):
    simple_endpoint = get_llm_model_endpoint(endpoint_name, user_id)
    assert simple_endpoint["spec"].get("deployment_state", {}).get("available_workers", 0)


def ensure_inference_task_response_is_correct(response: Dict[str, Any], return_pickled: bool):
    print(response)
    assert response["status"] == "SUCCESS"
    assert response["traceback"] is None
    if return_pickled:
        assert response["result"]["result_url"].startswith("s3://")
    else:
        assert response["result"] == {"result": '{"y": 1}'}


def ensure_llm_task_response_is_correct(
    response: Dict[str, Any],
    required_output_fields: Optional[List[str]],
    response_text_regex: Optional[str],
):
    print(response)
    assert response["output"] is not None

    if required_output_fields is not None:
        for field in required_output_fields:
            assert field in response["output"]

    if response_text_regex is not None:
        assert re.search(response_text_regex, response["output"]["text"])


def ensure_llm_task_stream_response_is_correct(
    response: str,
    required_output_fields: Optional[List[str]],
    response_text_regex: Optional[str],
):
    # parse response
    #  data has format "data: <data>\n\ndata: <data>\n\n"
    #  We want to get a list of dictionaries parsing out the 'data:' field
    parsed_response = [
        json.loads(r.split("data: ")[1]) for r in response.split("\n") if "data:" in r.strip()
    ]

    # Join the text field of the response
    response_text = "".join([r["output"]["text"] for r in parsed_response])
    print("response text: ", response_text)
    assert response_text is not None

    if response_text_regex is not None:
        assert re.search(response_text_regex, response_text)


# Wait up to 30 seconds for the tasks to be returned.
@retry(
    stop=stop_after_attempt(10), wait=wait_fixed(1), retry=retry_if_exception_type(AssertionError)
)
def ensure_all_async_tasks_success(task_ids: List[str], user_id: str, return_pickled: bool):
    responses = asyncio.run(get_async_tasks(task_ids, user_id))
    for response in responses:
        if response["status"] not in (TaskStatus.PENDING, TaskStatus.SUCCESS, TaskStatus.STARTED):
            print(response)
            raise ValueError("Task failed!")
        ensure_inference_task_response_is_correct(response, return_pickled)


def delete_existing_endpoints(
    users: Sequence[str] = DEFAULT_USERS, delete_suffix_only: bool = True
) -> None:
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

    if all([len(info) == 0 for info in all_endpoint_info]):
        return

    # delete the endpoints: if this fails, manually remove the dangling k8s deployments
    # and delete the user's endpoints from the hosted_model_inference.endpoints table
    # i.e. by default this is running the following SQL:
    #
    # >>>> delete from model_engine_server.endpoints where created_by in (
    # >>>>       'test00000000000000000000',
    # >>>>       'test11111111111111111111',
    # >>>> )
    #
    time.sleep(15)  # need to sleep to allow the cache to refresh
    print(f"[{len({users})}] Deleting all user endpoints...")
    try:
        for i, u in enumerate(users):
            suffix_msg = f" with suffix {SERVICE_IDENTIFIER}" if delete_suffix_only else ""
            print(f"[{i + 1}/{len(users)}] Deleting all endpoints{suffix_msg} for user with ID {u}")
            delete_all_endpoints(u, delete_suffix_only)
    except Exception:  # noqa
        try:
            j: str = json.dumps(all_endpoint_info, indent=2)
        except Exception as j_error:  # noqa
            j = f"[FAILED TO JSON ENCODE {j_error}]\n{all_endpoint_info}"
        barrier: str = "-" * 80
        print(f"ERROR! Deletion failed. All endpoint information:\n{barrier}\n{j}\n{barrier}")
        raise
    time.sleep(15)
