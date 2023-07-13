import argparse
from typing import Any, Dict

import requests
from tenacity import retry, stop_after_attempt, wait_fixed

from spellbook_serve.domain.entities import ModelEndpointType

DEFAULT_NETWORK_TIMEOUT_SEC = 10

SERVICE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "stablelm-tuned-7b-fast": {
        "bundle_configs": {
            "git_commit": "59121612a308693277863383fc4fb7533b98d4dd",
        },
        "endpoint_configs": {
            "min_workers": 0,
            "max_workers": 3,
            "per_worker": 10,
            "memory": "180Gi",
            "storage": "85Gi",
            "cpus": 32,
            "gpus": 4,
            "gpu_type": "nvidia-ampere-a10",
        },
    },
    "stablelm-tuned-7b-slow": {
        "bundle_configs": {
            "git_commit": "59121612a308693277863383fc4fb7533b98d4dd",
        },
        "endpoint_configs": {
            "min_workers": 0,
            "max_workers": 3,
            "per_worker": 10,
            "memory": "180Gi",
            "storage": "85Gi",
            "cpus": 32,
            "gpus": 4,
            "gpu_type": "nvidia-ampere-a10",
        },
    },
    "flan-t5-xxl-fast": {
        "bundle_configs": {
            "git_commit": "59121612a308693277863383fc4fb7533b98d4dd",
        },
        "endpoint_configs": {
            "min_workers": 0,
            "max_workers": 3,
            "per_worker": 10,
            "memory": "180Gi",
            "storage": "85Gi",
            "cpus": 32,
            "gpus": 4,
            "gpu_type": "nvidia-ampere-a10",
        },
    },
    "flan-t5-xxl-slow": {
        "bundle_configs": {
            "git_commit": "59121612a308693277863383fc4fb7533b98d4dd",
        },
        "endpoint_configs": {
            "min_workers": 0,
            "max_workers": 3,
            "per_worker": 10,
            "memory": "180Gi",
            "storage": "85Gi",
            "cpus": 32,
            "gpus": 4,
            "gpu_type": "nvidia-ampere-a10",
        },
    },
    "llama-7b": {
        "bundle_configs": {
            "git_commit": "59121612a308693277863383fc4fb7533b98d4dd",
        },
        "endpoint_configs": {
            "min_workers": 0,
            "max_workers": 5,
            "per_worker": 10,
            "memory": "180Gi",
            "storage": "85Gi",
            "cpus": 32,
            "gpus": 4,
            "gpu_type": "nvidia-ampere-a10",
        },
    },
    "gpt-j-6b": {
        "bundle_configs": {
            "git_commit": "59121612a308693277863383fc4fb7533b98d4dd",
        },
        "endpoint_configs": {
            "min_workers": 0,
            "max_workers": 5,
            "per_worker": 10,
            "memory": "180Gi",
            "storage": "85Gi",
            "cpus": 32,
            "gpus": 4,
            "gpu_type": "nvidia-ampere-a10",
        },
    },
}


@retry(stop=stop_after_attempt(30), wait=wait_fixed(10))
def ensure_gateway_ready(gateway_url: str):
    response = requests.get(f"{gateway_url}/healthz")
    assert response.ok


def spellbook_bundle_payload(
    *,
    git_commit: str,
    model_name: str,
    concurrency: int = 1,
    initial_delay: int = 1800,
):
    return {
        "name": f"spellbook-deepspeed-{model_name}",
        "schema_location": "unused",
        "flavor": {
            "flavor": "runnable_image",
            "repository": "spellbook-serve",
            "tag": f"spellbook_serve_llm_cuda_image_{git_commit}",
            "command": [
                "dumb-init",
                "--",
                "ddtrace-run",
                "run-service",
                "--http",
                "production_threads",
                "--concurrency",
                str(concurrency),
                "--config",
                "/install/spellbook/inference/service--spellbook_inference.yaml",
            ],
            "protocol": "http",
            "env": {
                "MODEL_NAME": model_name,
                "ML_INFRA_SERVICES_CONFIG_PATH": "/infra-config/config.yaml",
            },
            "readiness_initial_delay_seconds": initial_delay,
        },
    }


def spellbook_endpoint_payload(
    *,
    endpoint_name: str,
    bundle_name: str,
    endpoint_type: ModelEndpointType = ModelEndpointType.SYNC,
    min_workers: int = 0,
    max_workers: int = 1,
    memory: str = "185Gi",
    storage: str = "10Gi",
    cpus: int = 4,
    gpus: int = 44,
    per_worker: int = 1,
    gpu_type: str = "nvidia-ampere-a10",
):
    return {
        "name": endpoint_name,
        "bundle_name": bundle_name,
        "min_workers": min_workers,
        "max_workers": max_workers,
        "per_worker": per_worker,
        "endpoint_type": endpoint_type,
        "metadata": {},
        "cpus": cpus,
        "storage": storage,
        "memory": memory,
        "gpus": gpus,
        "gpu_type": gpu_type,
        "labels": {
            "team": "spellbook",
            "product": "spellbook",
        },
    }


def create_model_bundle(
    gateway_url: str, create_model_bundle_request: Dict[str, Any], user_id: str
) -> Dict[str, Any]:
    response = requests.post(
        f"{gateway_url}/v2/model-bundles",
        json=create_model_bundle_request,
        headers={"Content-Type": "application/json"},
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def get_latest_model_bundle(gateway_url: str, model_name: str, user_id: str) -> Dict[str, Any]:
    response = requests.get(
        f"{gateway_url}/v2/model-bundles/latest?model_name={model_name}",
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


def replace_model_bundle_name_with_id(gateway_url, request: Dict[str, Any], user_id: str):
    if "bundle_name" in request:
        model_bundle = get_latest_model_bundle(gateway_url, request["bundle_name"], user_id)
        request["model_bundle_id"] = model_bundle["id"]
        del request["bundle_name"]


def create_model_endpoint(
    gateway_url: str, create_model_endpoint_request: Dict[str, Any], user_id: str
) -> Dict[str, Any]:
    create_model_endpoint_request = create_model_endpoint_request.copy()
    replace_model_bundle_name_with_id(gateway_url, create_model_endpoint_request, user_id)
    response = requests.post(
        f"{gateway_url}/v1/model-endpoints",
        json=create_model_endpoint_request,
        headers={"Content-Type": "application/json"},
        auth=(user_id, ""),
        timeout=DEFAULT_NETWORK_TIMEOUT_SEC,
    )
    if not response.ok:
        raise ValueError(response.content)
    return response.json()


def create_spellbook_deployments(gateway_url: str):
    for model_name, service_config in SERVICE_CONFIGS.items():
        bundle_payload = spellbook_bundle_payload(
            model_name=model_name,
            **service_config["bundle_configs"],
        )
        endpoint_payload = spellbook_endpoint_payload(
            endpoint_name=f"endpoint-{bundle_payload['name']}",
            bundle_name=bundle_payload["name"],
            **service_config["endpoint_configs"],
        )

        print(f"Creating bundle {bundle_payload['name']}")
        create_model_bundle(gateway_url, bundle_payload, "scale")
        print(f"Creating endpoint {endpoint_payload['name']}")
        create_model_endpoint(gateway_url, endpoint_payload, "scale")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gateway-url", type=str, required=True)
    args = parser.parse_args()

    ensure_gateway_ready(args.gateway_url)
    create_spellbook_deployments(args.gateway_url)
