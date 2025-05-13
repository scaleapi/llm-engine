# This file contains standard settings for ML serve.
#

import hashlib
from typing import List, Tuple

from model_engine_server.common.config import hmi_config
from model_engine_server.core.config import infra_config

DEPLOYMENT_PREFIX = "launch"
LEGACY_DEPLOYMENT_PREFIX = "hmi"
SERVICE_BUILDER_QUEUE_PREFIX = "model-engine"
SERVICE_BUILDER_QUEUE_SUFFIX = "service-builder"
HOSTED_INFERENCE_SERVER_NAME = "hostedinference"
LAUNCH_SERVER_NAME = "launch"
K8S_CACHER_NAME = "launch-k8s-cacher"
PYSPARK_DEFAULT_ENDPOINT_PARAMS = dict(
    cpus=3,
    memory="12Gi",
    gpus=1,
    gpu_type="nvidia-tesla-t4",
    min_workers=0,
    max_workers=50,
    per_worker=40,
)  # TODO: we could probably determine an appropriate value for max_workers based on the size of the batch
PYSPARK_DEFAULT_MAX_EXECUTORS = 50
PYSPARK_DEFAULT_PARTITION_SIZE = 500

RESTRICTED_ENDPOINT_LABELS = set(
    [
        "user_id",
        "endpoint_name",
    ]
)

REQUIRED_ENDPOINT_LABELS = set(
    [
        "team",
        "product",
    ]
)

PRETRAINED_ENDPOINTS_CREATED_BY = ["nucleus-model-zoo", "bloom", "llm", "pretrained"]


def generate_deployment_name(user_id, endpoint_name):
    return "-".join(_generate_deployment_name_parts(user_id, endpoint_name))


def _generate_queue_name(user_id, endpoint_name):
    return ".".join(_generate_deployment_name_parts(user_id, endpoint_name))


def generate_destination(user_id: str, endpoint_name: str, endpoint_type: str) -> str:
    if endpoint_type == "async":
        return _generate_queue_name(user_id, endpoint_name)
    elif endpoint_type in {"sync", "streaming"}:
        return generate_deployment_name(user_id, endpoint_name)
    else:
        raise ValueError(f"Invalid endpoint_type: {endpoint_type}")


def _generate_deployment_name_parts(user_id: str, endpoint_name: str) -> List[str]:
    user_endpoint_hash = hashlib.md5((user_id + endpoint_name).encode("utf-8")).hexdigest()
    return [
        DEPLOYMENT_PREFIX,
        user_id[:24],
        endpoint_name[:8],
        user_endpoint_hash[:8],
    ]


def generate_batch_job_name(user_id: str, endpoint_name: str):
    batch_job_partial_name = "-".join(_generate_deployment_name_parts(user_id, endpoint_name))
    return f"batch-job-{batch_job_partial_name}"


def get_sync_endpoint_hostname_and_url(deployment_name: str) -> Tuple[str, str]:
    hostname = f"{deployment_name}.{hmi_config.endpoint_namespace}"
    return hostname, f"http://{hostname}/predict"


def get_sync_endpoint_elb_url(deployment_name: str) -> str:
    return f"http://{deployment_name}.{infra_config().dns_host_domain}/predict"


def get_service_builder_queue(service_identifier=None, service_builder_queue_name=None):
    if service_builder_queue_name:
        return service_builder_queue_name
    elif service_identifier:
        return f"{SERVICE_BUILDER_QUEUE_PREFIX}-{service_identifier}-{SERVICE_BUILDER_QUEUE_SUFFIX}"
    else:
        return f"{SERVICE_BUILDER_QUEUE_PREFIX}-{SERVICE_BUILDER_QUEUE_SUFFIX}"


def get_quart_server_name(service_identifier=None):
    return (
        f"{HOSTED_INFERENCE_SERVER_NAME}-{service_identifier}"
        if service_identifier
        else HOSTED_INFERENCE_SERVER_NAME
    )


def get_gateway_server_name(service_identifier=None):
    return (
        f"{LAUNCH_SERVER_NAME}-{service_identifier}" if service_identifier else LAUNCH_SERVER_NAME
    )


def get_service_builder_logs_location(user_id: str, endpoint_name: str):
    return f"s3://{infra_config().s3_bucket}/service_builder_logs/{user_id}_{endpoint_name}"


def get_k8s_cacher_service_name(service_identifier=None):
    return f"{K8S_CACHER_NAME}-{service_identifier}" if service_identifier else K8S_CACHER_NAME
