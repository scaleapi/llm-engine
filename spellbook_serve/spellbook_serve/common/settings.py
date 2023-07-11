# This file contains standard settings for ML serve.
#

import hashlib
from typing import List

from spellbook_serve.core.config import ml_infra_config

DEPLOYMENT_PREFIX = "spellbook-serve"
SERVICE_BUILDER_QUEUE_PREFIX = "spellbook-serve"
SERVICE_BUILDER_QUEUE_SUFFIX = "service-builder"

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


def get_service_builder_queue(service_identifier=None):
    return (
        f"{SERVICE_BUILDER_QUEUE_PREFIX}-{service_identifier}.{SERVICE_BUILDER_QUEUE_SUFFIX}"
        if service_identifier
        else f"{SERVICE_BUILDER_QUEUE_PREFIX}.{SERVICE_BUILDER_QUEUE_SUFFIX}"
    )


def get_service_builder_logs_location(user_id: str, endpoint_name: str):
    return f"s3://{ml_infra_config().s3_bucket}/service_builder_logs/{user_id}_{endpoint_name}"
