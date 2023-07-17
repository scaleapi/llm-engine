# Keep in line with service_config_{*}.yaml
# This file loads sensitive data that shouldn't make it to inference docker images
# Do not include this file in our inference/endpoint code
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml
from llm_engine_server.core.loggers import filename_wo_ext, make_logger

logger = make_logger(filename_wo_ext(__file__))

__all__: Sequence[str] = (
    "DEFAULT_SERVICE_CONFIG_PATH",
    "SERVICE_CONFIG_PATH",
    "HostedModelInferenceServiceConfig",
    "hmi_config",
)

DEFAULT_SERVICE_CONFIG_PATH = str(
    (
        Path(__file__).absolute().parent.parent.parent / "service_configs" / "service_config.yaml"
    ).absolute()
)

SERVICE_CONFIG_PATH = os.environ.get("DEPLOY_SERVICE_CONFIG_PATH", DEFAULT_SERVICE_CONFIG_PATH)


@dataclass
class HostedModelInferenceServiceConfig:
    endpoint_namespace: str
    cache_redis_url: str
    sqs_profile: str
    sqs_queue_policy_template: str
    sqs_queue_tag_template: str
    s3_file_llm_fine_tuning_job_repository: str

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, "r") as f:
            raw_data = yaml.safe_load(f)
        return HostedModelInferenceServiceConfig(**raw_data)


def read_default_config():
    logger.info(f"Using config file path: `{SERVICE_CONFIG_PATH}`")
    return HostedModelInferenceServiceConfig.from_yaml(SERVICE_CONFIG_PATH)


hmi_config = read_default_config()
