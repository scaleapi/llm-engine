# Keep in line with service_config_{*}.yaml
# This file loads sensitive data that shouldn't make it to inference docker images
# Do not include this file in our inference/endpoint code
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import yaml
from azure.identity import DefaultAzureCredential
from model_engine_server.core.aws.secrets import get_key_file
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger

logger = make_logger(logger_name())

__all__: Sequence[str] = (
    "DEFAULT_SERVICE_CONFIG_PATH",
    "SERVICE_CONFIG_PATH",
    "HostedModelInferenceServiceConfig",
    "hmi_config",
)

DEFAULT_SERVICE_CONFIG_PATH = str(
    (
        Path(__file__).absolute().parent.parent.parent
        / "service_configs"
        / "service_config_circleci.yaml"
    ).absolute()
)

SERVICE_CONFIG_PATH = os.environ.get("DEPLOY_SERVICE_CONFIG_PATH", DEFAULT_SERVICE_CONFIG_PATH)

redis_cache_expiration_timestamp = None


# duplicated from llm/finetune_pipeline
def get_model_cache_directory_name(model_name: str):
    """How huggingface maps model names to directory names in their cache for model files.
    We adopt this when storing model cache files in s3.

    Args:
        model_name (str): Name of the huggingface model
    """
    name = "models--" + model_name.replace("/", "--")
    return name


@dataclass
class HostedModelInferenceServiceConfig:
    gateway_namespace: str
    endpoint_namespace: str
    billing_queue_arn: str
    sqs_profile: str
    sqs_queue_policy_template: str
    sqs_queue_tag_template: str
    model_primitive_host: str
    cloud_file_llm_fine_tune_repository: str
    hf_user_fine_tuned_weights_prefix: str
    istio_enabled: bool
    dd_trace_enabled: bool
    tgi_repository: str
    vllm_repository: str
    lightllm_repository: str
    tensorrt_llm_repository: str
    batch_inference_vllm_repository: str
    user_inference_base_repository: str
    user_inference_pytorch_repository: str
    user_inference_tensorflow_repository: str
    docker_image_layer_cache_repository: str
    sensitive_log_mode: bool
    # Exactly one of the following four must be specified
    cache_redis_aws_url: Optional[str] = None  # also using this to store sync autoscaling metrics
    cache_redis_azure_host: Optional[str] = None
    cache_redis_aws_secret_name: Optional[str] = (
        None  # Not an env var because the redis cache info is already here
    )
    cache_redis_gcp_host: Optional[str] = None

    sglang_repository: Optional[str] = None

    @classmethod
    def from_json(cls, json):
        return cls(**{k: v for k, v in json.items() if k in inspect.signature(cls).parameters})

    @classmethod
    def from_yaml(cls, yaml_path):
        with open(yaml_path, "r") as f:
            raw_data = yaml.safe_load(f)
        return HostedModelInferenceServiceConfig.from_json(raw_data)

    @property
    def cache_redis_url(self) -> str:
        if self.cache_redis_aws_url:
            assert infra_config().cloud_provider == "aws", "cache_redis_aws_url is only for AWS"
            if self.cache_redis_aws_secret_name:
                logger.warning(
                    "Both cache_redis_aws_url and cache_redis_aws_secret_name are set. Using cache_redis_aws_url"
                )
            return self.cache_redis_aws_url
        elif self.cache_redis_aws_secret_name:
            assert (
                infra_config().cloud_provider == "aws"
            ), "cache_redis_aws_secret_name is only for AWS"
            creds = get_key_file(self.cache_redis_aws_secret_name)  # Use default role
            return creds["cache-url"]
        elif self.cache_redis_gcp_host:
            assert infra_config().cloud_provider == "gcp"
            return f"rediss://{self.cache_redis_gcp_host}"

        assert self.cache_redis_azure_host and infra_config().cloud_provider == "azure"
        username = os.getenv("AZURE_OBJECT_ID")
        token = DefaultAzureCredential().get_token("https://redis.azure.com/.default")
        password = token.token
        global redis_cache_expiration_timestamp
        redis_cache_expiration_timestamp = token.expires_on
        return f"rediss://{username}:{password}@{self.cache_redis_azure_host}"

    @property
    def cache_redis_url_expiration_timestamp(self) -> Optional[int]:
        global redis_cache_expiration_timestamp
        return redis_cache_expiration_timestamp

    @property
    def cache_redis_host_port(self) -> str:
        # redis://redis.url:6379/<db_index>
        # -> redis.url:6379
        if "rediss://" in self.cache_redis_url:
            return self.cache_redis_url.split("rediss://")[1].split("@")[-1].split("/")[0]
        return self.cache_redis_url.split("redis://")[1].split("/")[0]

    @property
    def cache_redis_db_index(self) -> int:
        # redis://redis.url:6379/<db_index>
        # -> <db_index>
        try:
            return int(self.cache_redis_url.split("/")[-1])
        except ValueError:
            return 0  # 0 is the default index used by redis if it's not specified


def read_default_config():
    logger.info(f"Using config file path: `{SERVICE_CONFIG_PATH}`")
    return HostedModelInferenceServiceConfig.from_yaml(SERVICE_CONFIG_PATH)


_hmi_config: Optional[HostedModelInferenceServiceConfig] = None


def get_hmi_config() -> HostedModelInferenceServiceConfig:
    global _hmi_config
    if _hmi_config is None:
        _hmi_config = read_default_config()
    return _hmi_config


hmi_config = get_hmi_config()
