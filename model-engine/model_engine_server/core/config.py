"""AWS configuration for ml-infra-services.

The configuration file is loaded from the ML_INFRA_SERVICES_CONFIG_PATH environment variable.
If this is not set, the default configuration file is used from
model_engine_server.core/configs/default.yaml.
"""
import os
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import yaml
from model_engine_server.core.loggers import logger_name, make_logger

logger = make_logger(logger_name())

__all__: Sequence[str] = (
    "DEFAULT_CONFIG_PATH",
    "CONFIG_PATH",
    "config_context",
    "get_config_path_for_env_name",
    "infra_config",
    "use_config_context",
)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "default.yaml"
CONFIG_PATH: str = os.getenv("ML_INFRA_SERVICES_CONFIG_PATH", str(DEFAULT_CONFIG_PATH))


@dataclass
class InfraConfig:
    cloud_provider: str
    env: str
    k8s_cluster_name: str
    dns_host_domain: str
    default_region: str
    ml_account_id: str
    docker_repo_prefix: str
    redis_host: str
    s3_bucket: str
    profile_ml_worker: str = "default"
    profile_ml_inference_worker: str = "default"
    identity_service_url: Optional[str] = None

    @classmethod
    def from_yaml(cls, yaml_path) -> "InfraConfig":
        with open(yaml_path, "r") as f:
            raw_data = yaml.safe_load(f)
        return InfraConfig(**raw_data)


def read_default_config():
    logger.info(f"Using config file path: `{CONFIG_PATH}`")
    return InfraConfig.from_yaml(CONFIG_PATH)


_infra_config: Optional[InfraConfig] = None


def infra_config() -> InfraConfig:
    global _infra_config
    if _infra_config is None:
        _infra_config = read_default_config()
    return _infra_config


@contextmanager
def config_context(config_path: str):
    """Context manager that temporarily changes the config file path."""
    global _infra_config
    current_config = deepcopy(_infra_config)
    try:
        _infra_config = InfraConfig.from_yaml(config_path)
        yield
    finally:
        _infra_config = current_config


def use_config_context(config_path: str):
    """Use the config file at the given path."""
    global _infra_config
    _infra_config = InfraConfig.from_yaml(config_path)


def get_config_path_for_env_name(env_name: str) -> Path:
    path = DEFAULT_CONFIG_PATH.parent / f"{env_name}.yaml"
    if not path.exists():
        print(path)
        raise ValueError(f"Config file does not exist for env: {env_name}")
    return path
