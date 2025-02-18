"""
A place for defining, setting, and referencing all environment variables used in Launch.
"""

import os
import sys
from typing import Optional, Sequence

from model_engine_server.common.constants import PROJECT_ROOT
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger

__all__: Sequence[str] = (
    "CIRCLECI",
    "GIT_TAG",
    "LAUNCH_SERVICE_TEMPLATE_CONFIG_MAP_PATH",
    "LAUNCH_SERVICE_TEMPLATE_FOLDER",
    "LOCAL",
    "SKIP_AUTH",
    "WORKSPACE",
    "get_boolean_env_var",
)

logger = make_logger(logger_name())


def get_boolean_env_var(name: str) -> bool:
    """For all env vars that are either on or off.

    An env var is ON iff:
    - it is defined
    - its value is the literal string 'true'

    If it is present but not set to 'true', it is considered to be OFF.
    """
    value = os.environ.get(name)
    if value is None:
        return False
    value = value.strip().lower()
    return "true" == value


CIRCLECI: bool = get_boolean_env_var("CIRCLECI")

LOCAL: bool = get_boolean_env_var("LOCAL")
"""Indicates that Launch is running in a local development environment. Also used for local testing.
"""

SKIP_AUTH: bool = get_boolean_env_var("SKIP_AUTH") or infra_config().identity_service_url is None
"""Indicates that Launch is running in a development environment where authentication is not
required.
"""

WORKSPACE: str = os.environ.get("WORKSPACE", "~/models")
"""The working directory where hosted_model_inference is installed.
"""

LAUNCH_SERVICE_TEMPLATE_CONFIG_MAP_PATH: str = os.environ.get(
    "LAUNCH_SERVICE_TEMPLATE_CONFIG_MAP_PATH",
    os.path.join(
        PROJECT_ROOT,
        "model_engine_server/infra/gateways/resources/templates",
        "service_template_config_map_circleci.yaml",
    ),
)
"""The path to the config map containing the Launch service template.
"""
logger.info(f"{LAUNCH_SERVICE_TEMPLATE_CONFIG_MAP_PATH=}")

LAUNCH_SERVICE_TEMPLATE_FOLDER: Optional[str] = os.environ.get("LAUNCH_SERVICE_TEMPLATE_FOLDER")
"""The path to the folder containing the Launch service template. If set, this overrides
LAUNCH_SERVICE_TEMPLATE_CONFIG_MAP_PATH.
"""

if LOCAL:
    logger.warning("LOCAL development & testing mode is ON")

# TODO: add a comment here once we understand what this does.
GIT_TAG: str = os.environ.get("GIT_TAG", "GIT_TAG_NOT_FOUND")
if GIT_TAG == "GIT_TAG_NOT_FOUND" and "pytest" not in sys.modules:
    raise ValueError("GIT_TAG environment variable must be set")
