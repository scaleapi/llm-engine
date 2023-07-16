"""
A place for defining, setting, and referencing all environment variables used in LLMEngine.
"""
import os
from typing import Optional, Sequence

from llm_engine_server.common.constants import PROJECT_ROOT
from llm_engine_server.core.loggers import logger_name, make_logger

__all__: Sequence[str] = (
    "CIRCLECI",
    "LLM_ENGINE_SERVICE_TEMPLATE_CONFIG_MAP_PATH",
    "LLM_ENGINE_SERVICE_TEMPLATE_FOLDER",
    "LOCAL",
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
"""Indicates that LLMEngine is running in a local development environment. Also used for local testing.
"""

WORKSPACE: str = os.environ.get("WORKSPACE", "~/models")
"""The working directory where llm_engine is installed.
"""

LLM_ENGINE_SERVICE_TEMPLATE_CONFIG_MAP_PATH: str = os.environ.get(
    "LLM_ENGINE_SERVICE_TEMPLATE_CONFIG_MAP_PATH",
    os.path.join(
        PROJECT_ROOT,
        "llm_engine_server/infra/gateways/resources/templates",
        "service_template_config_map_circleci.yaml",
    ),
)
"""The path to the config map containing the LLMEngine service template.
"""

LLM_ENGINE_SERVICE_TEMPLATE_FOLDER: Optional[str] = os.environ.get(
    "LLM_ENGINE_SERVICE_TEMPLATE_FOLDER"
)
"""The path to the folder containing the LLMEngine service template. If set, this overrides
LLM_ENGINE_SERVICE_TEMPLATE_CONFIG_MAP_PATH.
"""

if LOCAL:
    logger.warning("LOCAL development & testing mode is ON")
