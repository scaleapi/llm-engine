"""
A place for defining, setting, and referencing all environment variables used in SpellbookServe.
"""
import os
from typing import Optional, Sequence

from spellbook_serve.common.constants import PROJECT_ROOT
from spellbook_serve.core.loggers import logger_name, make_logger

__all__: Sequence[str] = (
    "CIRCLECI",
    "SPELLBOOK_SERVE_SERVICE_TEMPLATE_CONFIG_MAP_PATH",
    "SPELLBOOK_SERVE_SERVICE_TEMPLATE_FOLDER",
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
"""Indicates that SpellbookServe is running in a local development environment. Also used for local testing.
"""

WORKSPACE: str = os.environ.get("WORKSPACE", "~/models")
"""The working directory where spellbook_serve is installed.
"""

SPELLBOOK_SERVE_SERVICE_TEMPLATE_CONFIG_MAP_PATH: str = os.environ.get(
    "SPELLBOOK_SERVE_SERVICE_TEMPLATE_CONFIG_MAP_PATH",
    os.path.join(
        PROJECT_ROOT,
        "spellbook_serve/infra/gateways/resources/templates",
        "service_template_config_map_circleci.yaml",
    ),
)
"""The path to the config map containing the SpellbookServe service template.
"""

SPELLBOOK_SERVE_SERVICE_TEMPLATE_FOLDER: Optional[str] = os.environ.get(
    "SPELLBOOK_SERVE_SERVICE_TEMPLATE_FOLDER"
)
"""The path to the folder containing the SpellbookServe service template. If set, this overrides
SPELLBOOK_SERVE_SERVICE_TEMPLATE_CONFIG_MAP_PATH.
"""

if LOCAL:
    logger.warning("LOCAL development & testing mode is ON")
