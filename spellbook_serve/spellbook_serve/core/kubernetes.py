import logging
from enum import Enum
from pathlib import Path
from string import Template
from typing import Iterator, Union

import yaml
from kubeconfig import KubeConfig

from .loggers import make_logger

logger = make_logger(__file__, log_level=logging.DEBUG)
_config = KubeConfig()

_K8S_CONFIGS = {}


class LifecycleSelector(str, Enum):
    NORMAL = "normal"
    SPOT = "spot"


def k8s_config() -> str:
    """Returns the name of the current kubernetes context"""
    return _config.view()["current-context"].strip()


def check_k8s_config(env_name: str) -> bool:
    """
    Checks whether the current k8s context (i.e. which cluster you're on)
    is the one given by the config.
    """
    assert env_name in _K8S_CONFIGS
    cur_config = k8s_config()
    return cur_config.strip() == _K8S_CONFIGS[env_name].strip()


def substitute_yaml(fp: Union[str, Path], **kwargs) -> dict:
    """Read a file from disk, substitute options, return yaml

    The yaml file must have the variables to substitute written as $VAR or ${VAR}. See documentation
    for string.Template for more details.

    Args:
        fp: path to a yaml file
        **kwargs: all the keyword arguments needed to substitute flags in the yaml file

    Returns:
        Returns a dict of parsed yaml

    Raises:
        FileNotFoundError: If no file exists at the path
        KeyError: If a keyword argument is specified for a key that doesn't exist, or a key is
            specified and no corresponding argument is passed in.
    """
    with open(fp, "r") as template_f:
        config = yaml.safe_load(Template(template_f.read()).substitute(**kwargs))
    return config


def substitute_yamls(fp: Union[str, Path], **kwargs) -> Iterator:
    """Read a file from disk, substitute options, return yaml

    The yaml file must have the variables to substitute written as $VAR or ${VAR}. See documentation
    for string.Template for more details.

    Args:
        fp: path to a yaml file
        **kwargs: all the keyword arguments needed to substitute flags in the yaml file

    Returns:
        Returns a list of dicts of parsed yaml

    Raises:
        FileNotFoundError: If no file exists at the path
        KeyError: If a keyword argument is specified for a key that doesn't exist, or a key is
            specified and no corresponding argument is passed in.
    """
    with open(fp, "r") as template_f:
        config = yaml.safe_load_all(Template(template_f.read()).substitute(**kwargs))
    return config
