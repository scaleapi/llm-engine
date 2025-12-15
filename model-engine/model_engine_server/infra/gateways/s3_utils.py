import os
from typing import Any, Dict, Optional

import boto3
from botocore.config import Config
from model_engine_server.core.config import infra_config

_s3_config_logged = False


def _get_onprem_client_kwargs() -> Dict[str, Any]:
    global _s3_config_logged
    client_kwargs: Dict[str, Any] = {}

    s3_endpoint = getattr(infra_config(), "s3_endpoint_url", None) or os.getenv("S3_ENDPOINT_URL")
    if s3_endpoint:
        client_kwargs["endpoint_url"] = s3_endpoint

    addressing_style: str = getattr(infra_config(), "s3_addressing_style", "path")
    client_kwargs["config"] = Config(s3={"addressing_style": addressing_style})

    if not _s3_config_logged and s3_endpoint:
        from model_engine_server.core.loggers import logger_name, make_logger

        logger = make_logger(logger_name())
        logger.info(f"S3 configured for on-prem with endpoint: {s3_endpoint}")
        _s3_config_logged = True

    return client_kwargs


def get_s3_client(kwargs: Optional[Dict[str, Any]] = None) -> Any:
    kwargs = kwargs or {}
    client_kwargs: Dict[str, Any] = {}

    if infra_config().cloud_provider == "onprem":
        client_kwargs = _get_onprem_client_kwargs()
        session = boto3.Session()
    else:
        profile_name = kwargs.get("aws_profile", os.getenv("AWS_PROFILE"))
        session = boto3.Session(profile_name=profile_name)

    return session.client("s3", **client_kwargs)


def get_s3_resource(kwargs: Optional[Dict[str, Any]] = None) -> Any:
    kwargs = kwargs or {}
    resource_kwargs: Dict[str, Any] = {}

    if infra_config().cloud_provider == "onprem":
        resource_kwargs = _get_onprem_client_kwargs()
        session = boto3.Session()
    else:
        profile_name = kwargs.get("aws_profile", os.getenv("AWS_PROFILE"))
        session = boto3.Session(profile_name=profile_name)

    return session.resource("s3", **resource_kwargs)
