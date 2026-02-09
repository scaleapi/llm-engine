import os
from typing import Any, Dict, Literal, Optional, cast

import boto3
from botocore.config import Config

_s3_config_logged = False

AddressingStyle = Literal["auto", "virtual", "path"]


def _get_cloud_provider() -> str:
    """Get cloud provider with fallback to 'aws' if config fails."""
    try:
        from model_engine_server.core.config import infra_config

        return infra_config().cloud_provider
    except Exception:
        return "aws"


def _get_onprem_client_kwargs() -> Dict[str, Any]:
    """Get S3 client kwargs for on-prem (MinIO) configuration.

    Note: This function is only called when cloud_provider == "onprem",
    which means infra_config() has already succeeded in _get_cloud_provider().
    """
    global _s3_config_logged
    from model_engine_server.core.config import infra_config

    client_kwargs: Dict[str, Any] = {}

    # Get endpoint from config, fall back to env var
    s3_endpoint = getattr(infra_config(), "s3_endpoint_url", None) or os.getenv("S3_ENDPOINT_URL")
    if s3_endpoint:
        client_kwargs["endpoint_url"] = s3_endpoint

    # Get addressing style from config, default to "path" for MinIO compatibility
    addressing_style = cast(AddressingStyle, getattr(infra_config(), "s3_addressing_style", "path"))
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

    cloud_provider = _get_cloud_provider()

    if cloud_provider == "onprem":
        client_kwargs = _get_onprem_client_kwargs()
        session = boto3.Session()
    else:
        # Check aws_profile kwarg, then AWS_PROFILE, then S3_WRITE_AWS_PROFILE for backwards compatibility
        profile_name = kwargs.get(
            "aws_profile", os.getenv("AWS_PROFILE") or os.getenv("S3_WRITE_AWS_PROFILE")
        )
        session = boto3.Session(profile_name=profile_name)

        # Support for MinIO/S3-compatible storage in non-onprem environments (e.g., CircleCI, local dev)
        # This allows S3_ENDPOINT_URL to work even when cloud_provider is "aws"
        s3_endpoint = os.getenv("S3_ENDPOINT_URL")
        if s3_endpoint:
            client_kwargs["endpoint_url"] = s3_endpoint
            # MinIO typically requires path-style addressing
            client_kwargs["config"] = Config(s3={"addressing_style": "path"})

    return session.client("s3", **client_kwargs)


def get_s3_resource(kwargs: Optional[Dict[str, Any]] = None) -> Any:
    kwargs = kwargs or {}
    resource_kwargs: Dict[str, Any] = {}

    cloud_provider = _get_cloud_provider()

    if cloud_provider == "onprem":
        resource_kwargs = _get_onprem_client_kwargs()
        session = boto3.Session()
    else:
        # Check aws_profile kwarg, then AWS_PROFILE, then S3_WRITE_AWS_PROFILE for backwards compatibility
        profile_name = kwargs.get(
            "aws_profile", os.getenv("AWS_PROFILE") or os.getenv("S3_WRITE_AWS_PROFILE")
        )
        session = boto3.Session(profile_name=profile_name)

        # Support for MinIO/S3-compatible storage in non-onprem environments (e.g., CircleCI, local dev)
        # This allows S3_ENDPOINT_URL to work even when cloud_provider is "aws"
        s3_endpoint = os.getenv("S3_ENDPOINT_URL")
        if s3_endpoint:
            resource_kwargs["endpoint_url"] = s3_endpoint
            # MinIO typically requires path-style addressing
            resource_kwargs["config"] = Config(s3={"addressing_style": "path"})

    return session.resource("s3", **resource_kwargs)
