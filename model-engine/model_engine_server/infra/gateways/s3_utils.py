import os
from typing import Any, Dict, Optional

import boto3
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger

logger = make_logger(logger_name())


def get_s3_client(kwargs: Optional[Dict[str, Any]] = None):
    kwargs = kwargs or {}
    session = boto3.Session()
    client_kwargs = {}

    if infra_config().cloud_provider == "onprem":
        logger.debug("Using on-prem/MinIO S3-compatible configuration")

        s3_endpoint = getattr(infra_config(), "s3_endpoint_url", None) or os.getenv(
            "S3_ENDPOINT_URL"
        )
        if s3_endpoint:
            client_kwargs["endpoint_url"] = s3_endpoint
            logger.debug(f"Using S3 endpoint: {s3_endpoint}")

        addressing_style = getattr(infra_config(), "s3_addressing_style", "path")
        client_kwargs["config"] = boto3.session.Config(s3={"addressing_style": addressing_style})
    else:
        logger.debug("Using AWS S3 configuration")
        aws_profile = kwargs.get("aws_profile")
        if aws_profile:
            session = boto3.Session(profile_name=aws_profile)

    return session.client("s3", **client_kwargs)


def get_s3_resource(kwargs: Optional[Dict[str, Any]] = None):
    kwargs = kwargs or {}
    session = boto3.Session()
    resource_kwargs = {}

    if infra_config().cloud_provider == "onprem":
        logger.debug("Using on-prem/MinIO S3-compatible configuration")

        s3_endpoint = getattr(infra_config(), "s3_endpoint_url", None) or os.getenv(
            "S3_ENDPOINT_URL"
        )
        if s3_endpoint:
            resource_kwargs["endpoint_url"] = s3_endpoint
            logger.debug(f"Using S3 endpoint: {s3_endpoint}")

        addressing_style = getattr(infra_config(), "s3_addressing_style", "path")
        resource_kwargs["config"] = boto3.session.Config(s3={"addressing_style": addressing_style})
    else:
        logger.debug("Using AWS S3 configuration")
        aws_profile = kwargs.get("aws_profile")
        if aws_profile:
            session = boto3.Session(profile_name=aws_profile)

    return session.resource("s3", **resource_kwargs)
