"""Launch Input/Output utils."""

import os
from typing import Any

import boto3
import smart_open
from model_engine_server.core.config import infra_config


def open_wrapper(uri: str, mode: str = "rt", **kwargs):
    client: Any
    try:
        cloud_provider = infra_config().cloud_provider
    except Exception:
        cloud_provider = "aws"

    if cloud_provider == "azure":
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient

        client = BlobServiceClient(
            f"https://{os.getenv('ABS_ACCOUNT_NAME')}.blob.core.windows.net",
            DefaultAzureCredential(),
        )
    elif cloud_provider == "onprem":
        session = boto3.Session()
        client_kwargs = {}

        s3_endpoint = getattr(infra_config(), "s3_endpoint_url", None) or os.getenv(
            "S3_ENDPOINT_URL"
        )
        if s3_endpoint:
            client_kwargs["endpoint_url"] = s3_endpoint

        addressing_style = getattr(infra_config(), "s3_addressing_style", "path")
        client_kwargs["config"] = boto3.session.Config(s3={"addressing_style": addressing_style})

        client = session.client("s3", **client_kwargs)
    else:
        profile_name = kwargs.get("aws_profile", os.getenv("AWS_PROFILE"))
        session = boto3.Session(profile_name=profile_name)
        client = session.client("s3")

    transport_params = {"client": client}
    return smart_open.open(uri, mode, transport_params=transport_params)
