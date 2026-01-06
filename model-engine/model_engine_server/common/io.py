"""Launch Input/Output utils."""

import os
from typing import Any

import boto3
import smart_open
from model_engine_server.core.config import infra_config


def open_wrapper(uri: str, mode: str = "rt", **kwargs):
    client: Any
    cloud_provider: str
    # This follows the 5.1.0 smart_open API
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
    else:
        profile_name = kwargs.get("aws_profile", os.getenv("AWS_PROFILE"))
        # For on-prem: if profile_name is empty/None, use default credential chain (env vars)
        if profile_name:
            session = boto3.Session(profile_name=profile_name)
        else:
            session = boto3.Session()
        
        # Support for MinIO/on-prem S3-compatible storage
        endpoint_url = os.getenv("S3_ENDPOINT_URL")
        client = session.client("s3", endpoint_url=endpoint_url)

    transport_params = {"client": client}
    return smart_open.open(uri, mode, transport_params=transport_params)
