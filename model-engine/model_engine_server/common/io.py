"""Launch Input/Output utils."""
import os
from typing import Any

import boto3
import smart_open
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from model_engine_server.core.config import infra_config


def open_wrapper(uri: str, mode: str = "rt", **kwargs):
    client: Any
    # This follows the 5.1.0 smart_open API
    if infra_config().cloud_provider == "azure":
        client = BlobServiceClient(
            f"https://{os.getenv('ABS_ACCOUNT_NAME')}.blob.core.windows.net",
            DefaultAzureCredential(),
        )
    else:
        profile_name = kwargs.get("aws_profile", os.getenv("AWS_PROFILE"))
        session = boto3.Session(profile_name=profile_name)
        client = session.client("s3")

    transport_params = {"client": client}
    return smart_open.open(uri, mode, transport_params=transport_params)
