"""Launch Input/Output utils."""

import os
from typing import Any

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
    else:
        from model_engine_server.infra.gateways.s3_utils import get_s3_client

        client = get_s3_client(kwargs)

    transport_params = {"client": client}
    return smart_open.open(uri, mode, transport_params=transport_params)
