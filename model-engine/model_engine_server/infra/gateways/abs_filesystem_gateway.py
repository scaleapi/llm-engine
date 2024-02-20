import os
import re
from datetime import datetime, timedelta
from typing import IO

import smart_open
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobSasPermissions, BlobServiceClient, generate_blob_sas
from model_engine_server.infra.gateways.filesystem_gateway import FilesystemGateway


class ABSFilesystemGateway(FilesystemGateway):
    """
    Concrete implementation for interacting with a filesystem backed by Azure Blob Storage.
    """

    # uri should start with azure:// (as opposed to https://) unless the container is publicly accessible
    def open(self, uri: str, mode: str = "rt", **kwargs) -> IO:
        client = BlobServiceClient(
            f"https://{os.getenv('ABS_ACCOUNT_NAME')}.blob.core.windows.net",
            DefaultAzureCredential(),
        )
        transport_params = {"client": client}
        return smart_open.open(uri, mode, transport_params=transport_params)

    def generate_signed_url(self, uri: str, expiration: int = 3600, **kwargs) -> str:
        match = re.search("^https://([^/]+)\.blob\.core\.windows\.net/([^/]+)/(.*?)$", uri)
        assert match

        account_name, container_name, blob_name = match.group(1), match.group(2), match.group(3)

        blob_service_client = BlobServiceClient(
            f"https://{account_name}.blob.core.windows.net", DefaultAzureCredential()
        )
        user_delegation_key = blob_service_client.get_user_delegation_key(
            datetime.utcnow(), datetime.utcnow() + timedelta(seconds=expiration)
        )

        sas_blob = generate_blob_sas(
            account_name=account_name,
            container_name=container_name,
            blob_name=blob_name,
            user_delegation_key=user_delegation_key,
            permission=BlobSasPermissions(read=True, write=False, create=False),
            expiry=datetime.utcnow() + timedelta(seconds=expiration),
            **kwargs,
        )
        return uri + "?" + sas_blob
