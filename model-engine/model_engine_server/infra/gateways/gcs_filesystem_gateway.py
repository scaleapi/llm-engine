import asyncio
from datetime import timedelta
from typing import IO

import smart_open
from gcloud.aio.storage import Storage
from model_engine_server.infra.gateways.filesystem_gateway import FilesystemGateway
from model_engine_server.infra.gateways.gcs_storage_client import (
    get_gcs_sync_client,
    parse_gcs_uri,
)


class GCSFilesystemGateway(FilesystemGateway):
    """
    Concrete implementation for interacting with a filesystem backed by Google Cloud Storage.

    Provides both sync methods (required by FilesystemGateway ABC) and async-native
    counterparts using gcloud-aio-storage for use in async contexts.
    """

    def open(self, uri: str, mode: str = "rt", **kwargs) -> IO:
        client = get_gcs_sync_client()
        transport_params = {"client": client}
        return smart_open.open(uri, mode, transport_params=transport_params)

    def generate_signed_url(self, uri: str, expiration: int = 3600, **kwargs) -> str:
        bucket_name, blob_name = parse_gcs_uri(uri)
        client = get_gcs_sync_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.generate_signed_url(
            version="v4",
            expiration=timedelta(seconds=expiration),
            method="GET",
            **kwargs,
        )

    async def async_read(self, uri: str) -> bytes:
        """Async-native download of blob content."""
        bucket_name, blob_name = parse_gcs_uri(uri)
        async with Storage() as storage:
            return await storage.download(bucket_name, blob_name)

    async def async_write(self, uri: str, content: bytes) -> None:
        """Async-native upload of blob content."""
        bucket_name, blob_name = parse_gcs_uri(uri)
        async with Storage() as storage:
            await storage.upload(bucket_name, blob_name, content)

    async def async_generate_signed_url(
        self, uri: str, expiration: int = 3600, **kwargs
    ) -> str:
        """Async wrapper for signed URL generation (offloaded to a thread)."""
        return await asyncio.to_thread(
            self.generate_signed_url, uri, expiration, **kwargs
        )
