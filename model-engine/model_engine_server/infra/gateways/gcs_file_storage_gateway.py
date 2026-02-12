import asyncio
import os
from datetime import timedelta
from typing import List, Optional

from gcloud.aio.storage import Storage
from model_engine_server.core.config import infra_config
from model_engine_server.domain.gateways.file_storage_gateway import (
    FileMetadata,
    FileStorageGateway,
)
from model_engine_server.infra.gateways.gcs_storage_client import get_gcs_sync_client, parse_gcs_uri


def _get_gcs_key(owner: str, file_id: str) -> str:
    return os.path.join(owner, file_id)


def _get_gcs_url(owner: str, file_id: str) -> str:
    return f"gs://{infra_config().s3_bucket}/{_get_gcs_key(owner, file_id)}"


def _generate_signed_url_sync(uri: str, expiration: int = 3600) -> str:
    """Generate a V4 signed URL synchronously (gcloud-aio-storage does not support this)."""
    bucket_name, blob_name = parse_gcs_uri(uri)
    client = get_gcs_sync_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.generate_signed_url(
        version="v4",
        expiration=timedelta(seconds=expiration),
        method="GET",
    )


class GCSFileStorageGateway(FileStorageGateway):
    """
    Concrete implementation of a file storage gateway backed by GCS,
    using gcloud-aio-storage for async-native operations.
    """

    async def get_url_from_id(self, owner: str, file_id: str) -> Optional[str]:
        uri = _get_gcs_url(owner, file_id)
        return await asyncio.to_thread(_generate_signed_url_sync, uri)

    async def get_file(self, owner: str, file_id: str) -> Optional[FileMetadata]:
        bucket_name = infra_config().s3_bucket
        blob_name = _get_gcs_key(owner, file_id)
        try:
            async with Storage() as storage:
                metadata = await storage.download_metadata(bucket_name, blob_name)
                return FileMetadata(
                    id=file_id,
                    filename=file_id,
                    size=int(metadata.get("size", 0)),
                    owner=owner,
                    updated_at=metadata.get("updated"),
                )
        except Exception:
            return None

    async def get_file_content(self, owner: str, file_id: str) -> Optional[str]:
        bucket_name = infra_config().s3_bucket
        blob_name = _get_gcs_key(owner, file_id)
        try:
            async with Storage() as storage:
                content = await storage.download(bucket_name, blob_name)
                return content.decode("utf-8")
        except Exception:
            return None

    async def upload_file(self, owner: str, filename: str, content: bytes) -> str:
        bucket_name = infra_config().s3_bucket
        blob_name = _get_gcs_key(owner, filename)
        async with Storage() as storage:
            await storage.upload(bucket_name, blob_name, content)
        return filename

    async def delete_file(self, owner: str, file_id: str) -> bool:
        bucket_name = infra_config().s3_bucket
        blob_name = _get_gcs_key(owner, file_id)
        try:
            async with Storage() as storage:
                await storage.delete(bucket_name, blob_name)
                return True
        except Exception:
            return False

    async def list_files(self, owner: str) -> List[FileMetadata]:
        bucket_name = infra_config().s3_bucket
        async with Storage() as storage:
            files: List[FileMetadata] = []
            params = {"prefix": owner}
            while True:
                response = await storage.list_objects(bucket_name, params=params)
                for item in response.get("items", []):
                    blob_name = item.get("name", "")
                    file_id = blob_name.replace(f"{owner}/", "", 1)
                    files.append(
                        FileMetadata(
                            id=file_id,
                            filename=file_id,
                            size=int(item.get("size", 0)),
                            owner=owner,
                            updated_at=item.get("updated"),
                        )
                    )
                next_token = response.get("nextPageToken")
                if not next_token:
                    break
                params = {"prefix": owner, "pageToken": next_token}
            return files
