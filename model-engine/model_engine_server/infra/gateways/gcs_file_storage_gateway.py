import os
from typing import List, Optional

from google.auth import default
from google.cloud import storage
from model_engine_server.core.config import infra_config
from model_engine_server.domain.gateways.file_storage_gateway import (
    FileMetadata,
    FileStorageGateway,
)
from model_engine_server.infra.gateways.gcs_filesystem_gateway import GCSFilesystemGateway


def get_gcs_key(owner: str, file_id: str) -> str:
    return os.path.join(owner, file_id)


def get_gcs_url(owner: str, file_id: str) -> str:
    return f"gs://{infra_config().s3_bucket}/{get_gcs_key(owner, file_id)}"


class GCSFileStorageGateway(FileStorageGateway):
    """
    Concrete implementation of a file storage gateway backed by GCS.
    """

    def __init__(self):
        self.filesystem_gateway = GCSFilesystemGateway()

    def _get_client(self) -> storage.Client:
        credentials, project = default()
        return storage.Client(credentials=credentials, project=project)

    def _get_bucket(self) -> storage.Bucket:
        return self._get_client().bucket(infra_config().s3_bucket)

    async def get_url_from_id(self, owner: str, file_id: str) -> Optional[str]:
        return self.filesystem_gateway.generate_signed_url(get_gcs_url(owner, file_id))

    async def get_file(self, owner: str, file_id: str) -> Optional[FileMetadata]:
        try:
            bucket = self._get_bucket()
            blob = bucket.blob(get_gcs_key(owner, file_id))
            blob.reload()  # Fetch metadata
            return FileMetadata(
                id=file_id,
                filename=file_id,
                size=blob.size,
                owner=owner,
                updated_at=blob.updated,
            )
        except Exception:
            return None

    async def get_file_content(self, owner: str, file_id: str) -> Optional[str]:
        try:
            with self.filesystem_gateway.open(get_gcs_url(owner, file_id)) as f:
                return f.read()
        except Exception:
            return None

    async def upload_file(self, owner: str, filename: str, content: bytes) -> str:
        with self.filesystem_gateway.open(get_gcs_url(owner, filename), mode="w") as f:
            f.write(content.decode("utf-8"))
        return filename

    async def delete_file(self, owner: str, file_id: str) -> bool:
        try:
            bucket = self._get_bucket()
            blob = bucket.blob(get_gcs_key(owner, file_id))
            blob.delete()
            return True
        except Exception:
            return False

    async def list_files(self, owner: str) -> List[FileMetadata]:
        bucket = self._get_bucket()
        blobs = bucket.list_blobs(prefix=owner)
        files = []
        for blob in blobs:
            file_id = blob.name.replace(f"{owner}/", "", 1)
            file_metadata = await self.get_file(owner, file_id)
            if file_metadata is not None:
                files.append(file_metadata)
        return files
