import os
from typing import List, Optional

from google.cloud import storage

from model_engine_server.core.config import infra_config
from model_engine_server.domain.gateways.file_storage_gateway import (
    FileMetadata,
    FileStorageGateway,
)
from model_engine_server.infra.gateways.gcs_filesystem_gateway import GCSFilesystemGateway


def get_gcs_key(owner: str, file_id: str) -> str:
    """
    Constructs a GCS object key from the owner and file_id.
    """
    return os.path.join(owner, file_id)


def get_gcs_url(owner: str, file_id: str) -> str:
    """
    Returns the gs:// URL for the bucket, using the GCS key.
    """
    return f"gs://{infra_config().gcs_bucket}/{get_gcs_key(owner, file_id)}"


class GCSFileStorageGateway(FileStorageGateway):
    """
    Concrete implementation of a file storage gateway backed by GCS.
    """

    def __init__(self):
        self.filesystem_gateway = GCSFilesystemGateway()

    async def get_url_from_id(self, owner: str, file_id: str) -> Optional[str]:
        """
        Returns a signed GCS URL for the given file.
        """
        try:
            return self.filesystem_gateway.generate_signed_url(get_gcs_url(owner, file_id))
        except Exception:
            return None

    async def get_file(self, owner: str, file_id: str) -> Optional[FileMetadata]:
        """
        Retrieves file metadata if it exists. Returns None if the file is missing.
        """
        try:
            client = self.filesystem_gateway.get_storage_client({})
            bucket = client.bucket(infra_config().gcs_bucket)
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
        """
        Reads and returns the string content of the file.
        """
        try:
            with self.filesystem_gateway.open(get_gcs_url(owner, file_id)) as f:
                return f.read()
        except Exception:
            return None

    async def upload_file(self, owner: str, filename: str, content: bytes) -> str:
        """
        Uploads the file to the GCS bucket. Returns the filename used in bucket.
        """
        with self.filesystem_gateway.open(
            get_gcs_url(owner, filename), mode="w"
        ) as f:
            f.write(content.decode("utf-8"))
        return filename

    async def delete_file(self, owner: str, file_id: str) -> bool:
        """
        Deletes the file from the GCS bucket. Returns True if successful, False otherwise.
        """
        try:
            client = self.filesystem_gateway.get_storage_client({})
            bucket = client.bucket(infra_config().gcs_bucket)
            blob = bucket.blob(get_gcs_key(owner, file_id))
            blob.delete()
            return True
        except Exception:
            return False

    async def list_files(self, owner: str) -> List[FileMetadata]:
        """
        Lists all files in the GCS bucket for the given owner.
        """
        client = self.filesystem_gateway.get_storage_client({})
        blobs = client.list_blobs(infra_config().gcs_bucket, prefix=owner)
        files = [await self.get_file(owner, b.name[len(owner) + 1 :]) for b in blobs if b.name != owner]
        return [f for f in files if f is not None] 