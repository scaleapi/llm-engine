from typing import List, Optional

from model_engine_server.domain.gateways.file_storage_gateway import (
    FileMetadata,
    FileStorageGateway,
)
from model_engine_server.infra.gateways.gcs_filesystem_gateway import GCSFilesystemGateway


class GCSFileStorageGateway(FileStorageGateway):
    """
    Concrete implementation of a file storage gateway backed by GCS.
    """

    def __init__(self):
        self.filesystem_gateway = GCSFilesystemGateway()

    async def get_url_from_id(self, owner: str, file_id: str) -> Optional[str]:
        raise NotImplementedError("GCS file storage not fully implemented yet")

    async def get_file(self, owner: str, file_id: str) -> Optional[FileMetadata]:
        raise NotImplementedError("GCS file storage not fully implemented yet")

    async def get_file_content(self, owner: str, file_id: str) -> Optional[str]:
        raise NotImplementedError("GCS file storage not fully implemented yet")

    async def upload_file(self, owner: str, filename: str, content: bytes) -> str:
        raise NotImplementedError("GCS file storage not fully implemented yet")

    async def delete_file(self, owner: str, file_id: str) -> bool:
        raise NotImplementedError("GCS file storage not fully implemented yet")

    async def list_files(self, owner: str) -> List[FileMetadata]:
        raise NotImplementedError("GCS file storage not fully implemented yet")
