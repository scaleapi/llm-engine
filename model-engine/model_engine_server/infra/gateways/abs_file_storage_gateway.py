from typing import List, Optional

from model_engine_server.domain.gateways.file_storage_gateway import (
    FileMetadata,
    FileStorageGateway,
)
from model_engine_server.infra.gateways import ABSFilesystemGateway


class ABSFileStorageGateway(FileStorageGateway):
    """
    Concrete implementation of a file storage gateway backed by ABS.
    """

    def __init__(self):
        self.filesystem_gateway = ABSFilesystemGateway()

    async def get_url_from_id(self, owner: str, file_id: str) -> Optional[str]:
        raise NotImplementedError("ABS not supported yet")

    async def get_file(self, owner: str, file_id: str) -> Optional[FileMetadata]:
        raise NotImplementedError("ABS not supported yet")

    async def get_file_content(self, owner: str, file_id: str) -> Optional[str]:
        raise NotImplementedError("ABS not supported yet")

    async def upload_file(self, owner: str, filename: str, content: bytes) -> str:
        raise NotImplementedError("ABS not supported yet")

    async def delete_file(self, owner: str, file_id: str) -> bool:
        raise NotImplementedError("ABS not supported yet")

    async def list_files(self, owner: str) -> List[FileMetadata]:
        raise NotImplementedError("ABS not supported yet")
