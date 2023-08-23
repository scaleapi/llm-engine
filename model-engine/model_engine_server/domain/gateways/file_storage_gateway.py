from abc import ABC, abstractmethod
from typing import List, Optional

from model_engine_server.domain.entities import FileMetadata


class FileStorageGateway(ABC):
    """
    Base class for file storage gateway
    """

    @abstractmethod
    async def get_url_from_id(self, owner: str, file_id: str) -> Optional[str]:
        """
        Get file URL from file ID

        Args:
            owner: The user who owns the file.
            file_id: The ID of the file.

        Returns:
            The URL of the file, or None if it does not exist.
        """
        pass

    @abstractmethod
    async def upload_file(self, owner: str, filename: str, content: bytes) -> str:
        """
        Upload a file

        Args:
            owner: The user who owns the file.
            filename: The name of the file.
            content: The content of the file.

        Returns:
            The ID of the file.
        """
        pass

    @abstractmethod
    async def get_file(self, owner: str, file_id: str) -> Optional[FileMetadata]:
        """
        Get metadata about a file.

        Args:
            owner: The user who owns the file.
            file_id: The ID of the file.

        Returns:
            Information about the file, or None if it does not exist.
        """
        pass

    @abstractmethod
    async def list_files(self, owner: str) -> List[FileMetadata]:
        """
        List all files for a given owner.

        Args:
            owner: The owner whose files to list.

        Returns:
            The list of files.
        """
        pass

    @abstractmethod
    async def delete_file(self, owner: str, file_id: str) -> bool:
        """
        Delete a file.

        Args:
            owner: The user who owns the files.
            file_id: The ID of the file.

        Returns:
            Whether the file was deleted successfully.
        """
        pass

    @abstractmethod
    async def get_file_content(self, owner: str, file_id: str) -> Optional[str]:
        """
        Get a file's content.

        Args:
            owner: The user who owns the file.
            file_id: The ID of the file.

        Returns:
            The content of the file, or None if it does not exist.
        """
        pass
