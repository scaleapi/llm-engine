from abc import ABC, abstractmethod
from typing import IO


class FilesystemGateway(ABC):
    """
    Abstract Base Class for interacting with a filesystem.
    """

    @abstractmethod
    def open(self, uri: str, mode: str = "rt", **kwargs) -> IO:
        """
        Opens a file with the given mode.
        """
        pass

    @abstractmethod
    def generate_signed_url(self, uri: str, expiration: int = 3600, **kwargs) -> str:
        """
        Generates a signed URI for the given URI.
        """
        pass
