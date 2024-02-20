from abc import ABC, abstractmethod
from typing import Any, Dict


class StreamingStorageGateway(ABC):
    """
    Base class for a gateway that stores data through a streaming mechanism.
    """

    @abstractmethod
    def put_record(self, stream_name: str, record: Dict[str, Any]) -> Any:
        """
        Put a record into a streaming storage mechanism.

        Args:
            stream_name: The name of the stream.
            record: The record to put into the stream.
        """
        pass
