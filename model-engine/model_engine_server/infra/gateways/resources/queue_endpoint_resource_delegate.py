from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple, Optional, Sequence

__all__: Sequence[str] = (
    "QueueInfo",
    "QueueEndpointResourceDelegate",
)


class QueueInfo(NamedTuple):
    queue_name: str
    queue_url: Optional[str]


class QueueEndpointResourceDelegate(ABC):
    """
    Base class for an interactor with SQS or ASB. This is used by the LiveEndpointResourceGateway.
    """

    @abstractmethod
    async def create_queue_if_not_exists(
        self,
        endpoint_id: str,
        endpoint_name: str,
        endpoint_created_by: str,
        endpoint_labels: Dict[str, Any],
    ) -> QueueInfo:
        """
        Creates a queue associated with the given endpoint_id. Other fields are set as tags on the queue.
        """

    @abstractmethod
    async def delete_queue(self, endpoint_id: str) -> None:
        """
        Deletes a queue associated with the given endpoint_id. This is a no-op if the queue does not exist.
        """

    @abstractmethod
    async def get_queue_attributes(self, endpoint_id: str) -> Dict[str, Any]:
        """
        Get attributes of a queue.
        """

    @staticmethod
    def endpoint_id_to_queue_name(endpoint_id: str) -> str:
        return f"launch-endpoint-id-{endpoint_id}"
