from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple, Sequence

from mypy_boto3_sqs.type_defs import GetQueueAttributesResultTypeDef

__all__: Sequence[str] = (
    "SQSQueueInfo",
    "SQSEndpointResourceDelegate",
)


class SQSQueueInfo(NamedTuple):
    queue_name: str
    queue_url: str


class SQSEndpointResourceDelegate(ABC):
    """
    Base class for an interactor with SQS. This is used by the LiveEndpointResourceGateway.
    """

    @abstractmethod
    async def create_queue_if_not_exists(
        self,
        endpoint_id: str,
        endpoint_name: str,
        endpoint_created_by: str,
        endpoint_labels: Dict[str, Any],
    ) -> SQSQueueInfo:
        """
        Creates an SQS queue associated with the given endpoint_id. Other fields are set as tags on the queue.
        """

    @abstractmethod
    async def delete_queue(self, endpoint_id: str) -> None:
        """
        Deletes an SQS queue associated with the given endpoint_id. This is a no-op if the queue does not exist.
        """

    @abstractmethod
    async def get_queue_attributes(self, endpoint_id: str) -> GetQueueAttributesResultTypeDef:
        """
        Get all attributes of an SQS queue.
        """

    @staticmethod
    def endpoint_id_to_queue_name(endpoint_id: str) -> str:
        return f"spellbook-serve-endpoint-id-{endpoint_id}"
