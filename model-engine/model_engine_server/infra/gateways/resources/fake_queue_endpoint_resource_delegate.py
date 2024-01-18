from typing import Any, Dict, Sequence

from model_engine_server.infra.gateways.resources.queue_endpoint_resource_delegate import (
    QueueEndpointResourceDelegate,
    QueueInfo,
)

__all__: Sequence[str] = ("FakeQueueEndpointResourceDelegate",)


class FakeQueueEndpointResourceDelegate(QueueEndpointResourceDelegate):
    async def create_queue_if_not_exists(
        self,
        endpoint_id: str,
        endpoint_name: str,
        endpoint_created_by: str,
        endpoint_labels: Dict[str, Any],
    ) -> QueueInfo:
        queue_name = QueueEndpointResourceDelegate.endpoint_id_to_queue_name(endpoint_id)
        queue_url = f"http://foobar.com/{queue_name}"
        return QueueInfo(queue_name, queue_url)

    async def delete_queue(self, endpoint_id: str) -> None:
        # Don't need to do anything, since the contract says that deleting is a no-op,
        # and we don't need to simulate real exceptions.
        pass

    async def get_queue_attributes(self, endpoint_id: str) -> Dict[str, Any]:
        return {
            "Attributes": {
                "ApproximateNumberOfMessages": "100",
            },
            "ResponseMetadata": {
                "RequestId": "",
                "HostId": "",
                "HTTPStatusCode": 200,
                "HTTPHeaders": {},
                "RetryAttempts": 0,
            },
        }
