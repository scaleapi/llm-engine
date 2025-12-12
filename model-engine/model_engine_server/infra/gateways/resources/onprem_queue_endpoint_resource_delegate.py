from typing import Any, Dict, Sequence

from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.infra.gateways.resources.queue_endpoint_resource_delegate import (
    QueueEndpointResourceDelegate,
    QueueInfo,
)

logger = make_logger(logger_name())

__all__: Sequence[str] = ("OnPremQueueEndpointResourceDelegate",)


class OnPremQueueEndpointResourceDelegate(QueueEndpointResourceDelegate):
    async def create_queue_if_not_exists(
        self,
        endpoint_id: str,
        endpoint_name: str,
        endpoint_created_by: str,
        endpoint_labels: Dict[str, Any],
    ) -> QueueInfo:
        queue_name = QueueEndpointResourceDelegate.endpoint_id_to_queue_name(endpoint_id)

        logger.debug(
            f"On-prem queue for endpoint {endpoint_id}: {queue_name} "
            f"(Redis queues don't require explicit creation)"
        )

        return QueueInfo(queue_name=queue_name, queue_url=queue_name)

    async def delete_queue(self, endpoint_id: str) -> None:
        queue_name = QueueEndpointResourceDelegate.endpoint_id_to_queue_name(endpoint_id)
        logger.debug(f"Delete request for queue {queue_name} (no-op for Redis-based queues)")

    async def get_queue_attributes(self, endpoint_id: str) -> Dict[str, Any]:
        queue_name = QueueEndpointResourceDelegate.endpoint_id_to_queue_name(endpoint_id)

        logger.warning(
            f"Getting queue attributes for {queue_name} - returning hardcoded values. "
            f"On-prem Redis queues do not support real-time message counts. "
            f"Do not rely on ApproximateNumberOfMessages for autoscaling decisions."
        )

        return {
            "Attributes": {
                "ApproximateNumberOfMessages": "0",
                "QueueName": queue_name,
            },
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
            },
        }
