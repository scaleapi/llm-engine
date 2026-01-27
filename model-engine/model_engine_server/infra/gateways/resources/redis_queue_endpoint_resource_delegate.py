"""Redis-based queue endpoint resource delegate for GCP deployments.

When using Redis (Memorystore) as the Celery broker on GCP, queues are implicit
Redis lists that don't need explicit creation/deletion. This delegate manages
queue lifecycle for async endpoints using Redis.
"""

from typing import Any, Dict, Sequence

from model_engine_server.core.celery.app import get_redis_instance
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.infra.gateways.resources.queue_endpoint_resource_delegate import (
    QueueEndpointResourceDelegate,
    QueueInfo,
)

logger = make_logger(logger_name())

__all__: Sequence[str] = ("RedisQueueEndpointResourceDelegate",)


class RedisQueueEndpointResourceDelegate(QueueEndpointResourceDelegate):
    """
    Redis-based queue delegate for GCP deployments using Memorystore.

    Redis queues (lists) are created implicitly when messages are pushed,
    so this delegate mainly handles queue name management and metrics retrieval.
    """

    async def create_queue_if_not_exists(
        self,
        endpoint_id: str,
        endpoint_name: str,
        endpoint_created_by: str,
        endpoint_labels: Dict[str, Any],
    ) -> QueueInfo:
        """
        For Redis, queues are created implicitly. We just return the queue name.
        The queue_url is None since Redis doesn't use URLs for queue access.
        """
        queue_name = QueueEndpointResourceDelegate.endpoint_id_to_queue_name(endpoint_id)
        logger.info(f"Redis queue ready for endpoint: {queue_name}")
        return QueueInfo(queue_name=queue_name, queue_url=None)

    async def delete_queue(self, endpoint_id: str) -> None:
        """
        Delete the Redis queue (list) for the endpoint.
        This removes all pending messages in the queue.
        """
        queue_name = QueueEndpointResourceDelegate.endpoint_id_to_queue_name(endpoint_id)
        try:
            redis = get_redis_instance()
            # Delete the queue (Redis list)
            deleted = redis.delete(queue_name)
            if deleted:
                logger.info(f"Deleted Redis queue: {queue_name}")
            else:
                logger.info(f"Redis queue already empty or doesn't exist: {queue_name}")
            redis.close()
        except Exception as e:
            logger.warning(f"Error deleting Redis queue {queue_name}: {e}")

    async def get_queue_attributes(self, endpoint_id: str) -> Dict[str, Any]:
        """
        Get queue attributes including the approximate number of messages.
        """
        queue_name = QueueEndpointResourceDelegate.endpoint_id_to_queue_name(endpoint_id)
        try:
            redis = get_redis_instance()
            queue_length = redis.llen(queue_name)
            redis.close()

            # Return in a format compatible with the existing code
            # that checks for "Attributes.ApproximateNumberOfMessages"
            return {
                "name": queue_name,
                "Attributes": {
                    "ApproximateNumberOfMessages": str(queue_length),
                },
            }
        except Exception as e:
            logger.warning(f"Error getting Redis queue attributes for {queue_name}: {e}")
            return {
                "name": queue_name,
                "Attributes": {
                    "ApproximateNumberOfMessages": "0",
                },
            }
