from typing import Any, Dict, Optional, Sequence

import aioredis
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.infra.gateways.resources.queue_endpoint_resource_delegate import (
    QueueEndpointResourceDelegate,
    QueueInfo,
)

logger = make_logger(logger_name())

__all__: Sequence[str] = ("OnPremQueueEndpointResourceDelegate",)


class OnPremQueueEndpointResourceDelegate(QueueEndpointResourceDelegate):
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self._redis_client = redis_client

    def _get_redis_client(self) -> Optional[aioredis.Redis]:
        if self._redis_client is not None:
            return self._redis_client
        try:
            from model_engine_server.api.dependencies import get_or_create_aioredis_pool

            self._redis_client = aioredis.Redis(connection_pool=get_or_create_aioredis_pool())
            return self._redis_client
        except Exception as e:
            logger.warning(f"Failed to initialize Redis client for queue metrics: {e}")
            return None

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
        message_count = 0

        redis_client = self._get_redis_client()
        if redis_client is not None:
            try:
                message_count = await redis_client.llen(queue_name)
            except Exception as e:
                logger.warning(f"Failed to get queue length for {queue_name}: {e}")

        return {
            "Attributes": {
                "ApproximateNumberOfMessages": str(message_count),
                "QueueName": queue_name,
            },
            "ResponseMetadata": {
                "HTTPStatusCode": 200,
            },
        }
