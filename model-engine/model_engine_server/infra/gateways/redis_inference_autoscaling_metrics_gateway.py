from typing import Optional

import aioredis
from model_engine_server.domain.gateways.inference_autoscaling_metrics_gateway import (
    InferenceAutoscalingMetricsGateway,
)

EXPIRY_SECONDS = 60  # 1 minute; this gets added to the cooldown time present in the keda ScaledObject to get total
# scaledown time. This also needs to be larger than the keda ScaledObject's refresh rate.
PREWARM_EXPIRY_SECONDS = 60 * 60  # 1 hour


class RedisInferenceAutoscalingMetricsGateway(InferenceAutoscalingMetricsGateway):
    def __init__(
        self, redis_info: Optional[str] = None, redis_client: Optional[aioredis.Redis] = None
    ):
        assert redis_info or redis_client, "Either redis_info or redis_client must be defined."
        if redis_info:
            # If aioredis cannot create a connection pool, reraise that as an error because the
            # default error message is cryptic and not obvious.
            try:
                self._redis = aioredis.from_url(redis_info, health_check_interval=60)
            except Exception as exc:
                raise RuntimeError(
                    "If redis_info is specified, RedisInferenceAutoscalingMetricsGateway must be"
                    "initialized within a coroutine. Please specify the redis_client directly."
                ) from exc
        else:
            assert redis_client is not None  # for mypy
            self._redis = redis_client

    @staticmethod
    def _find_redis_key(endpoint_id: str):
        # Keep in line with keda scaled object yaml
        return f"launch-endpoint-autoscaling:{endpoint_id}"

    async def _emit_metric(self, endpoint_id: str, expiry_time: int):
        key = self._find_redis_key(endpoint_id)
        await self._redis.expire(key, expiry_time)  # does nothing if key doesn't exist,
        # but avoids a race condition where the key expires in between the lpush and subsequent expire commands
        await self._redis.lpush(key, 1)  # we only care about the length of the list, not the values
        await self._redis.ltrim(key, 0, 0)  # we only want to scale from 0 to 1 for redis
        await self._redis.expire(key, expiry_time)

    async def emit_inference_autoscaling_metric(self, endpoint_id: str):
        await self._emit_metric(endpoint_id, EXPIRY_SECONDS)

    async def emit_prewarm_metric(self, endpoint_id: str):
        await self._emit_metric(endpoint_id, PREWARM_EXPIRY_SECONDS)
