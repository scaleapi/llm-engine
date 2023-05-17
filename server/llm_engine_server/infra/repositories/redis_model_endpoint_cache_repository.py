import json
from typing import Optional

import aioredis
from llm_engine_server.domain.entities import ModelEndpointInfraState
from llm_engine_server.infra.repositories.model_endpoint_cache_repository import (
    ModelEndpointCacheRepository,
)


class RedisModelEndpointCacheRepository(ModelEndpointCacheRepository):
    # TODO figure out exceptions that can be thrown
    def __init__(
        self,
        redis_info: Optional[str] = None,
        redis_client: Optional[aioredis.Redis] = None,
    ):
        assert redis_info or redis_client, "Either redis_info or redis_client must be defined."
        if redis_info:
            # If aioredis cannot create a connection pool, reraise that as an error because the
            # default error message is cryptic and not obvious.
            try:
                self._redis = aioredis.from_url(redis_info, health_check_interval=60)
            except Exception as exc:
                raise RuntimeError(
                    "If redis_info is specified, RedisModelEndpointCacheRepository must be"
                    "initialized within a coroutine. Please specify the redis_client directly."
                ) from exc
        else:
            assert redis_client is not None  # for mypy
            self._redis = redis_client

    @staticmethod
    def _find_redis_key(key: str):
        return f"llm-engine-k8s-cache:{key}"

    async def write_endpoint_info(
        self,
        endpoint_id: str,
        endpoint_info: ModelEndpointInfraState,
        ttl_seconds: float,
    ):
        key = self._find_redis_key(endpoint_id or endpoint_info.deployment_name)
        endpoint_info_str = json.dumps(endpoint_info.dict())
        await self._redis.set(key, endpoint_info_str, ex=ttl_seconds)

    async def read_endpoint_info(
        self, endpoint_id: str, deployment_name: str
    ) -> Optional[ModelEndpointInfraState]:
        endpoint_id_key = self._find_redis_key(endpoint_id)
        info = await self._redis.get(endpoint_id_key)
        if info is None:
            # TODO is None if not exists
            deployment_name_key = self._find_redis_key(deployment_name)
            info = await self._redis.get(deployment_name_key)
            if info is None:
                return None
        return ModelEndpointInfraState(**json.loads(info))
