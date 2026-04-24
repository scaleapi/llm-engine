from typing import Optional

import redis.asyncio as aioredis
from model_engine_server.common.aioredis_pool import build_aioredis_client
from model_engine_server.infra.repositories.feature_flag_repository import FeatureFlagRepository


class RedisFeatureFlagRepository(FeatureFlagRepository):
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
                self._redis = build_aioredis_client(redis_info)
            except Exception as exc:
                raise RuntimeError(
                    "If redis_info is specified, RedisFeatureFlagRepository must be"
                    "initialized within a coroutine. Please specify the redis_client directly."
                ) from exc
        else:
            assert redis_client is not None  # for mypy
            self._redis = redis_client

    @staticmethod
    def _to_redis_key(key: str):
        return f"launch-feature-flag:{key}"

    async def write_feature_flag_bool(self, key: str, value: bool):
        if not isinstance(value, bool):
            raise TypeError(
                f"Expected a bool value when setting value for feature flag {key=}, got {type(value)}"
            )
        await self._redis.set(self._to_redis_key(key), str(value))

    async def read_feature_flag_bool(self, key: str) -> Optional[bool]:
        flag = await self._redis.get(self._to_redis_key(key))
        if flag is None:
            return None
        flag = flag.decode()
        if flag not in ("True", "False"):
            raise TypeError(
                f"Expected the value for feature flag {key=} to be either 'True' or 'False', got {flag=}"
            )
        return flag == "True"
