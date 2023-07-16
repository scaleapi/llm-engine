from unittest.mock import Mock

import aioredis
import pytest

from llm_engine_server.infra.repositories.redis_feature_flag_repository import (
    RedisFeatureFlagRepository,
)


@pytest.mark.asyncio
async def test_read_write_bool(fake_redis):
    aioredis.client.Redis.from_url = Mock(return_value=fake_redis)
    repo = RedisFeatureFlagRepository(redis_info="redis://test")

    assert await repo.read_feature_flag_bool(key="LIRA") is None
    await repo.write_feature_flag_bool(
        key="LIRA",
        value=True,
    )
    value = await repo.read_feature_flag_bool(key="LIRA")
    assert value is True
    fake_redis.force_expire_all()
    assert await repo.read_feature_flag_bool(key="LIRA") is None
