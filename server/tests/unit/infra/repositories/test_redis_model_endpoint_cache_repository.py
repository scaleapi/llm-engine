from unittest.mock import Mock

import aioredis
import pytest
from llm_engine_server.infra.repositories.redis_model_endpoint_cache_repository import (
    RedisModelEndpointCacheRepository,
)


@pytest.mark.asyncio
async def test_read_write_cache(entity_model_endpoint_infra_state, fake_redis):
    aioredis.client.Redis.from_url = Mock(return_value=fake_redis)
    repo = RedisModelEndpointCacheRepository(redis_info="redis://test")
    endpoint_id = "my_endpoint_id"
    assert not await repo.read_endpoint_info(
        endpoint_id=endpoint_id,
        deployment_name=entity_model_endpoint_infra_state.deployment_name,
    )
    await repo.write_endpoint_info(
        endpoint_id=endpoint_id,
        endpoint_info=entity_model_endpoint_infra_state,
        ttl_seconds=60,
    )
    infra_state = await repo.read_endpoint_info(
        endpoint_id=endpoint_id,
        deployment_name=entity_model_endpoint_infra_state.deployment_name,
    )
    assert infra_state == entity_model_endpoint_infra_state
    print(fake_redis.db)
    fake_redis.force_expire_all()
    assert not await repo.read_endpoint_info(
        endpoint_id=endpoint_id,
        deployment_name=entity_model_endpoint_infra_state.deployment_name,
    )
