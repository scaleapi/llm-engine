from unittest.mock import Mock

import pytest
import redis.asyncio as aioredis
from model_engine_server.domain.entities import CallbackAuth, CallbackBasicAuth, ModelEndpointConfig
from model_engine_server.infra.repositories.redis_model_endpoint_cache_repository import (
    RedisModelEndpointCacheRepository,
)


@pytest.mark.asyncio
async def test_read_write_cache(entity_model_endpoint_infra_state, fake_redis):
    aioredis.Redis.from_url = Mock(return_value=fake_redis)
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


@pytest.mark.asyncio
async def test_write_cache_drops_default_callback_auth(
    entity_model_endpoint_infra_state, fake_redis
):
    aioredis.Redis.from_url = Mock(return_value=fake_redis)
    repo = RedisModelEndpointCacheRepository(redis_info="redis://test")
    infra_state = entity_model_endpoint_infra_state.copy(deep=True)
    infra_state.user_config_state.endpoint_config = ModelEndpointConfig(
        endpoint_name="test_endpoint",
        bundle_name="test_bundle",
        default_callback_url="https://example.com/callback",
        default_callback_auth=CallbackAuth(
            root=CallbackBasicAuth(kind="basic", username="user", password="secret")
        ),
    )
    await repo.write_endpoint_info(
        endpoint_id="my_endpoint_id", endpoint_info=infra_state, ttl_seconds=60
    )

    cached = await repo.read_endpoint_info(
        endpoint_id="my_endpoint_id", deployment_name=infra_state.deployment_name
    )

    assert "secret" not in str(fake_redis.db)
    expected = infra_state.copy(deep=True)
    assert expected.user_config_state.endpoint_config is not None
    expected.user_config_state.endpoint_config.default_callback_auth = None
    assert cached == expected
    # The caller's object is not modified.
    assert infra_state.user_config_state.endpoint_config.default_callback_auth is not None
