import redis.asyncio as aioredis
from redis.asyncio.retry import Retry
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from model_engine_server.common.aioredis_pool import (
    HEALTH_CHECK_INTERVAL_SECONDS,
    RETRY_MAX_ATTEMPTS,
    SOCKET_CONNECT_TIMEOUT_SECONDS,
    build_aioredis_client,
    build_aioredis_pool,
    get_aioredis_connection_kwargs,
)


def test_kwargs_include_keepalive_health_check_and_retry():
    kwargs = get_aioredis_connection_kwargs()
    assert kwargs["socket_keepalive"] is True
    assert kwargs["socket_connect_timeout"] == SOCKET_CONNECT_TIMEOUT_SECONDS
    assert kwargs["health_check_interval"] == HEALTH_CHECK_INTERVAL_SECONDS
    assert kwargs["retry_on_error"] == [RedisConnectionError, RedisTimeoutError]
    assert isinstance(kwargs["retry"], Retry)


def test_kwargs_do_not_share_mutable_state_across_calls():
    first = get_aioredis_connection_kwargs()
    second = get_aioredis_connection_kwargs()
    assert first["retry"] is not second["retry"]
    assert first["retry_on_error"] is not second["retry_on_error"]


def test_build_aioredis_pool_applies_kwargs_to_connection():
    pool = build_aioredis_pool("redis://localhost:6379/0")
    assert isinstance(pool, aioredis.BlockingConnectionPool)
    conn = pool.connection_class(**pool.connection_kwargs)
    assert conn.socket_keepalive is True
    assert conn.socket_connect_timeout == SOCKET_CONNECT_TIMEOUT_SECONDS
    assert conn.health_check_interval == HEALTH_CHECK_INTERVAL_SECONDS
    # retries is the only reliable way to tell we replaced the default
    # NoBackoff(0) with our configured Retry.
    assert conn.retry._retries == RETRY_MAX_ATTEMPTS


def test_build_aioredis_client_applies_kwargs_to_connection():
    client = build_aioredis_client("redis://localhost:6379/0")
    pool = client.connection_pool
    conn = pool.connection_class(**pool.connection_kwargs)
    assert conn.socket_keepalive is True
    assert conn.health_check_interval == HEALTH_CHECK_INTERVAL_SECONDS
    assert conn.retry._retries == RETRY_MAX_ATTEMPTS
