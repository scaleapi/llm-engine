"""Shared construction for aioredis connection pools and clients.

redis-py defaults leave pooled connections vulnerable to middlebox idle
timeouts (Istio sidecars, cloud load balancers, NAT): when `retry` and
`retry_on_error` are both unset, `AbstractConnection.__init__` uses
`Retry(NoBackoff(), 0)` — zero retries — so the first command issued on a
silently-closed pooled socket surfaces as `ConnectionError` to the caller.

The helpers here turn on TCP keepalive, a pre-command PING after idle periods,
and transparent retry with exponential backoff. Use them for every long-lived
aioredis pool or client in this repo.
"""

from typing import Any, Dict

import redis.asyncio as aioredis
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

HEALTH_CHECK_INTERVAL_SECONDS = 30
SOCKET_CONNECT_TIMEOUT_SECONDS = 5
RETRY_BACKOFF_CAP_SECONDS = 2.0
RETRY_BACKOFF_BASE_SECONDS = 0.2
RETRY_MAX_ATTEMPTS = 3


def get_aioredis_connection_kwargs() -> Dict[str, Any]:
    """Return kwargs for constructing a resilient aioredis pool or client.

    A fresh Retry and error list are returned on every call so callers can
    never accidentally share mutable state across pools.
    """
    return {
        "socket_keepalive": True,
        "socket_connect_timeout": SOCKET_CONNECT_TIMEOUT_SECONDS,
        "health_check_interval": HEALTH_CHECK_INTERVAL_SECONDS,
        "retry_on_error": [RedisConnectionError, RedisTimeoutError],
        "retry": Retry(
            ExponentialBackoff(cap=RETRY_BACKOFF_CAP_SECONDS, base=RETRY_BACKOFF_BASE_SECONDS),
            retries=RETRY_MAX_ATTEMPTS,
        ),
    }


def build_aioredis_pool(url: str) -> aioredis.ConnectionPool:
    return aioredis.BlockingConnectionPool.from_url(url, **get_aioredis_connection_kwargs())


def build_aioredis_client(url: str) -> aioredis.Redis:
    return aioredis.from_url(url, **get_aioredis_connection_kwargs())
