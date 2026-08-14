"""Per-user request rate limiting, enforced after authentication resolves identity.

Limits are fixed 1-second windows in Redis, keyed on (route class, user_id).
The limiter fails open: if Redis is unavailable or slow, requests are allowed.
"""

import asyncio
import time
from typing import Optional

import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, status
from model_engine_server.api.dependencies import get_or_create_aioredis_pool, verify_authentication
from model_engine_server.common.config import hmi_config
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.core.loggers import logger_name, make_logger

logger = make_logger(logger_name())

# The limiter must stay cheap when the pod is saturated; a slow Redis answer is
# treated as an outage and fails open.
_REDIS_TIMEOUT_SECONDS = 0.1
_FAIL_OPEN_LOG_INTERVAL_SECONDS = 60.0

_client: Optional[aioredis.Redis] = None
_client_pool: Optional[aioredis.ConnectionPool] = None
_last_fail_open_log: float = 0.0


def _get_client() -> aioredis.Redis:
    # Cached per pool: the pool is rebuilt when its credentials expire, and
    # constructing a Redis client per request copies its full callback tables.
    global _client, _client_pool
    pool = get_or_create_aioredis_pool()
    if _client is None or _client_pool is not pool:
        _client = aioredis.Redis(connection_pool=pool)
        _client_pool = pool
    return _client


async def _count_request(key: str) -> int:
    redis = _get_client()
    count = int(await redis.incr(key))
    if count == 1:
        await redis.expire(key, 2)
    return count


async def enforce_user_rate_limit(route_class: str, user: User) -> None:
    """Raises 429 with Retry-After if the user is over their per-route limit.

    No-op unless `user_rate_limits` is configured; counts but does not reject
    unless `user_rate_limits.enforce` is true (log-only rollout mode).
    """
    config = hmi_config.user_rate_limits or {}
    limit = (config.get("routes") or {}).get(route_class)
    if not limit:
        return
    limit = int(limit)
    key = f"user-rate-limit:{route_class}:{user.user_id}:{int(time.time())}"
    try:
        count = await asyncio.wait_for(_count_request(key), timeout=_REDIS_TIMEOUT_SECONDS)
    except Exception:
        # Sampled: during a Redis outage this fires on every authenticated request.
        global _last_fail_open_log
        now = time.monotonic()
        if now - _last_fail_open_log >= _FAIL_OPEN_LOG_INTERVAL_SECONDS:
            _last_fail_open_log = now
            logger.warning(
                f"Rate limiter failing open for route_class={route_class}", exc_info=True
            )
        return
    if count <= limit:
        return
    if not config.get("enforce"):
        logger.warning(
            f"Rate limit exceeded (log-only): user_id={user.user_id} "
            f"route_class={route_class} count={count} limit={limit}"
        )
        return
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=(
            f"Rate limit exceeded for {route_class}: {limit} requests per second per user. "
            "Retry after the indicated delay."
        ),
        headers={"Retry-After": "1"},
    )


def user_rate_limit(route_class: str):
    """FastAPI dependency limiting the authenticated user on this route.

    Composes with verify_authentication (FastAPI caches it per request, so
    authentication still runs once).
    """

    async def dependency(auth: User = Depends(verify_authentication)) -> None:
        await enforce_user_rate_limit(route_class, auth)

    return dependency
