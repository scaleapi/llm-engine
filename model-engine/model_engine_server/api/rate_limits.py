"""Per-user request rate limiting, enforced where identity is resolved.

Limits are fixed 1-second windows in Redis, keyed on (route class, user_id).
The limiter fails open: if Redis is unavailable or slow, requests are allowed.
"""

import asyncio
import time
from typing import Optional, Tuple

import redis.asyncio as aioredis
from fastapi import HTTPException, status
from model_engine_server.common.config import hmi_config
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.core.loggers import logger_name, make_logger

logger = make_logger(logger_name())

# Longest prefix first; (method, path prefix, route class).
_ROUTE_CLASSES: Tuple[Tuple[str, str, str], ...] = (
    ("GET", "/v1/async-tasks/", "get_async_task"),
    ("POST", "/v1/async-tasks", "post_async_tasks"),
)
_WINDOW_SECONDS = 1
# The limiter must stay cheap when the pod is saturated; a slow Redis answer is
# treated as an outage and fails open.
_REDIS_TIMEOUT_SECONDS = 0.1


def _classify_route(method: str, path: str) -> str:
    for route_method, prefix, route_class in _ROUTE_CLASSES:
        if method == route_method and path.startswith(prefix):
            return route_class
    return "default"


def _get_limit(route_class: str) -> Optional[int]:
    config = hmi_config.user_rate_limits
    if not config:
        return None
    routes = config.get("routes") or {}
    limit = routes.get(route_class, routes.get("default"))
    return int(limit) if limit else None


async def _count_request(redis: aioredis.Redis, key: str) -> int:
    async with redis.pipeline(transaction=False) as pipe:
        pipe.incr(key)
        pipe.expire(key, _WINDOW_SECONDS * 2)
        count, _ = await pipe.execute()
    return int(count)


async def enforce_user_rate_limit(
    method: str, path: str, user: User, redis: aioredis.Redis
) -> None:
    """Raises 429 with Retry-After if the user is over their per-route limit.

    No-op unless `user_rate_limits` is configured; counts but does not reject
    unless `user_rate_limits.enforce` is true (log-only rollout mode).
    """
    route_class = _classify_route(method, path)
    limit = _get_limit(route_class)
    if limit is None:
        return
    window = int(time.time()) // _WINDOW_SECONDS
    key = f"user-rate-limit:{route_class}:{user.user_id}:{window}"
    try:
        count = await asyncio.wait_for(_count_request(redis, key), timeout=_REDIS_TIMEOUT_SECONDS)
    except Exception:
        logger.warning(f"Rate limiter failing open for route_class={route_class}", exc_info=True)
        return
    if count <= limit:
        return
    if not (hmi_config.user_rate_limits or {}).get("enforce"):
        logger.warning(
            f"Rate limit exceeded (log-only): user_id={user.user_id} "
            f"route_class={route_class} count={count} limit={limit}"
        )
        return
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=(
            f"Rate limit exceeded for {route_class}: {limit} requests per "
            f"{_WINDOW_SECONDS}s per user. Retry after the indicated delay."
        ),
        headers={"Retry-After": str(_WINDOW_SECONDS)},
    )
