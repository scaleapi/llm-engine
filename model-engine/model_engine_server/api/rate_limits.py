"""Per-user request rate limiting, enforced after authentication resolves identity.

Limits are fixed 1-second windows in Redis, keyed on (route class, user_id) and,
for routes that declare a scope query parameter, additionally on that parameter's
value (e.g. post_async_tasks is limited per (user, model_endpoint_id), so one
noisy pipeline behind a shared credential cannot starve the credential's other
pipelines). The limiter fails open: if Redis is unavailable or slow, requests
are allowed.
"""

import asyncio
import time
from typing import Optional

import redis.asyncio as aioredis
from datadog import statsd
from fastapi import Depends, HTTPException, Request, status
from model_engine_server.api.dependencies import get_or_create_aioredis_pool, verify_authentication
from model_engine_server.common.config import hmi_config
from model_engine_server.common.env_vars import DD_ENV
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.core.loggers import logger_name, make_logger

logger = make_logger(logger_name())

# The limiter must stay cheap when the pod is saturated; a slow Redis answer is
# treated as an outage and fails open.
_REDIS_TIMEOUT_SECONDS = 0.1
_LOG_INTERVAL_SECONDS = 60.0
# Circuit breaker: each timed-out call abandons its pooled connection, so during a
# Redis brownout per-request checks become a reconnect storm against the slow Redis
# (and churn the shared cache pool). After enough consecutive failures, fail open
# without touching Redis for a cooldown period.
_BREAKER_FAILURE_THRESHOLD = 5
_BREAKER_COOLDOWN_SECONDS = 10.0

# Atomic so a partial failure cannot leave a counter key without a TTL. Counts
# every provided key in one round trip (scoped routes check the per-user
# aggregate and the per-scope bucket together) and returns the counts in order.
_INCR_WITH_TTL_LUA = """
local counts = {}
for i, key in ipairs(KEYS) do
  local count = redis.call('INCR', key)
  if count == 1 then
    redis.call('EXPIRE', key, 2)
  end
  counts[i] = count
end
return counts
"""
# Rotating caller-supplied scope values must not bypass the per-user limit, so
# scoped routes always carry an aggregate ceiling; when aggregate_routes does
# not configure one it defaults to this multiple of the per-scope limit.
_DEFAULT_AGGREGATE_MULTIPLIER = 4

_client: Optional[aioredis.Redis] = None
_client_pool: Optional[aioredis.ConnectionPool] = None
_last_log_times: dict = {}
_consecutive_failures: int = 0
_breaker_open_until: float = 0.0


def _emit_decision(
    outcome: str, route_class: str, user_id: str, scope: Optional[str] = None
) -> None:
    # Fire-and-forget UDP to the local Datadog agent. This doubles as the per-tenant
    # volume signal on the rate-limited routes (outcome:allowed) and as enforcement
    # telemetry (throttled/would_throttle/fail_open/breaker_open), replacing the
    # postmortem's proposed raw request-volume monitor.
    tags = [
        f"env:{DD_ENV}",
        f"route_class:{route_class}",
        f"outcome:{outcome}",
        f"user_id:{user_id}",
    ]
    # The scope is caller-supplied: tagging it on allowed traffic (or on the
    # fail-open/breaker outcomes, which fire for every request during a Redis
    # outage) would let a client mint unbounded metric cardinality. Over-limit
    # outcomes require sustained volume on a single scope value, which bounds
    # the cardinality an abuser can create.
    if scope is not None and outcome in ("throttled", "would_throttle"):
        tags.append(f"scope:{scope}")
    statsd.increment("model_engine.user_rate_limit.decision", tags=tags)


def _should_log(log_key: str) -> bool:
    # Both limiter logs fire per request when things go wrong (Redis outage, or a
    # noisy tenant in log-only mode), so they are sampled per key.
    now = time.monotonic()
    if now - _last_log_times.get(log_key, 0.0) < _LOG_INTERVAL_SECONDS:
        return False
    if len(_last_log_times) > 1000:
        _last_log_times.clear()
    _last_log_times[log_key] = now
    return True


def _get_client() -> aioredis.Redis:
    # Cached per pool: the pool is rebuilt when its credentials expire, and
    # constructing a Redis client per request copies its full callback tables.
    global _client, _client_pool
    pool = get_or_create_aioredis_pool()
    if _client is None or _client_pool is not pool:
        _client = aioredis.Redis(connection_pool=pool)
        _client_pool = pool
    return _client


async def _count_requests(keys: list) -> list:
    counts = await _get_client().eval(_INCR_WITH_TTL_LUA, len(keys), *keys)
    return [int(count) for count in counts]


async def enforce_user_rate_limit(
    route_class: str, user: User, scope: Optional[str] = None, limit: Optional[int] = None
) -> None:
    """Raises 429 with Retry-After if the user is over their per-route limit.

    When `scope` is set, the limit applies per (user, scope) rather than per user,
    and `user_rate_limits.aggregate_routes` may additionally bound the user's
    total across all scopes. No-op unless `user_rate_limits` is configured;
    counts but does not reject unless `user_rate_limits.enforce` is true
    (log-only rollout mode).
    """
    config = hmi_config.user_rate_limits or {}
    if limit is None:
        limit = (config.get("routes") or {}).get(route_class)
    if not limit:
        return
    limit = int(limit)
    global _consecutive_failures, _breaker_open_until
    if time.monotonic() < _breaker_open_until:
        _emit_decision("breaker_open", route_class, user.user_id, scope)
        return
    second = int(time.time())
    base_key = f"user-rate-limit:{route_class}:{user.user_id}"
    keys = []
    limits = []
    if scope is not None:
        # The scope value is caller-supplied, so scoped buckets alone are
        # bypassable by rotating bogus values; the aggregate per-user ceiling
        # (configured, or a default multiple of the per-scope limit) bounds the
        # user's total rate across all scopes. Both buckets are counted in one
        # Redis round trip under the shared timeout and emit one decision.
        aggregate_limit = (config.get("aggregate_routes") or {}).get(route_class)
        if aggregate_limit:
            aggregate_limit = int(aggregate_limit)
        else:
            aggregate_limit = limit * _DEFAULT_AGGREGATE_MULTIPLIER
        keys.append(f"{base_key}:{second}")
        limits.append(aggregate_limit)
        keys.append(f"{base_key}:{scope}:{second}")
        limits.append(limit)
    else:
        keys.append(f"{base_key}:{second}")
        limits.append(limit)
    try:
        counts = await asyncio.wait_for(_count_requests(keys), timeout=_REDIS_TIMEOUT_SECONDS)
        _consecutive_failures = 0
    except Exception:
        _consecutive_failures += 1
        if _consecutive_failures >= _BREAKER_FAILURE_THRESHOLD:
            _breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECONDS
            _consecutive_failures = 0
        if _should_log("fail-open"):
            logger.warning(
                f"Rate limiter failing open for route_class={route_class}", exc_info=True
            )
        _emit_decision("fail_open", route_class, user.user_id, scope)
        return
    exceeded = next(
        (
            (count, bucket_limit)
            for count, bucket_limit in zip(counts, limits)
            if count > bucket_limit
        ),
        None,
    )
    if exceeded is None:
        _emit_decision("allowed", route_class, user.user_id, scope)
        return
    count, limit = exceeded
    if not config.get("enforce"):
        if _should_log(f"over-limit:{user.user_id}:{route_class}"):
            logger.warning(
                f"Rate limit exceeded (log-only): user_id={user.user_id} "
                f"route_class={route_class} scope={scope} count={count} limit={limit}"
            )
        _emit_decision("would_throttle", route_class, user.user_id, scope)
        return
    _emit_decision("throttled", route_class, user.user_id, scope)
    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=(
            f"Rate limit exceeded for {route_class}: {limit} requests per second per user. "
            "Retry after the indicated delay."
        ),
        headers={"Retry-After": "1"},
    )


def user_rate_limit(route_class: str, scope_query_param: Optional[str] = None):
    """FastAPI dependency limiting the authenticated user on this route.

    When `scope_query_param` names a query parameter, its value is folded into
    the bucket key so the limit applies per (user, scope) instead of per user.
    Composes with verify_authentication (FastAPI caches it per request, so
    authentication still runs once).
    """

    async def dependency(request: Request, auth: User = Depends(verify_authentication)) -> None:
        scope = request.query_params.get(scope_query_param) if scope_query_param else None
        await enforce_user_rate_limit(route_class, auth, scope)

    return dependency
