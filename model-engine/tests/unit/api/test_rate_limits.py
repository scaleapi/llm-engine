import asyncio

import pytest
from fastapi import HTTPException
from model_engine_server.api import rate_limits
from model_engine_server.core.auth.authentication_repository import User


class FakeRateLimitRedis:
    """Counter-only fake for the limiter's eval call."""

    def __init__(self, count=1, error=None, delay=0.0):
        self.count = count
        self.error = error
        self.delay = delay

    async def eval(self, script, numkeys, *keys):
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return self.count


USER = User(user_id="test-user", team_id="test-team", is_privileged_user=False)
LIMITS = {"enforce": True, "routes": {"get_async_task": 5}}
LIMITS_LOG_ONLY = {"enforce": False, "routes": {"get_async_task": 5}}


@pytest.mark.parametrize(
    "config,route_class,count,expect_429",
    [
        pytest.param(None, "get_async_task", 100, False, id="disabled-no-config"),
        pytest.param(LIMITS, "get_async_task", 5, False, id="at-limit-allowed"),
        pytest.param(LIMITS, "get_async_task", 6, True, id="over-limit-rejected"),
        pytest.param(LIMITS, "post_async_tasks", 100, False, id="unconfigured-route-unlimited"),
        pytest.param(LIMITS_LOG_ONLY, "get_async_task", 100, False, id="log-only-allows"),
    ],
)
@pytest.mark.asyncio
async def test_enforce_user_rate_limit(monkeypatch, config, route_class, count, expect_429):
    monkeypatch.setattr(rate_limits.hmi_config, "user_rate_limits", config, raising=False)
    monkeypatch.setattr(rate_limits, "_get_client", lambda: FakeRateLimitRedis(count=count))
    if expect_429:
        with pytest.raises(HTTPException) as exc_info:
            await rate_limits.enforce_user_rate_limit(route_class, USER)
        assert exc_info.value.status_code == 429
        assert exc_info.value.headers is not None
        assert "Retry-After" in exc_info.value.headers
    else:
        await rate_limits.enforce_user_rate_limit(route_class, USER)


@pytest.mark.parametrize(
    "redis",
    [
        pytest.param(FakeRateLimitRedis(error=ConnectionError("redis down")), id="redis-error"),
        pytest.param(FakeRateLimitRedis(count=100, delay=1.0), id="redis-slow"),
    ],
)
@pytest.mark.asyncio
async def test_fail_open(monkeypatch, redis):
    monkeypatch.setattr(rate_limits.hmi_config, "user_rate_limits", LIMITS, raising=False)
    monkeypatch.setattr(rate_limits, "_get_client", lambda: redis)
    # Over-limit counts must still be allowed when Redis errors or times out.
    await rate_limits.enforce_user_rate_limit("get_async_task", USER)
