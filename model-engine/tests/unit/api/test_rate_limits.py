import asyncio

import pytest
from fastapi import HTTPException
from model_engine_server.api import rate_limits
from model_engine_server.core.auth.authentication_repository import User


class FakePipeline:
    def __init__(self, count, error=None, delay=0.0):
        self.count = count
        self.error = error
        self.delay = delay

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def incr(self, key):
        pass

    def expire(self, key, seconds):
        pass

    async def execute(self):
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return [self.count, True]


class FakeRedis:
    def __init__(self, count=1, error=None, delay=0.0):
        self._pipeline = FakePipeline(count, error=error, delay=delay)

    def pipeline(self, transaction=False):
        return self._pipeline


USER = User(user_id="test-user", team_id="test-team", is_privileged_user=False)
LIMITS = {"enforce": True, "routes": {"get_async_task": 5, "default": 2}}
LIMITS_LOG_ONLY = {"enforce": False, "routes": {"get_async_task": 5}}
LIMITS_NO_DEFAULT = {"enforce": True, "routes": {"get_async_task": 5, "default": None}}


@pytest.mark.parametrize(
    "config,method,path,count,expect_429",
    [
        pytest.param(None, "GET", "/v1/async-tasks/tid", 100, False, id="disabled-no-config"),
        pytest.param(LIMITS, "GET", "/v1/async-tasks/tid", 5, False, id="at-limit-allowed"),
        pytest.param(LIMITS, "GET", "/v1/async-tasks/tid", 6, True, id="over-limit-rejected"),
        pytest.param(LIMITS, "POST", "/v1/async-tasks", 3, True, id="default-limit-applies"),
        pytest.param(
            LIMITS_LOG_ONLY, "GET", "/v1/async-tasks/tid", 100, False, id="log-only-allows"
        ),
        pytest.param(
            LIMITS_NO_DEFAULT,
            "POST",
            "/v1/model-endpoints",
            100,
            False,
            id="null-default-unlimited",
        ),
    ],
)
@pytest.mark.asyncio
async def test_enforce_user_rate_limit(monkeypatch, config, method, path, count, expect_429):
    monkeypatch.setattr(rate_limits.hmi_config, "user_rate_limits", config, raising=False)
    redis = FakeRedis(count=count)
    if expect_429:
        with pytest.raises(HTTPException) as exc_info:
            await rate_limits.enforce_user_rate_limit(method, path, USER, redis)
        assert exc_info.value.status_code == 429
        assert exc_info.value.headers is not None
        assert "Retry-After" in exc_info.value.headers
    else:
        await rate_limits.enforce_user_rate_limit(method, path, USER, redis)


@pytest.mark.parametrize(
    "redis",
    [
        pytest.param(FakeRedis(error=ConnectionError("redis down")), id="redis-error"),
        pytest.param(FakeRedis(count=100, delay=1.0), id="redis-slow"),
    ],
)
@pytest.mark.asyncio
async def test_fail_open(monkeypatch, redis):
    monkeypatch.setattr(rate_limits.hmi_config, "user_rate_limits", LIMITS, raising=False)
    # Over-limit counts must still be allowed when Redis errors or times out.
    await rate_limits.enforce_user_rate_limit("GET", "/v1/async-tasks/tid", USER, redis)


@pytest.mark.parametrize(
    "method,path,expected",
    [
        pytest.param("GET", "/v1/async-tasks/some-task-id", "get_async_task", id="poll"),
        pytest.param("POST", "/v1/async-tasks", "post_async_tasks", id="submit"),
        pytest.param(
            "POST", "/v1/async-tasks?model_endpoint_id=x", "post_async_tasks", id="submit-query"
        ),
        pytest.param("GET", "/v1/model-endpoints", "default", id="control-plane"),
        pytest.param("GET", "/v1/async-tasks", "default", id="poll-collection-is-default"),
    ],
)
def test_classify_route(method, path, expected):
    assert rate_limits._classify_route(method, path) == expected
