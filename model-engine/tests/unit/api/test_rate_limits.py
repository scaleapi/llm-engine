import asyncio
import dataclasses
import logging
import time

import pytest
import redis.asyncio as aioredis
from fastapi import Depends, HTTPException
from model_engine_server.api import rate_limits
from model_engine_server.api.app import app
from model_engine_server.api.dependencies import basic_auth, oauth2_scheme, verify_authentication
from model_engine_server.common.config import HostedModelInferenceServiceConfig
from model_engine_server.core.auth.authentication_repository import User
from tests.unit.api.conftest import fake_verify_authentication, get_test_auth_repository


class FakeRateLimitRedis:
    """Counter-only fake for the limiter's eval call."""

    def __init__(self, count=1, error=None, delay=0.0, sequence=None):
        self.count = count
        self.error = error
        self.delay = delay
        self.sequence = list(sequence) if sequence is not None else None
        self.calls = 0
        self.keys = []

    async def eval(self, script, numkeys, *keys):
        self.calls += 1
        self.keys.extend(keys)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.sequence is not None:
            item = self.sequence.pop(0)
            if isinstance(item, Exception):
                raise item
            return item if isinstance(item, list) else [item] * numkeys
        if self.error:
            raise self.error
        return [self.count] * numkeys


class FakeClock:
    """Stands in for the `time` module inside rate_limits (monotonic + wall clock)."""

    def __init__(self):
        self.now = time.monotonic()
        self.wall = time.time()

    def monotonic(self):
        return self.now

    def time(self):
        return self.wall


def _install_fake_clock(monkeypatch) -> FakeClock:
    clock = FakeClock()
    monkeypatch.setattr(rate_limits, "time", clock)
    return clock


@pytest.fixture(autouse=True)
def _reset_limiter_state(monkeypatch):
    monkeypatch.setattr(rate_limits, "_consecutive_failures", 0)
    monkeypatch.setattr(rate_limits, "_breaker_open_until", 0.0)
    monkeypatch.setattr(rate_limits, "_last_log_times", {})
    monkeypatch.setattr(rate_limits, "_client", None)
    monkeypatch.setattr(rate_limits, "_client_pool", None)


USER = User(user_id="test-user", team_id="test-team", is_privileged_user=False)
LIMITS = {"enforce": True, "routes": {"get_async_task": 5}}
LIMITS_LOG_ONLY = {"enforce": False, "routes": {"get_async_task": 5}}
LIMITS_STRING_VALUE = {"enforce": True, "routes": {"get_async_task": "5"}}


@pytest.mark.parametrize(
    "config,route_class,count,expect_429",
    [
        pytest.param(None, "get_async_task", 100, False, id="disabled-no-config"),
        pytest.param(LIMITS, "get_async_task", 5, False, id="at-limit-allowed"),
        pytest.param(LIMITS, "get_async_task", 6, True, id="over-limit-rejected"),
        pytest.param(LIMITS, "post_async_tasks", 100, False, id="unconfigured-route-unlimited"),
        pytest.param(LIMITS_LOG_ONLY, "get_async_task", 100, False, id="log-only-allows"),
        pytest.param(LIMITS_STRING_VALUE, "get_async_task", 6, True, id="string-limit-coerced"),
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


@pytest.mark.asyncio
async def test_circuit_breaker_stops_touching_redis(monkeypatch):
    monkeypatch.setattr(rate_limits.hmi_config, "user_rate_limits", LIMITS, raising=False)
    redis = FakeRateLimitRedis(error=ConnectionError("redis down"))
    monkeypatch.setattr(rate_limits, "_get_client", lambda: redis)
    for _ in range(rate_limits._BREAKER_FAILURE_THRESHOLD + 5):
        await rate_limits.enforce_user_rate_limit("get_async_task", USER)
    # After the threshold trips, the cooldown window must not touch Redis at all.
    assert redis.calls == rate_limits._BREAKER_FAILURE_THRESHOLD


@pytest.mark.asyncio
async def test_breaker_cooldown_expiry_resumes_checks_and_enforcement(monkeypatch):
    monkeypatch.setattr(rate_limits.hmi_config, "user_rate_limits", LIMITS, raising=False)
    clock = _install_fake_clock(monkeypatch)
    failing = FakeRateLimitRedis(error=ConnectionError("redis down"))
    monkeypatch.setattr(rate_limits, "_get_client", lambda: failing)
    for _ in range(rate_limits._BREAKER_FAILURE_THRESHOLD):
        await rate_limits.enforce_user_rate_limit("get_async_task", USER)

    healthy = FakeRateLimitRedis(count=6)
    monkeypatch.setattr(rate_limits, "_get_client", lambda: healthy)
    # Inside the cooldown window Redis stays untouched even though it recovered.
    await rate_limits.enforce_user_rate_limit("get_async_task", USER)
    assert healthy.calls == 0

    clock.now += rate_limits._BREAKER_COOLDOWN_SECONDS + 0.1
    with pytest.raises(HTTPException) as exc_info:
        await rate_limits.enforce_user_rate_limit("get_async_task", USER)
    assert exc_info.value.status_code == 429
    assert healthy.calls == 1


@pytest.mark.asyncio
async def test_success_resets_consecutive_failure_counter(monkeypatch):
    monkeypatch.setattr(rate_limits.hmi_config, "user_rate_limits", LIMITS, raising=False)
    error = ConnectionError("redis down")
    # 4 failures + 1 success + 4 failures: one short of the threshold on each side.
    redis = FakeRateLimitRedis(sequence=[error] * 4 + [1] + [error] * 4)
    monkeypatch.setattr(rate_limits, "_get_client", lambda: redis)
    for _ in range(9):
        await rate_limits.enforce_user_rate_limit("get_async_task", USER)
    assert rate_limits._breaker_open_until == 0.0
    assert redis.calls == 9


@pytest.mark.asyncio
async def test_log_sampling_suppresses_per_key_and_reemits(monkeypatch, caplog):
    monkeypatch.setattr(rate_limits.hmi_config, "user_rate_limits", LIMITS_LOG_ONLY, raising=False)
    # The limiter's logger does not propagate to root by default, which hides it from caplog.
    monkeypatch.setattr(rate_limits.logger, "propagate", True)
    caplog.set_level(logging.WARNING)
    clock = _install_fake_clock(monkeypatch)
    error = ConnectionError("redis down")

    def warning_count():
        return sum(
            1
            for record in caplog.records
            if record.name == rate_limits.logger.name and record.levelno == logging.WARNING
        )

    # (redis behavior, expected cumulative warning count); None advances the clock
    # past the sampling window. The fail-open and over-limit log keys are independent.
    steps = [
        (error, 1),  # first fail-open logged
        (error, 1),  # suppressed within 60s
        (100, 2),  # over-limit key logs despite fail-open suppression window
        (100, 2),  # suppressed within 60s
        (None, 2),
        (error, 3),  # re-emitted after window
        (100, 4),  # re-emitted after window
    ]
    for behavior, expected in steps:
        if behavior is None:
            clock.now += rate_limits._LOG_INTERVAL_SECONDS + 0.1
        else:
            fake = FakeRateLimitRedis(
                error=behavior if isinstance(behavior, Exception) else None,
                count=behavior if not isinstance(behavior, Exception) else 1,
            )
            monkeypatch.setattr(rate_limits, "_get_client", lambda fake=fake: fake)
            await rate_limits.enforce_user_rate_limit("get_async_task", USER)
        assert warning_count() == expected


def test_last_log_times_cleared_past_1000_keys(monkeypatch):
    now = time.monotonic()
    monkeypatch.setattr(rate_limits, "_last_log_times", {f"key-{i}": now for i in range(1001)})
    assert rate_limits._should_log("new-key")
    assert set(rate_limits._last_log_times) == {"new-key"}


def test_get_client_cached_per_pool(monkeypatch):
    pool_a = aioredis.ConnectionPool.from_url("redis://localhost:6379/0")
    pool_b = aioredis.ConnectionPool.from_url("redis://localhost:6379/0")
    pools = iter([pool_a, pool_a, pool_b])
    monkeypatch.setattr(rate_limits, "get_or_create_aioredis_pool", lambda: next(pools))

    client_a = rate_limits._get_client()
    assert rate_limits._get_client() is client_a
    # A rebuilt pool (credential rotation) must produce a new client bound to it.
    client_b = rate_limits._get_client()
    assert client_b is not client_a
    assert client_b.connection_pool is pool_b


@pytest.mark.asyncio
async def test_window_rollover_uses_distinct_key_per_second(monkeypatch):
    monkeypatch.setattr(rate_limits.hmi_config, "user_rate_limits", LIMITS, raising=False)
    clock = _install_fake_clock(monkeypatch)
    redis = FakeRateLimitRedis(count=1)
    monkeypatch.setattr(rate_limits, "_get_client", lambda: redis)

    await rate_limits.enforce_user_rate_limit("get_async_task", USER)
    clock.wall += 1
    await rate_limits.enforce_user_rate_limit("get_async_task", USER)

    expected_seconds = [int(clock.wall) - 1, int(clock.wall)]
    assert redis.keys == [
        f"user-rate-limit:get_async_task:{USER.user_id}:{second}" for second in expected_seconds
    ]


def test_rate_limit_dependency_composition_over_http(monkeypatch, simple_client, test_api_key):
    monkeypatch.setattr(rate_limits.hmi_config, "user_rate_limits", LIMITS, raising=False)
    monkeypatch.setattr(rate_limits, "_get_client", lambda: FakeRateLimitRedis(count=100))
    auth_calls = []

    def counting_auth(
        credentials=Depends(basic_auth),
        tokens=Depends(oauth2_scheme),
        auth_repo=Depends(get_test_auth_repository),
    ):
        auth_calls.append(1)
        return fake_verify_authentication(credentials, tokens, auth_repo)

    app.dependency_overrides[verify_authentication] = counting_auth

    response = simple_client.get("/v1/async-tasks/test_task_id", auth=(test_api_key, ""))
    assert response.status_code == 429
    assert response.headers.get("Retry-After") == "1"
    # FastAPI caches verify_authentication per request across the auth + limiter deps.
    assert len(auth_calls) == 1
    # Healthchecks carry no rate-limit dependency and must be unaffected.
    assert simple_client.get("/readyz").status_code == 200


def _minimal_service_config_json():
    return {
        field.name: "x"
        for field in dataclasses.fields(HostedModelInferenceServiceConfig)
        if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING
    }


@pytest.mark.parametrize(
    "user_rate_limits",
    [
        pytest.param(None, id="absent-defaults-none"),
        pytest.param({"enforce": False, "routes": {"get_async_task": 200}}, id="round-trips"),
    ],
)
def test_service_config_round_trips_user_rate_limits(user_rate_limits):
    raw = _minimal_service_config_json()
    if user_rate_limits is not None:
        raw["user_rate_limits"] = user_rate_limits
    config = HostedModelInferenceServiceConfig.from_json(raw)
    assert config.user_rate_limits == user_rate_limits


@pytest.mark.parametrize(
    "config,count,error,expected_outcome",
    [
        pytest.param(LIMITS, 3, None, "allowed", id="allowed"),
        pytest.param(LIMITS, 6, None, "throttled", id="throttled"),
        pytest.param(LIMITS_LOG_ONLY, 6, None, "would_throttle", id="would-throttle"),
        pytest.param(LIMITS, 1, ConnectionError("down"), "fail_open", id="fail-open"),
    ],
)
@pytest.mark.asyncio
async def test_decision_metric_emitted(monkeypatch, config, count, error, expected_outcome):
    monkeypatch.setattr(rate_limits.hmi_config, "user_rate_limits", config, raising=False)
    monkeypatch.setattr(
        rate_limits, "_get_client", lambda: FakeRateLimitRedis(count=count, error=error)
    )
    emitted = []
    monkeypatch.setattr(
        rate_limits.statsd, "increment", lambda name, tags: emitted.append((name, tags))
    )
    try:
        await rate_limits.enforce_user_rate_limit("get_async_task", USER)
    except HTTPException:
        pass
    assert len(emitted) == 1
    name, tags = emitted[0]
    assert name == "model_engine.user_rate_limit.decision"
    assert f"outcome:{expected_outcome}" in tags
    assert "user_id:test-user" in tags
    assert "route_class:get_async_task" in tags


@pytest.mark.asyncio
async def test_decision_metric_breaker_open(monkeypatch):
    monkeypatch.setattr(rate_limits.hmi_config, "user_rate_limits", LIMITS, raising=False)
    monkeypatch.setattr(
        rate_limits, "_get_client", lambda: FakeRateLimitRedis(error=ConnectionError("down"))
    )
    emitted = []
    monkeypatch.setattr(rate_limits.statsd, "increment", lambda name, tags: emitted.append(tags))
    for _ in range(rate_limits._BREAKER_FAILURE_THRESHOLD + 1):
        await rate_limits.enforce_user_rate_limit("get_async_task", USER)
    assert any("outcome:breaker_open" in tags for tags in emitted[-1:])


@pytest.mark.parametrize(
    "scope,expected_keys",
    [
        pytest.param("end_abc123", 2, id="scoped-counts-aggregate-and-scope"),
        pytest.param(None, 1, id="no-scope-single-key"),
    ],
)
@pytest.mark.asyncio
async def test_scope_folds_into_bucket_key(monkeypatch, scope, expected_keys):
    monkeypatch.setattr(rate_limits.hmi_config, "user_rate_limits", LIMITS, raising=False)
    redis = FakeRateLimitRedis(count=1)
    monkeypatch.setattr(rate_limits, "_get_client", lambda: redis)
    await rate_limits.enforce_user_rate_limit("get_async_task", USER, scope)
    assert redis.calls == 1  # one round trip regardless of scope
    assert len(redis.keys) == expected_keys
    assert redis.keys[0].count(":") == 3  # aggregate key: route:user:second
    if scope is not None:
        assert f":test-user:{scope}:" in redis.keys[1]


@pytest.mark.asyncio
async def test_scopes_use_independent_buckets(monkeypatch):
    """Two scopes under one user must hit distinct Redis keys in the same second."""
    monkeypatch.setattr(rate_limits.hmi_config, "user_rate_limits", LIMITS, raising=False)
    redis = FakeRateLimitRedis(count=1)
    monkeypatch.setattr(rate_limits, "_get_client", lambda: redis)
    await rate_limits.enforce_user_rate_limit("get_async_task", USER, "end_a")
    await rate_limits.enforce_user_rate_limit("get_async_task", USER, "end_b")
    scoped_keys = {key for key in redis.keys if key.count(":") == 4}
    assert len(scoped_keys) == 2


@pytest.mark.parametrize(
    "scope,counts,expect_429,expect_tag",
    [
        # aggregate fine (1), scoped bucket over (6 > 5): the scope earns the tag
        pytest.param("end_abc123", [[1, 6]], True, True, id="scope-tagged-on-scoped-throttle"),
        pytest.param("end_abc123", [[1, 1]], False, False, id="scope-untagged-on-allowed"),
        pytest.param(None, [[100]], True, False, id="no-scope-no-tag"),
    ],
)
@pytest.mark.asyncio
async def test_decision_metric_scope_tag(monkeypatch, scope, counts, expect_429, expect_tag):
    monkeypatch.setattr(rate_limits.hmi_config, "user_rate_limits", LIMITS, raising=False)
    monkeypatch.setattr(rate_limits, "_get_client", lambda: FakeRateLimitRedis(sequence=counts))
    emitted = []
    monkeypatch.setattr(rate_limits.statsd, "increment", lambda name, tags: emitted.append(tags))
    if expect_429:
        with pytest.raises(HTTPException):
            await rate_limits.enforce_user_rate_limit("get_async_task", USER, scope)
    else:
        await rate_limits.enforce_user_rate_limit("get_async_task", USER, scope)
    assert len(emitted) == 1
    assert (f"scope:{scope}" in emitted[0]) == expect_tag


@pytest.mark.asyncio
async def test_dependency_extracts_scope_query_param(monkeypatch):
    """The route dependency reads scope_query_param off the request and keys on it."""
    monkeypatch.setattr(
        rate_limits.hmi_config,
        "user_rate_limits",
        {"enforce": True, "routes": {"post_async_tasks": 5}},
        raising=False,
    )
    redis = FakeRateLimitRedis(count=1)
    monkeypatch.setattr(rate_limits, "_get_client", lambda: redis)

    class FakeRequest:
        query_params = {"model_endpoint_id": "end_xyz789"}

    dependency = rate_limits.user_rate_limit(
        "post_async_tasks", scope_query_param="model_endpoint_id"
    )
    # Call the dependency directly; auth is normally injected by FastAPI.
    await dependency(FakeRequest(), USER)
    assert any(":test-user:end_xyz789:" in key for key in redis.keys)


@pytest.mark.asyncio
async def test_aggregate_ceiling_bounds_rotating_scopes(monkeypatch):
    """Rotating scope values must not bypass the per-user aggregate ceiling."""
    monkeypatch.setattr(
        rate_limits.hmi_config,
        "user_rate_limits",
        {
            "enforce": True,
            "routes": {"post_async_tasks": 100},
            "aggregate_routes": {"post_async_tasks": 5},
        },
        raising=False,
    )
    # Aggregate (unscoped) bucket is over its ceiling; the scoped bucket is fresh.
    redis = FakeRateLimitRedis(sequence=[[6, 1]])
    monkeypatch.setattr(rate_limits, "_get_client", lambda: redis)
    with pytest.raises(HTTPException) as exc_info:
        await rate_limits.enforce_user_rate_limit("post_async_tasks", USER, "end_rotating_1")
    assert exc_info.value.status_code == 429
    # One round trip counted both keys: aggregate (no scope segment) first.
    assert redis.calls == 1
    assert ":test-user:" in redis.keys[0] and "end_rotating_1" not in redis.keys[0]
    assert "end_rotating_1" in redis.keys[1]


@pytest.mark.asyncio
async def test_aggregate_rejection_not_tagged_with_scope(monkeypatch):
    """Aggregate-ceiling rejections must not carry the rotated scope as a tag."""
    monkeypatch.setattr(
        rate_limits.hmi_config,
        "user_rate_limits",
        {
            "enforce": True,
            "routes": {"post_async_tasks": 100},
            "aggregate_routes": {"post_async_tasks": 5},
        },
        raising=False,
    )
    emitted = []
    monkeypatch.setattr(rate_limits.statsd, "increment", lambda name, tags: emitted.append(tags))
    redis = FakeRateLimitRedis(sequence=[[6, 1]])
    monkeypatch.setattr(rate_limits, "_get_client", lambda: redis)
    with pytest.raises(HTTPException):
        await rate_limits.enforce_user_rate_limit("post_async_tasks", USER, "end_rotated_9")
    assert len(emitted) == 1
    assert not any(tag.startswith("scope:") for tag in emitted[0])


@pytest.mark.asyncio
async def test_default_aggregate_ceiling_without_config(monkeypatch):
    """Scoped routes are aggregate-bounded even when aggregate_routes is absent."""
    monkeypatch.setattr(
        rate_limits.hmi_config,
        "user_rate_limits",
        {"enforce": True, "routes": {"post_async_tasks": 5}},
        raising=False,
    )
    over_default_aggregate = 5 * rate_limits._DEFAULT_AGGREGATE_MULTIPLIER + 1
    redis = FakeRateLimitRedis(sequence=[[over_default_aggregate, 1]])
    monkeypatch.setattr(rate_limits, "_get_client", lambda: redis)
    with pytest.raises(HTTPException) as exc_info:
        await rate_limits.enforce_user_rate_limit("post_async_tasks", USER, "end_fresh_scope")
    assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_scope_tag_omitted_on_allowed(monkeypatch):
    """Caller-supplied scope must not mint metric cardinality on allowed traffic."""
    monkeypatch.setattr(rate_limits.hmi_config, "user_rate_limits", LIMITS, raising=False)
    emitted = []
    monkeypatch.setattr(rate_limits.statsd, "increment", lambda name, tags: emitted.append(tags))
    monkeypatch.setattr(rate_limits, "_get_client", lambda: FakeRateLimitRedis(count=1))
    await rate_limits.enforce_user_rate_limit("get_async_task", USER, "end_abc")
    # aggregate fine, scoped bucket over: the throttle is attributable to the scope
    monkeypatch.setattr(rate_limits, "_get_client", lambda: FakeRateLimitRedis(sequence=[[1, 100]]))
    with pytest.raises(HTTPException):
        await rate_limits.enforce_user_rate_limit("get_async_task", USER, "end_abc")
    allowed_tags, throttled_tags = emitted
    assert not any(t.startswith("scope:") for t in allowed_tags)
    assert "scope:end_abc" in throttled_tags
