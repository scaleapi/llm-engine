import pytest
from model_engine_server.common.config import (
    HostedModelInferenceServiceConfig,
    _apply_redis_auth_token,
)
from model_engine_server.core.celery.app import (
    build_redis_url,
    get_redis_instance,
    redis_tls_enabled,
)
from redis.connection import SSLConnection

TOKEN_WITH_RESERVED_CHARS = "p@ss/w#rd"
ENCODED_RESERVED_CHARS = "p%40ss%2Fw%23rd"


@pytest.fixture
def redis_env(monkeypatch):
    def _set(auth_token=None, enable_tls=None):
        for name, value in (
            ("REDIS_AUTH_TOKEN", auth_token),
            ("REDIS_ENABLE_TLS", enable_tls),
        ):
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)

    return _set


def test_tls_inferred_from_credential_when_unset(redis_env):
    redis_env(auth_token="tok")
    assert redis_tls_enabled() is True
    redis_env()
    assert redis_tls_enabled() is False


@pytest.mark.parametrize("value,expected", [("true", True), ("True", True), ("false", False)])
def test_explicit_tls_setting_overrides_credential(redis_env, value, expected):
    redis_env(auth_token="tok", enable_tls=value)
    assert redis_tls_enabled() is expected


def test_credential_without_tls_yields_plaintext_url(redis_env):
    redis_env(auth_token=TOKEN_WITH_RESERVED_CHARS, enable_tls="false")
    assert build_redis_url("redis", 6379, 2) == f"redis://:{ENCODED_RESERVED_CHARS}@redis:6379/2"


def test_tls_without_credential_yields_rediss_url(redis_env):
    redis_env(enable_tls="true")
    assert build_redis_url("host", 6379, 0) == "rediss://host:6379/0?ssl_cert_reqs=none"


def test_credential_defaults_to_tls_url(redis_env):
    redis_env(auth_token="tok")
    assert build_redis_url("host", 6379, 1) == "rediss://:tok@host:6379/1?ssl_cert_reqs=none"


def test_no_credential_no_tls_yields_plain_url(redis_env):
    redis_env()
    assert build_redis_url("host", 6379, 0) == "redis://host:6379/0"


def test_auth_token_injected_into_credential_free_url(redis_env):
    redis_env(auth_token=TOKEN_WITH_RESERVED_CHARS)
    assert (
        _apply_redis_auth_token("redis://redis:6379/0")
        == f"redis://:{ENCODED_RESERVED_CHARS}@redis:6379/0"
    )


def test_existing_credential_is_preserved(redis_env):
    redis_env(auth_token="tok")
    assert _apply_redis_auth_token("redis://:mine@h:6379/0") == "redis://:mine@h:6379/0"


def test_url_unchanged_without_auth_token(redis_env):
    redis_env()
    assert _apply_redis_auth_token("redis://redis:6379/0") == "redis://redis:6379/0"


def _host_port(url: str) -> str:
    """Evaluate the cache_redis_host_port property against a stubbed URL."""

    class _Stub:
        cache_redis_url = url

    return HostedModelInferenceServiceConfig.cache_redis_host_port.fget(_Stub())


@pytest.mark.parametrize(
    "url,expected",
    [
        ("redis://redis.url:6379/0", "redis.url:6379"),
        ("rediss://redis.url:6379/0", "redis.url:6379"),
        ("redis://:p%40ss@redis.url:6379/2", "redis.url:6379"),
        ("rediss://:tok@redis.url:6379/0", "redis.url:6379"),
        ("rediss://user:tok@cache.redis.azure.com", "cache.redis.azure.com"),
        ("redis://redis:6379", "redis:6379"),
    ],
)
def test_host_port_strips_credentials_for_every_scheme(url, expected):
    assert _host_port(url) == expected


def test_host_port_never_leaks_password_into_scaler_address(redis_env):
    redis_env(auth_token=TOKEN_WITH_RESERVED_CHARS)
    url = _apply_redis_auth_token("redis://redis:6379/0")
    host_port = _host_port(url)
    assert "@" not in host_port
    assert ENCODED_RESERVED_CHARS not in host_port
    assert host_port == "redis:6379"


@pytest.fixture
def fixed_redis_host(monkeypatch):
    monkeypatch.setattr(
        "model_engine_server.core.celery.app.get_redis_host_port", lambda: ("h", 6379)
    )


def _uses_tls(client) -> bool:
    return client.connection_pool.connection_class is SSLConnection


def test_instance_credential_without_tls(redis_env, fixed_redis_host):
    redis_env(auth_token="tok", enable_tls="false")
    client = get_redis_instance(1)
    assert client.connection_pool.connection_kwargs["password"] == "tok"
    assert not _uses_tls(client)


def test_instance_tls_without_credential(redis_env, fixed_redis_host):
    redis_env(enable_tls="true")
    client = get_redis_instance()
    assert client.connection_pool.connection_kwargs.get("password") is None
    assert _uses_tls(client)


def test_instance_credential_implies_tls_when_unset(redis_env, fixed_redis_host):
    redis_env(auth_token="tok")
    client = get_redis_instance()
    assert client.connection_pool.connection_kwargs["password"] == "tok"
    assert _uses_tls(client)


def test_instance_plain_when_neither_set(redis_env, fixed_redis_host):
    redis_env()
    client = get_redis_instance()
    assert client.connection_pool.connection_kwargs.get("password") is None
    assert not _uses_tls(client)
