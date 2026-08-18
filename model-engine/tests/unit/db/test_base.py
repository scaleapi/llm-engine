import asyncio
import threading
import time
from types import SimpleNamespace
from unittest import mock
from unittest.mock import AsyncMock

import pytest
from model_engine_server.db.base import DBConnection, DBManager
from sqlalchemy.pool import NullPool, QueuePool

INFRA_CONFIG = SimpleNamespace(
    db_engine_disconnect_strategy="pessimistic",
    db_engine_pool_size=2,
    db_engine_max_overflow=1,
    db_engine_echo=False,
    db_engine_echo_pool=False,
)

# Engines are constructed but never connected, so a dummy URL is safe.
DUMMY_URL = "postgresql://user:pass@localhost:1/db"


def make_manager(build_delay=0.0, expiry=None):
    """`expiry` may be a single value for every build, or a list consumed per build."""
    calls = []
    expiries = list(expiry) if isinstance(expiry, list) else None

    def fake_get_engine_url(self, read_only, sync):
        calls.append((read_only, sync))
        if build_delay:
            time.sleep(build_delay)
        url = DUMMY_URL if sync else DUMMY_URL.replace("postgresql://", "postgresql+asyncpg://")
        return DBConnection(url, expiries.pop(0) if expiries is not None else expiry)

    with mock.patch.object(DBManager, "_get_engine_url", fake_get_engine_url):
        manager = DBManager(INFRA_CONFIG)
    manager._get_engine_url = fake_get_engine_url.__get__(manager)  # keep patch after exit
    return manager, calls


@pytest.mark.asyncio
async def test_engines_built_lazily_per_kind():
    manager, calls = make_manager()
    assert calls == []

    await manager.get_session_async_ro()
    assert calls == [(True, False)]

    # Same kind again: no rebuild. Different kind: one more build only.
    await manager.get_session_async_ro()
    assert len(calls) == 1
    manager.get_session_sync()
    assert calls == [(True, False), (False, True)]


def test_credential_expiry_rebuilds_in_use_kinds():
    manager, calls = make_manager(expiry=int(time.time()) + 10_000)
    session = manager.get_session_sync()
    assert manager.get_session_sync() is session

    manager.credential_expiration_timestamp = time.time() - 1
    rebuilt = manager.get_session_sync()
    assert rebuilt is not session
    assert len(calls) == 2


def test_credential_expiry_awaits_async_engine_disposal_before_loop_closes():
    manager, _ = make_manager(expiry=int(time.time()) + 10_000)
    asyncio.run(manager.get_session_async())
    disposal_completed = False

    async def dispose_engine():
        nonlocal disposal_completed
        await asyncio.sleep(0)
        disposal_completed = True

    with mock.patch(
        "model_engine_server.db.base.AsyncEngine.dispose",
        new_callable=AsyncMock,
        side_effect=dispose_engine,
    ) as dispose:
        manager.credential_expiration_timestamp = time.time() - 1
        asyncio.run(manager.get_session_async())

    dispose.assert_awaited_once_with()
    assert disposal_completed


@pytest.mark.parametrize("kind_getter", ["get_session_sync", "get_session_sync_ro"])
def test_concurrent_first_use_builds_once(kind_getter):
    manager, calls = make_manager(build_delay=0.05)
    results = []

    def fetch():
        results.append(getattr(manager, kind_getter)())

    threads = [threading.Thread(target=fetch) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(calls) == 1
    assert all(r is results[0] for r in results)


ALL_KINDS = [
    pytest.param("sync", (False, True), id="sync"),
    pytest.param("sync_ro", (True, True), id="sync_ro"),
    pytest.param("async", (False, False), id="async"),
    pytest.param("async_ro", (True, False), id="async_ro"),
    pytest.param("async_null_pool", (False, False), id="async_null_pool"),
]

KIND_GETTERS = {
    "sync": "get_session_sync",
    "sync_ro": "get_session_sync_ro",
    "async": "get_session_async",
    "async_ro": "get_session_async_ro",
    "async_null_pool": "get_session_async_null_pool",
}


def fetch_session(manager, kind):
    result = getattr(manager, KIND_GETTERS[kind])()
    return asyncio.run(result) if asyncio.iscoroutine(result) else result


def fetch_db_session(manager, kind):
    """Internal per-kind session dataclass (exposes .engine, unlike the getters)."""
    if kind.startswith("async"):
        return asyncio.run(manager._get_async_session(kind))
    return manager._get_session(kind)


@pytest.mark.parametrize("kind, expected_url_args", ALL_KINDS)
def test_each_kind_builds_lazily_with_correct_url_args(kind, expected_url_args):
    manager, calls = make_manager()
    assert calls == []
    session = fetch_session(manager, kind)
    assert calls == [expected_url_args]
    # Repeat access is cached: no second build.
    assert fetch_session(manager, kind) is session
    assert len(calls) == 1


@pytest.mark.parametrize("kind, _expected_url_args", ALL_KINDS)
def test_pool_class_and_sizing_per_kind(kind, _expected_url_args):
    manager, _ = make_manager()
    engine = fetch_db_session(manager, kind).engine
    pool = engine.pool if kind.startswith("sync") else engine.sync_engine.pool
    if kind == "async_null_pool":
        assert isinstance(pool, NullPool)
    else:
        assert isinstance(pool, QueuePool)
        assert pool.size() == INFRA_CONFIG.db_engine_pool_size
        # QueuePool has no public accessor for the configured overflow.
        assert pool._max_overflow == INFRA_CONFIG.db_engine_max_overflow


@pytest.mark.asyncio
async def test_mixed_loop_and_thread_cold_race_builds_once_per_kind():
    manager, calls = make_manager(build_delay=0.05)
    sync_results = []
    threads = [
        threading.Thread(target=lambda: sync_results.append(manager.get_session_sync()))
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    async_sessions = await asyncio.wait_for(
        asyncio.gather(manager.get_session_async(), manager.get_session_async()), timeout=10
    )
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert sorted(calls) == [(False, False), (False, True)]
    assert all(r is sync_results[0] for r in sync_results)
    assert async_sessions[0] is async_sessions[1]


def test_credential_expiration_timestamp_lifecycle():
    first, later, reseed = (int(time.time()) + offset for offset in (10_000, 20_000, 30_000))
    manager, _ = make_manager(expiry=[first, later, reseed])

    manager.get_session_sync()
    assert manager.credential_expiration_timestamp == first
    # The first built engine's expiry stays authoritative for later builds.
    manager.get_session_sync_ro()
    assert manager.credential_expiration_timestamp == first

    manager.credential_expiration_timestamp = time.time() - 1
    assert manager._take_expired_sessions()
    assert manager.credential_expiration_timestamp is None

    manager.get_session_sync()
    assert manager.credential_expiration_timestamp == reseed


@pytest.mark.parametrize(
    "call_context",
    [
        pytest.param("plain-thread", id="no-running-loop-uses-asyncio-run"),
        pytest.param("inside-loop", id="running-loop-disposes-sync-engine"),
    ],
)
def test_sync_getter_disposes_expired_async_engine(call_context):
    # DB-7: the sync getter must dispose an expired ASYNC engine without raising,
    # both from a plain thread and from a thread whose event loop is running.
    manager, _ = make_manager(expiry=int(time.time()) + 10_000)
    asyncio.run(manager.get_session_async())
    manager.credential_expiration_timestamp = time.time() - 1

    if call_context == "plain-thread":
        session = manager.get_session_sync()
    else:

        async def call_sync_getter_in_loop():
            return manager.get_session_sync()

        session = asyncio.run(call_sync_getter_in_loop())

    assert session is not None
    # The expired async kind was evicted; only the freshly built sync kind remains.
    assert list(manager._sessions) == ["sync"]
