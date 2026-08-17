import asyncio
import threading
import time
from types import SimpleNamespace
from unittest import mock
from unittest.mock import AsyncMock

import pytest
from model_engine_server.db.base import DBConnection, DBManager

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
    calls = []

    def fake_get_engine_url(self, read_only, sync):
        calls.append((read_only, sync))
        if build_delay:
            time.sleep(build_delay)
        url = DUMMY_URL if sync else DUMMY_URL.replace("postgresql://", "postgresql+asyncpg://")
        return DBConnection(url, expiry)

    with mock.patch.object(DBManager, "_get_engine_url", fake_get_engine_url):
        manager = DBManager(INFRA_CONFIG)
    manager._get_engine_url = fake_get_engine_url.__get__(manager)  # keep patch after exit
    return manager, calls


def test_engines_built_lazily_per_kind():
    manager, calls = make_manager()
    assert calls == []

    manager.get_session_async_ro()
    assert calls == [(True, False)]

    # Same kind again: no rebuild. Different kind: one more build only.
    manager.get_session_async_ro()
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dispose_in_thread",
    [
        pytest.param(False, id="event-loop"),
        pytest.param(True, id="thread"),
    ],
)
async def test_credential_expiry_awaits_async_engine_disposal(dispose_in_thread):
    manager, _ = make_manager(expiry=int(time.time()) + 10_000)
    manager.get_session_async()

    with mock.patch(
        "model_engine_server.db.base.AsyncEngine.dispose", new_callable=AsyncMock
    ) as dispose:
        manager.credential_expiration_timestamp = time.time() - 1
        if dispose_in_thread:
            await asyncio.to_thread(manager.get_session_async)
        else:
            manager.get_session_async()
            await asyncio.sleep(0)

    dispose.assert_awaited_once_with()


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
