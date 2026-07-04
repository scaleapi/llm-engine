"""Unit tests for start_celery_service worker wiring (MLI-7328).

CELERY_WORKER_MAX_TASKS_PER_CHILD is an opt-in, prefork-only knob: when set it recycles each
worker child after N tasks (defense-in-depth against per-task memory residue such as glibc arena
retention). It is off by default so prefork behaviour is unchanged, and ignored under gevent
(which has no per-child recycling). app.Worker is mocked so .start() does not run a real worker.
"""

from unittest.mock import MagicMock

from model_engine_server.inference.forwarding import celery_forwarder


def _worker_kwargs(monkeypatch, pool, env_value):
    monkeypatch.setattr(celery_forwarder, "CELERY_WORKER_POOL", pool)
    if env_value is None:
        monkeypatch.delenv("CELERY_WORKER_MAX_TASKS_PER_CHILD", raising=False)
    else:
        monkeypatch.setenv("CELERY_WORKER_MAX_TASKS_PER_CHILD", env_value)
    app = MagicMock()
    celery_forwarder.start_celery_service(app, "q", 4)
    app.Worker.assert_called_once()
    app.Worker.return_value.start.assert_called_once()
    return app.Worker.call_args.kwargs


def test_max_tasks_per_child_unset_by_default(monkeypatch):
    # Prefork unchanged when the env is not set.
    kwargs = _worker_kwargs(monkeypatch, "prefork", None)
    assert "max_tasks_per_child" not in kwargs


def test_max_tasks_per_child_applied_under_prefork(monkeypatch):
    kwargs = _worker_kwargs(monkeypatch, "prefork", "500")
    assert kwargs["max_tasks_per_child"] == 500


def test_max_tasks_per_child_ignored_under_gevent(monkeypatch):
    # gevent runs one process with no per-child recycling, so the knob is a no-op there.
    kwargs = _worker_kwargs(monkeypatch, "gevent", "500")
    assert kwargs["pool"] == "gevent"
    assert "max_tasks_per_child" not in kwargs
