"""Unit tests for start_celery_service worker wiring (MLI-7328).

CELERY_WORKER_MAX_TASKS_PER_CHILD is an opt-in, prefork-only knob: when set it recycles each
worker child after N tasks (defense-in-depth against per-task memory residue such as glibc arena
retention). It is off by default so prefork behaviour is unchanged, and ignored under gevent
(which has no per-child recycling). app.Worker is mocked so .start() does not run a real worker.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from celery import Celery
from model_engine_server.common.constants import DEFAULT_CELERY_TASK_NAME, LIRA_CELERY_TASK_NAME
from model_engine_server.inference.forwarding import celery_forwarder


@pytest.mark.parametrize("task_name", [LIRA_CELERY_TASK_NAME, DEFAULT_CELERY_TASK_NAME])
def test_forwarder_tasks_use_configured_wall_clock_limit(monkeypatch, task_name):
    app = Celery("test-forwarder", broker="memory://", backend="cache+memory://")
    monkeypatch.setattr(celery_forwarder, "celery_app", lambda **_kwargs: app)
    monkeypatch.setattr(
        celery_forwarder,
        "infra_config",
        lambda: SimpleNamespace(s3_bucket="test", profile_ml_inference_worker=None),
    )
    monkeypatch.setattr(celery_forwarder, "DatadogInferenceMonitoringMetricsGateway", MagicMock)
    forwarder = MagicMock(timeout_seconds=123.5, post_inference_hooks_handler=None)

    celery_forwarder.create_celery_service(
        forwarder=forwarder,
        task_visibility=celery_forwarder.TaskVisibility.VISIBILITY_24H,
        broker_type="redis",
        backend_protocol="redis",
        queue_name="test-queue",
    )

    # Strictly above the forwarder's own HTTP timeout, so that timeout raises before the kill.
    assert (
        app.tasks[task_name].time_limit == 123.5 + celery_forwarder.CELERY_TIME_LIMIT_GRACE_SECONDS
    )
    assert app.tasks[task_name].time_limit > forwarder.timeout_seconds


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


def _run_entrypoint(monkeypatch, async_config):
    """Drive entrypoint() with everything past forwarder construction stubbed out."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "celery_forwarder",
            "--config",
            "ignored.yaml",
            "--task-visibility",
            "VISIBILITY_24H",
            "--num-workers",
            "1",
            "--queue",
            "test-queue",
        ],
    )
    monkeypatch.setattr(
        celery_forwarder, "load_named_config", lambda *_args, **_kwargs: {"async": async_config}
    )
    load_forwarder = MagicMock()
    monkeypatch.setattr(celery_forwarder, "LoadForwarder", load_forwarder)
    monkeypatch.setattr(celery_forwarder, "create_celery_service", MagicMock())
    monkeypatch.setattr(celery_forwarder, "start_celery_service", MagicMock())

    celery_forwarder.entrypoint()

    return load_forwarder.call_args.kwargs


def test_async_forwarder_does_not_inherit_the_sync_deadline(monkeypatch):
    # An async task is bounded by the queue's 24h visibility, not by the sync request deadline.
    kwargs = _run_entrypoint(monkeypatch, {"user_port": 5005})

    assert kwargs["timeout_seconds"] == celery_forwarder.MAX_ASYNC_TIMEOUT_SECONDS
    # The whole budget, hard limit included, fits inside the window the queue waits for an ack.
    assert (
        kwargs["timeout_seconds"] + celery_forwarder.CELERY_TIME_LIMIT_GRACE_SECONDS
        == celery_forwarder.DEFAULT_TASK_VISIBILITY_SECONDS
    )


def test_async_forwarder_deadline_is_overridable(monkeypatch):
    # The config file and --set stay authoritative over the default.
    kwargs = _run_entrypoint(monkeypatch, {"user_port": 5005, "timeout_seconds": 7200})

    assert kwargs["timeout_seconds"] == 7200


def test_async_forwarder_deadline_capped_at_the_visibility_window(monkeypatch):
    # An override past the ceiling would put the hard limit beyond the visibility timeout, where
    # the broker redelivers a task that is still running (task_acks_late is on).
    warn = MagicMock()
    monkeypatch.setattr(celery_forwarder, "logger", SimpleNamespace(warning=warn))

    kwargs = _run_entrypoint(monkeypatch, {"user_port": 5005, "timeout_seconds": 172800})

    assert kwargs["timeout_seconds"] == celery_forwarder.MAX_ASYNC_TIMEOUT_SECONDS
    warn.assert_called_once()
    assert "Capping forwarder.async.timeout_seconds" in warn.call_args.args[0]


@pytest.mark.parametrize(
    "visibility, expected",
    [
        ("VISIBILITY_24H", 86400),
        ("VISIBILITY_1H", 3600),
        ("VISIBILITY_10M", 600),
        ("VISIBILITY_1M", 60),
    ],
)
def test_task_time_limit_never_exceeds_visibility(visibility, expected):
    """celery_app asserts this for its app-level limit; a per-task time_limit bypasses that check.

    A forwarder built outside entrypoint() never passed the config-time cap, so the ceiling has to
    hold here as well.
    """
    limit = celery_forwarder.async_task_time_limit(
        999999, getattr(celery_forwarder.TaskVisibility, visibility)
    )

    assert limit == expected


def test_task_time_limit_stays_above_the_http_timeout_when_it_fits():
    limit = celery_forwarder.async_task_time_limit(
        7200, celery_forwarder.TaskVisibility.VISIBILITY_24H
    )

    assert limit == 7200 + celery_forwarder.CELERY_TIME_LIMIT_GRACE_SECONDS
    assert limit > 7200


def test_default_async_budget_lands_exactly_on_the_visibility_boundary():
    limit = celery_forwarder.async_task_time_limit(
        celery_forwarder.MAX_ASYNC_TIMEOUT_SECONDS, celery_forwarder.TaskVisibility.VISIBILITY_24H
    )

    assert limit == celery_forwarder.DEFAULT_TASK_VISIBILITY_SECONDS
    assert limit > celery_forwarder.MAX_ASYNC_TIMEOUT_SECONDS
