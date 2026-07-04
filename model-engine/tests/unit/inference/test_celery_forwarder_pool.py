"""Guards the celery-forwarder's worker-pool wiring (MLI-7328).

The gevent monkey-patch lives at the top of celery_forwarder.py and must run BEFORE the module
imports celery/boto/requests, or their sockets stay un-patched and greenlets never yield. These
run in a subprocess (monkey-patching is process-global) and the child does the real assertion and
exits non-zero on failure, so the parent checks the exit code rather than scraping stdout/stderr:
  - CELERY_WORKER_POOL=gevent  -> importing the module patches socket + ssl
  - default (prefork)          -> nothing is patched (prefork is byte-for-byte unaffected)
  - an unsupported pool value  -> raises ValueError at import
  - gevent under ddtrace-run (-m, the prod launch) boots without the import-lock crash
"""

import os
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("gevent")

_IMPORT = "import model_engine_server.inference.forwarding.celery_forwarder"


def _run(check: str, pool: str) -> subprocess.CompletedProcess:
    # Subprocess because gevent monkey-patching is process-global; the child asserts and exits, so
    # we read the exit code instead of string-matching its output. timeout so a regression that
    # deadlocks (e.g. the import-lock hazard) fails the test instead of hanging the run.
    return subprocess.run(
        [sys.executable, "-c", check],
        env={**os.environ, "CELERY_WORKER_POOL": pool},
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_gevent_pool_monkeypatches_socket_and_ssl():
    check = (
        f"{_IMPORT}; import gevent.monkey as g;"
        " assert g.is_module_patched('socket') and g.is_module_patched('ssl')"
    )
    assert _run(check, "gevent").returncode == 0


def test_prefork_pool_does_not_monkeypatch():
    # default/prefork must not import or patch gevent, so existing behaviour is unchanged.
    check = (
        f"{_IMPORT}; import gevent.monkey as g;"
        " assert not g.is_module_patched('socket') and not g.is_module_patched('ssl')"
    )
    assert _run(check, "prefork").returncode == 0


def test_unsupported_pool_fails_fast():
    # A typo'd / unsupported pool must raise ValueError at import; exit 0 only if it did.
    check = (
        "try:\n"
        f"    {_IMPORT}\n"
        "except ValueError:\n"
        "    raise SystemExit(0)\n"
        "raise SystemExit(1)\n"
    )
    assert _run(check, "bogus").returncode == 0


def test_gevent_boots_under_ddtrace_run():
    # MLI-7328 #2 ordering guard: prod launches `ddtrace-run python -m ...celery_forwarder`.
    # ddtrace instruments at startup, then the module's patch_all() runs. The chain must boot via
    # __main__ (-m); a plain import under ddtrace-run instead deadlocks the import lock, which
    # surfaces as a non-zero exit. --help exits right after the module body, so no infra is needed.
    if not shutil.which("ddtrace-run"):
        pytest.skip("ddtrace-run not installed")
    proc = subprocess.run(
        [
            "ddtrace-run",
            sys.executable,
            "-m",
            "model_engine_server.inference.forwarding.celery_forwarder",
            "--help",
        ],
        env={**os.environ, "CELERY_WORKER_POOL": "gevent"},
        capture_output=True,
        text=True,
        timeout=60,  # a deadlock regression must fail here, not hang the run
    )
    assert proc.returncode == 0, proc.stderr
