"""Guards the celery-forwarder's worker-pool wiring (MLI-7328).

The gevent monkey-patch lives at the top of celery_forwarder.py and must run BEFORE the module
imports celery/boto/requests, or their sockets stay un-patched and greenlets never yield. These
run in a subprocess (monkey-patching is process-global) and assert:
  - CELERY_WORKER_POOL=gevent  -> importing the module patches sockets
  - default (prefork)          -> nothing is patched (prefork is byte-for-byte unaffected)
"""
import os
import subprocess
import sys

import pytest

pytest.importorskip("gevent")

_CHECK = (
    "import model_engine_server.inference.forwarding.celery_forwarder;"  # runs the guarded patch
    "import gevent.monkey;"
    "print('socket_patched', gevent.monkey.is_module_patched('socket'))"
)


def _import_with_pool(pool: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", _CHECK],
        env={**os.environ, "CELERY_WORKER_POOL": pool},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_gevent_pool_monkeypatches_sockets():
    assert "socket_patched True" in _import_with_pool("gevent")


def test_prefork_pool_does_not_monkeypatch():
    # default/prefork must not import or patch gevent, so existing behaviour is unchanged.
    assert "socket_patched False" in _import_with_pool("prefork")
