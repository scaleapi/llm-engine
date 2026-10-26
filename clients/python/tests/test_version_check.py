import json
import os
import socket
import subprocess
import sys
import textwrap
import threading
from typing import Iterator

import pytest

PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")

# Points pypi.org at a local socket that accepts connections but never responds.
# `_accepted` records when the version check connects.
BLACKHOLE_PYPI = textwrap.dedent(
    """
    import socket, threading, time
    _sock = socket.socket()
    _sock.bind(("127.0.0.1", 0))
    _sock.listen()
    _conns = []
    _accepted = threading.Event()
    _accepted_at = []
    def _accept_forever():
        while True:
            _conns.append(_sock.accept())
            _accepted_at.append(time.monotonic())
            _accepted.set()
    threading.Thread(target=_accept_forever, daemon=True).start()
    _getaddrinfo = socket.getaddrinfo
    def _patched(host, port, *args, **kwargs):
        if host == "pypi.org":
            return _getaddrinfo("127.0.0.1", _sock.getsockname()[1], *args, **kwargs)
        return _getaddrinfo(host, port, *args, **kwargs)
    socket.getaddrinfo = _patched
    """
)

IMPORT_AND_REPORT = textwrap.dedent(
    """
    import json, time
    import llmengine
    imported_at = time.monotonic()
    connected = _accepted.wait(10)
    print(json.dumps({
        "connected": connected,
        "import_after_connect": imported_at - _accepted_at[0] if connected else None,
    }))
    """
)


@pytest.fixture
def blackholed_pypi(monkeypatch) -> Iterator[None]:
    for var in PROXY_VARS:
        monkeypatch.delenv(var, raising=False)
    namespace: dict = {}
    original = socket.getaddrinfo
    exec(BLACKHOLE_PYPI, namespace)
    yield
    socket.getaddrinfo = original


def test_import_does_not_wait_for_version_check():
    env = {
        k: v
        for k, v in os.environ.items()
        if k != "LLM_ENGINE_DISABLE_VERSION_CHECK" and k not in PROXY_VARS
    }
    result = subprocess.run(
        [sys.executable, "-c", BLACKHOLE_PYPI + IMPORT_AND_REPORT],
        env=env,
        check=True,
        timeout=60,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout.strip().splitlines()[-1])
    # The version check still runs...
    assert report["connected"]
    # ...but import finishes while its request is outstanding. If import waited on the request,
    # it could only finish after the request's timeout (2s) expired.
    assert report["import_after_connect"] < 1.0


def test_check_version_times_out_on_unresponsive_pypi(blackholed_pypi, capsys):
    from llmengine import check_version

    thread = threading.Thread(target=check_version, daemon=True)
    thread.start()
    thread.join(timeout=15)
    assert not thread.is_alive(), "check_version did not time out"
    assert "Failed to check" in capsys.readouterr().out
