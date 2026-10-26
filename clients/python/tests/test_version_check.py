import os
import socket
import subprocess
import sys
import textwrap
import threading
import time
from typing import Iterator

import pytest

# Points pypi.org at a local socket that accepts connections but never responds.
BLACKHOLE_PYPI = textwrap.dedent(
    """
    import socket, threading
    _sock = socket.socket()
    _sock.bind(("127.0.0.1", 0))
    _sock.listen()
    _conns = []
    threading.Thread(target=lambda: [_conns.append(_sock.accept()) for _ in iter(int, 1)], daemon=True).start()
    _getaddrinfo = socket.getaddrinfo
    def _patched(host, port, *args, **kwargs):
        if host == "pypi.org":
            return _getaddrinfo("127.0.0.1", _sock.getsockname()[1], *args, **kwargs)
        return _getaddrinfo(host, port, *args, **kwargs)
    socket.getaddrinfo = _patched
    """
)


@pytest.fixture
def blackholed_pypi() -> Iterator[None]:
    namespace: dict = {}
    original = socket.getaddrinfo
    exec(BLACKHOLE_PYPI, namespace)
    yield
    socket.getaddrinfo = original


def test_import_does_not_block_on_unresponsive_pypi():
    env = {k: v for k, v in os.environ.items() if k != "LLM_ENGINE_DISABLE_VERSION_CHECK"}
    start = time.monotonic()
    subprocess.run(
        [sys.executable, "-c", BLACKHOLE_PYPI + "\nimport llmengine\n"],
        env=env,
        check=True,
        timeout=30,
    )
    # Importing must not wait for the version check at all
    assert time.monotonic() - start < 15


def test_check_version_times_out_on_unresponsive_pypi(blackholed_pypi, capsys):
    from llmengine import check_version

    thread = threading.Thread(target=check_version, daemon=True)
    thread.start()
    thread.join(timeout=15)
    assert not thread.is_alive(), "check_version did not time out"
    assert "Failed to check" in capsys.readouterr().out
