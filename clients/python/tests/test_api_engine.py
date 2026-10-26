import http.server
import threading
from typing import Iterator, List, Optional, Tuple

import pytest
from llmengine import api_engine
from llmengine.api_engine import APIEngine
from llmengine.errors import UnknownError

API_KEY = "test-secret-key"

# (host, x-api-key, Authorization) for every request the server receives
Seen = List[Tuple[str, Optional[str], Optional[str]]]


@pytest.fixture
def redirecting_server() -> Iterator[Tuple[int, Seen]]:
    """Requests to 127.0.0.1 are redirected to localhost, i.e. a different origin."""
    seen: Seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _handle(self):
            host = self.headers["Host"]
            seen.append((host, self.headers.get("x-api-key"), self.headers.get("Authorization")))
            if host.startswith("127.0.0.1"):
                self.send_response(307)
                self.send_header("Location", f"http://localhost:{port}/steal")
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

        do_GET = do_POST = do_PUT = do_DELETE = _handle

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    api_engine.set_base_path(f"http://127.0.0.1:{port}/")
    api_engine.set_api_key(API_KEY)
    yield port, seen
    api_engine.set_base_path(None)
    api_engine.set_api_key(None)
    server.shutdown()


def _assert_key_not_redirected(seen: Seen):
    assert len(seen) == 1, f"redirect was followed: {seen}"
    assert seen[0][0].startswith("127.0.0.1")


@pytest.mark.parametrize(
    "call",
    [
        lambda: APIEngine._get("v1/files", timeout=5),
        lambda: APIEngine.put("v1/files", data={}, timeout=5),
        lambda: APIEngine._delete("v1/files", timeout=5),
        lambda: APIEngine.post_sync("v1/files", data={}, timeout=5),
        lambda: list(APIEngine.post_stream("v1/files", data={}, timeout=5)),
        lambda: APIEngine.post_file("v1/files", files={}, timeout=5),
    ],
    ids=["get", "put", "delete", "post_sync", "post_stream", "post_file"],
)
def test_sync_methods_do_not_follow_redirects(redirecting_server, call):
    _, seen = redirecting_server
    with pytest.raises(UnknownError, match="redirect"):
        call()
    _assert_key_not_redirected(seen)


async def test_apost_sync_does_not_follow_redirects(redirecting_server):
    _, seen = redirecting_server
    with pytest.raises(UnknownError, match="redirect"):
        await APIEngine.apost_sync("v1/files", data={}, timeout=5)
    _assert_key_not_redirected(seen)


async def test_apost_stream_does_not_follow_redirects(redirecting_server):
    _, seen = redirecting_server
    with pytest.raises(UnknownError, match="redirect"):
        async for _ in APIEngine.apost_stream("v1/files", data={}, timeout=5):
            pass
    _assert_key_not_redirected(seen)
