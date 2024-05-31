"""
Starts the Gateway Service.

You can do this with `start-fastapi-server`.
"""

import argparse
import subprocess
from typing import List

# Uvicorn returns 503 instead of 429 when concurrency exceeds the limit
# We'll autoscale at target concurrency of a much lower number (around 50), and this just makes sure we don't 503 with bursty traffic
# We set this very high since model_engine_server/api/app.py sets a lower per-pod concurrency at which we start returning 429s
CONCURRENCY_LIMIT = 10000


def start_uvicorn_server(port: int, debug: bool) -> None:
    """Starts a Uvicorn server locally."""
    additional_args: List[str] = []
    if debug:
        additional_args.extend(["--reload", "--timeout-graceful-shutdown", "0"])
    command = [
        "uvicorn",
        "--host",
        "::",
        "--port",
        f"{port}",
        "--timeout-graceful-shutdown",
        "60",
        "--timeout-keep-alive",
        "2",
        # uvloop and httptools are both faster than their alternatives, but they are not compatible
        # with Windows or PyPy.
        "--loop",
        "uvloop",
        "--http",
        "httptools",
        "--limit-concurrency",
        f"{CONCURRENCY_LIMIT}",
        "--workers",
        "1",  # Let the Kubernetes deployment handle the number of pods
        *additional_args,
        "model_engine_server.api.app:app",
    ]

    subprocess.run(command, check=True)


def entrypoint():
    """Entrypoint for starting a local server."""

    # We can probably use asyncio since this service is going to be more I/O bound.
    parser = argparse.ArgumentParser(description="Hosted Inference Server")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", "-d", action="store_true")
    args = parser.parse_args()

    start_uvicorn_server(args.port, args.debug)


if __name__ == "__main__":
    entrypoint()
