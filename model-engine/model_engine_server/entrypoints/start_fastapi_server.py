"""
Starts the Gateway Service.

You can do this with `start-fastapi-server`.
"""

import argparse
import os
import subprocess
from typing import List


def start_gunicorn_server(port: int, num_workers: int, debug: bool) -> None:
    """Starts a GUnicorn server locally."""
    additional_args: List[str] = []
    if debug:
        additional_args.extend(["--reload", "--timeout", "0"])
    
    # Use environment variables for configuration with fallbacks
    timeout = int(os.environ.get('WORKER_TIMEOUT', os.environ.get('GUNICORN_TIMEOUT', 60)))
    graceful_timeout = int(os.environ.get('GUNICORN_GRACEFUL_TIMEOUT', timeout))
    keep_alive = int(os.environ.get('GUNICORN_KEEP_ALIVE', 2))
    worker_class = os.environ.get('GUNICORN_WORKER_CLASS', 'model_engine_server.api.worker.LaunchWorker')
    
    command = [
        "gunicorn",
        "--bind",
        f"[::]:{port}",
        "--timeout",
        str(timeout),
        "--graceful-timeout",
        str(graceful_timeout),
        "--keep-alive",
        str(keep_alive),
        "--worker-class",
        worker_class,
        "--workers",
        f"{num_workers}",
        *additional_args,
        "model_engine_server.api.app:app",
    ]

    subprocess.run(command, check=True)


def entrypoint():
    """Entrypoint for starting a local server."""

    # We can probably use asyncio since this service is going to be more I/O bound.
    parser = argparse.ArgumentParser(description="Hosted Inference Server")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--debug", "-d", action="store_true")
    args = parser.parse_args()

    start_gunicorn_server(args.port, args.num_workers, args.debug)


if __name__ == "__main__":
    entrypoint()
