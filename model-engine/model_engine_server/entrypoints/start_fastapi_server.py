"""
Starts the Gateway Service.

You can do this with `start-fastapi-server`.
"""
import argparse
import subprocess
from typing import List


def start_gunicorn_server(
    port: int, num_workers: int, debug: bool, graceful_timeout: int, extra_args: List[str]
) -> None:
    """Starts a GUnicorn server locally."""
    additional_args: List[str] = []
    if debug:
        additional_args.extend(["--reload", "--timeout", "0"])
    command = [
        "gunicorn",
        "--bind",
        f"[::]:{port}",
        "--timeout",
        "60",
        "--keep-alive",
        "2",
        "--worker-class",
        "model_engine_server.api.worker.LaunchWorker",
        "--workers",
        f"{num_workers}",
        "--graceful-timeout",
        f"{graceful_timeout}",
        *additional_args,
        "model_engine_server.api.app:app",
        *extra_args,
    ]

    subprocess.run(command, check=True)


def entrypoint():
    """Entrypoint for starting a local server."""

    # We can probably use asyncio since this service is going to be more I/O bound.
    parser = argparse.ArgumentParser(description="Hosted Inference Server")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--graceful-timeout", type=int, default=600)
    args, extra_args = parser.parse_known_args()

    start_gunicorn_server(
        args.port, args.num_workers, args.debug, args.graceful_timeout, extra_args
    )


if __name__ == "__main__":
    entrypoint()
