import argparse
import os
import subprocess

from model_engine_server.inference.common import unset_sensitive_envvars
from model_engine_server.inference.sync_inference.constants import NUM_PROCESSES

PORT = os.environ["PORT"]


def start_server():
    parser = argparse.ArgumentParser()
    _, extra_args = parser.parse_known_args()

    # TODO: HTTPS
    command = [
        "gunicorn",
        "--bind",
        f"[::]:{PORT}",
        "--timeout",
        "1200",
        "--keep-alive",
        "2",
        "--worker-class",
        "uvicorn.workers.UvicornWorker",
        "--workers",
        str(NUM_PROCESSES),
        "model_engine_server.inference.sync_inference.fastapi_server:app",
        *extra_args,
    ]
    unset_sensitive_envvars()
    subprocess.run(command)


if __name__ == "__main__":
    start_server()
