import os
import subprocess

from llm_engine_server.inference.common import unset_sensitive_envvars
from llm_engine_server.inference.sync_inference.constants import NUM_PROCESSES

PORT = os.environ["PORT"]


def start_server():
    # TODO: HTTPS
    # Copied from std-ml-srv
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
        "llm_engine_server.inference.sync_inference.fastapi_server:app",
    ]
    unset_sensitive_envvars()
    subprocess.run(command)


if __name__ == "__main__":
    start_server()
