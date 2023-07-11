import json
import subprocess
from pathlib import Path

from spellbook_serve.api.app import app

MODULE_PATH = Path(__file__).resolve()
LAUNCH_SERVICE_BASE = MODULE_PATH.parents[2].resolve()
OPENAPI_PATH = (LAUNCH_SERVICE_BASE / "clients/openapi.json").resolve()
LANGUAGE_TO_GENERATOR_NAME = dict(python="python", typescript="typescript-axios")


def dump_openapi(openapi_path: str):
    """Writes the OpenAPI schema to the specified path."""
    with open(openapi_path, "w") as file:
        schema = app.openapi()
        file.write(json.dumps(schema, indent=4, sort_keys=True))


def run_openapi_generator():
    """Launches a subprocess with the OpenAPI generator."""
    print("🏭 Generating client")
    command = ["docker-compose run openapi-generator-cli"]
    subprocess.run(
        command,
        cwd=str((LAUNCH_SERVICE_BASE / "../ml_infra_core").resolve()),
        check=True,
        shell=True,
    )


def entrypoint():
    """Entrypoint for autogenerating client and documentation."""
    dump_openapi(str(OPENAPI_PATH))
    run_openapi_generator()


if __name__ == "__main__":
    entrypoint()
