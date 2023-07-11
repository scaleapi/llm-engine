import os
import random
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator
from uuid import uuid4

import multiprocess
import pytest
import requests
import uvicorn
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_fixed

from spellbook_serve.common.constants import READYZ_FPATH
from spellbook_serve.common.serialization_utils import python_json_to_b64
from spellbook_serve.core.config import ml_infra_config
from spellbook_serve.domain.entities import CallbackAuth, CallbackBasicAuth, ModelEndpointConfig

MODULE_PATH = Path(__file__).resolve()
BASE_PATH = MODULE_PATH.parents[4]

# For reference the load_predict_fn is defined as follows:
#
# def returns_returns_1(x):
#   def returns_1(y):
#       return 1
#
#   return returns_1

QUEUE = str(uuid4())[-12:]
CALLBACK_PORT = 8000 + random.randint(0, 1000)


@pytest.fixture(scope="session")
def queue() -> str:
    return QUEUE


@pytest.fixture(scope="session")
def test_user_id() -> str:
    return "test_user_id_1"


@pytest.fixture(scope="session")
def test_default_callback_auth() -> CallbackAuth:
    return CallbackAuth(
        __root__=CallbackBasicAuth(kind="basic", username="test_user", password="test_password")
    )


@pytest.fixture(scope="session")
def user_config_location() -> Iterator[str]:
    with NamedTemporaryFile(mode="w+") as f:
        f.write(python_json_to_b64(None))
        f.seek(0)
        yield f.name


@pytest.fixture(scope="session")
def endpoint_config_location(callback_port: int, test_user_id: str) -> Iterator[str]:
    endpoint_config_serialized = ModelEndpointConfig(
        endpoint_name="test-endpoint",
        bundle_name="test-bundle",
        post_inference_hooks=["callback"],
        default_callback_url=f"http://localhost:{callback_port}/v0/callback",
        user_id=test_user_id,
    ).serialize()
    with NamedTemporaryFile(mode="w+") as f:
        f.write(endpoint_config_serialized)
        f.seek(0)
        yield f.name


@pytest.fixture(scope="session")
def launch_celery_app(
    queue: str, user_config_location: str, endpoint_config_location: str
) -> Iterator[subprocess.Popen]:
    env = dict(
        AWS_PROFILE="default" if os.getenv("CIRCLECI") else "ml-worker",
        BROKER_TYPE="redis",
        USE_REDIS_LOCALHOST=1,
        CELERY_S3_BUCKET=ml_infra_config().s3_bucket,
        RESULTS_S3_BUCKET=ml_infra_config().s3_bucket,
        CHILD_FN_INFO="{}",
        BASE_PATH=str(BASE_PATH),
        PREWARM=True,
        BUNDLE_URL=f"s3://{ml_infra_config().s3_bucket}/model_bundles/61a67d767bce560024c7eb96/f0142411-51e1-4357-a405-ee5fef87d977",
        USER_CONFIG_LOCATION=user_config_location,
        ENDPOINT_CONFIG_LOCATION=endpoint_config_location,
    )

    env_str = " ".join(f"{k}={v}" for k, v in env.items())
    command = (
        f"{env_str} exec celery --app=spellbook_serve.inference.async_inference worker "
        f"--loglevel=INFO --concurrency=1 --queues={queue}"
    )
    # Wait up to 10 seconds for process to start and be ready.
    with subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as process:
        for attempt in Retrying(
            wait=wait_fixed(1),
            stop=stop_after_attempt(10),
            retry=retry_if_exception_type(FileNotFoundError),
        ):
            with attempt:
                if not os.path.exists(READYZ_FPATH):
                    raise FileNotFoundError
        yield process
        process.kill()

        outs, errs = process.communicate()
        info = f"""
        Stdout: {outs.decode()}
        Stderr: {errs.decode()}
        """
        print(info)


@pytest.fixture(scope="session")
def callback_port() -> int:
    return CALLBACK_PORT


@pytest.fixture(scope="session")
def callback_app(callback_port: int) -> Iterator[FastAPI]:
    app = FastAPI(
        callback_count={0: 0, 1: 0},
        last_request={0: None, 1: None},
        last_auth={0: None, 1: None},
    )

    AUTH = HTTPBasic(auto_error=False)

    @app.get("/readyz")
    def readyz():
        return "OK"

    @app.post("/v0/callback")
    async def callback_v0(
        request: Request,
        creds: HTTPBasicCredentials = Depends(AUTH),
    ):
        app.extra["callback_count"][0] += 1
        app.extra["last_request"][0] = await request.json()
        app.extra["last_auth"][0] = dict(kind="basic", **creds.dict())
        return "OK"

    @app.post("/v1/callback")
    async def callback_v1(
        request: Request,
        creds: HTTPBasicCredentials = Depends(AUTH),
    ):
        app.extra["callback_count"][1] += 1
        app.extra["last_request"][1] = await request.json()
        app.extra["last_auth"][1] = dict(kind="basic", **creds.dict())
        return "OK"

    @app.get("/callback-stats")
    def callback_stats():
        return JSONResponse(content=app.extra)

    process = multiprocess.context.Process(
        target=uvicorn.run, args=(app,), kwargs={"port": callback_port}
    )
    process.start()

    for attempt in Retrying(
        wait=wait_fixed(1),
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type((AssertionError, requests.exceptions.RequestException)),
        reraise=True,
    ):
        with attempt:
            readyz_response = requests.get(f"http://localhost:{callback_port}/readyz", timeout=1)
            assert readyz_response.status_code == 200

    yield app

    process.terminate()
