"""Suite of integration tests for async inference with Celery. Enable with USE_REDIS_LOCALHOST=1."""

import os
import subprocess
from functools import lru_cache
from typing import Any, List, Optional, Tuple

import pytest
import redis
import requests
from fastapi import FastAPI
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_fixed

from spellbook_serve.common.dtos.model_endpoints import BrokerType
from spellbook_serve.common.dtos.tasks import (
    CallbackAuth,
    EndpointPredictV1Request,
    ResponseSchema,
    TaskStatus,
)
from spellbook_serve.common.env_vars import CIRCLECI
from spellbook_serve.infra.gateways import (
    CeleryTaskQueueGateway,
    LiveAsyncModelEndpointInferenceGateway,
)


@lru_cache(1)
def redis_available() -> bool:
    if not os.getenv("USE_REDIS_LOCALHOST"):
        return False
    try:
        return redis.Redis().ping()
    except:
        return False


@pytest.mark.skipif(CIRCLECI, reason="Skip on circleci before we figure out S3 access")
@pytest.mark.skipif(not redis_available(), reason="Redis is not available")
@pytest.mark.parametrize(
    "task_args,cloudpickle,expected_status,expected_result",
    [
        ({"y": 1}, False, TaskStatus.SUCCESS, ResponseSchema(__root__={"result": "1"})),
        ({"x": False, "y": 1}, False, TaskStatus.FAILURE, None),
    ],
)
def test_submit_and_get_tasks(
    queue: str,
    launch_celery_app: subprocess.Popen,
    callback_app: FastAPI,
    task_args: List[Any],
    cloudpickle: bool,
    expected_status: TaskStatus,
    expected_result: Any,
):
    gateway = LiveAsyncModelEndpointInferenceGateway(
        CeleryTaskQueueGateway(broker_type=BrokerType.REDIS_24H)
    )
    task = gateway.create_task(
        topic=queue,
        predict_request=EndpointPredictV1Request(
            args=task_args,
            cloudpickle=cloudpickle,
        ),
        task_timeout_seconds=60,
    )

    # Wait up to 10 seconds for the task to complete.
    for attempt in Retrying(
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type(AssertionError),
        wait=wait_fixed(1),
        reraise=True,
    ):
        with attempt:
            response = gateway.get_task(task.task_id)
            assert response.status == expected_status

    assert response.result == expected_result
    if expected_result == TaskStatus.FAILURE:
        assert response.traceback is not None


@pytest.mark.skipif(CIRCLECI, reason="Skip on circleci before we figure out S3 access")
@pytest.mark.skipif(not redis_available(), reason="Redis is not available")
@pytest.mark.parametrize(
    "callback_version,expected_callback_payload,callback_auth",
    [
        (None, {"result": "1", "task_id": "placeholder"}, None),
        ("0", {"result": "1", "task_id": "placeholder"}, None),
        ("1", {"result": "1", "task_id": "placeholder"}, ("basic", "user", "pass")),
    ],
)
def test_async_callbacks(
    queue: str,
    callback_port: int,
    test_user_id: str,
    launch_celery_app: subprocess.Popen,
    callback_app: FastAPI,
    callback_version: Optional[str],
    expected_callback_payload: Any,
    callback_auth: Optional[Tuple[str, str, str]],
):
    gateway = LiveAsyncModelEndpointInferenceGateway(
        CeleryTaskQueueGateway(broker_type=BrokerType.REDIS_24H)
    )

    task_args = {"y": 1}
    cloudpickle = False
    if callback_version is not None:
        callback_url = f"http://localhost:{callback_port}/v{callback_version}/callback"
    else:
        # If callback_version is None, then we don't pass a callback_url. Now we set it to 0 since
        # that is actually the default value.
        callback_url = None
        callback_version = "0"

    if callback_auth is not None:
        expected_credentials = dict(
            kind="basic", username=callback_auth[1], password=callback_auth[2]
        )
        task_callback_auth = CallbackAuth.parse_obj(expected_credentials)
    else:
        expected_credentials = dict(kind="basic", username=test_user_id, password="")
        task_callback_auth = None

    curr_callback_stats = requests.get(f"http://localhost:{callback_port}/callback-stats").json()
    previous_callback_count = curr_callback_stats["callback_count"][callback_version]

    gateway.create_task(
        topic=queue,
        predict_request=EndpointPredictV1Request(
            args=task_args,
            cloudpickle=cloudpickle,
            callback_url=callback_url,
            callback_auth=task_callback_auth,
        ),
        task_timeout_seconds=60,
    )

    # Wait up to 10 seconds for the task to complete.
    for attempt in Retrying(
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type(AssertionError),
        wait=wait_fixed(1),
        reraise=True,
    ):
        with attempt:
            # Note: We need to use a request to query for the callback stats instead of directly
            # using the callback_app fixture because the app is spawned in another process, meaning
            # that callback_app fixture is a copy of the one that is actually active.
            callback_stats = requests.get(f"http://localhost:{callback_port}/callback-stats").json()
            curr_callback_count = callback_stats["callback_count"][callback_version]
            assert curr_callback_count == previous_callback_count + 1

    actual_payload = callback_stats["last_request"][callback_version]
    expected_callback_payload["task_id"] = actual_payload["task_id"]
    assert actual_payload == expected_callback_payload

    assert callback_stats["last_auth"][callback_version] == expected_credentials
