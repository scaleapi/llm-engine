"""Integration tests for the celery-forwarder worker pools (MLI-7328).

Exercises the real forwarder (`python -m ...celery_forwarder`) end to end against a stub model
server, covering:
  - task completes under BOTH prefork (default) and gevent pools (redis broker),
  - gevent handles many concurrent in-flight tasks in ONE process (the memory win's premise),
  - gevent warm-shutdown drains an in-flight task on SIGTERM (pod scale-down),
  - the forwarder works over the SQS broker (prod's broker) under gevent, via localstack.

Local only: needs redis on localhost (USE_REDIS_LOCALHOST=1); the SQS case also needs a
localstack SQS endpoint (AWS_ENDPOINT_URL). Skipped in CI, like test_async_inference.py.
"""

import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from tempfile import NamedTemporaryFile
from typing import Iterator
from uuid import uuid4

import multiprocess
import pytest
import redis
import requests
import uvicorn
from fastapi import FastAPI, Request
from model_engine_server.common.constants import LIRA_CELERY_TASK_NAME
from model_engine_server.core.celery import TaskVisibility, celery_app
from model_engine_server.domain.entities import ModelEndpointConfig
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_fixed

MAIN_PORT = 5005  # forwarder.async.user_port (overridden via --set below)
CONFIG_PATH = "model_engine_server/inference/configs/service--forwarder.yaml"
AWS_ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL", "http://localhost:4566")


def _redis_available() -> bool:
    try:
        return bool(redis.Redis(host="localhost", port=6379).ping())
    except Exception:
        return False


def _localstack_sqs_available() -> bool:
    try:
        import boto3

        boto3.client("sqs", endpoint_url=AWS_ENDPOINT_URL, region_name="us-west-2").list_queues()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(not _redis_available(), reason="needs redis on localhost"),
    pytest.mark.skipif(
        bool(os.getenv("CIRCLECI")), reason="forwarder integration infra not wired in CI yet"
    ),
]


@pytest.fixture(autouse=True)
def _use_localhost_redis(monkeypatch):
    # The in-process producer resolves redis via get_redis_host_port(), which honors this env. Set
    # it per-test (auto-restored) instead of mutating global env at collection time.
    monkeypatch.setenv("USE_REDIS_LOCALHOST", "1")


@pytest.fixture(scope="module")
def stub_main() -> Iterator[int]:
    """Stand-in for the model container. /predict sleeps for an optional `sleep_s` found anywhere
    in the forwarded body (async sleep, so it serves concurrent requests like a real model)."""
    app = FastAPI()

    def _find_sleep(obj):
        if isinstance(obj, dict):
            if "sleep_s" in obj:
                return obj["sleep_s"]
            for v in obj.values():
                found = _find_sleep(v)
                if found is not None:
                    return found
        return None

    @app.get("/readyz")
    def readyz():
        return "OK"

    @app.post("/predict")
    async def predict(request: Request):
        import asyncio

        try:
            body = await request.json()
        except Exception:
            body = {}
        sleep_s = _find_sleep(body) or 0
        if sleep_s:
            await asyncio.sleep(float(sleep_s))
        return {"result": "ok"}

    proc = multiprocess.context.Process(
        target=uvicorn.run, args=(app,), kwargs={"port": MAIN_PORT, "log_level": "warning"}
    )
    proc.start()
    for attempt in Retrying(
        wait=wait_fixed(1),
        stop=stop_after_attempt(15),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=True,
    ):
        with attempt:
            assert (
                requests.get(f"http://localhost:{MAIN_PORT}/readyz", timeout=1).status_code == 200
            )
    yield MAIN_PORT
    proc.terminate()
    proc.join(timeout=10)
    if proc.is_alive():  # don't leave port 5005 bound for the next run
        proc.kill()


@pytest.fixture
def endpoint_config_location() -> Iterator[str]:
    # No post-inference hooks: keeps these tests independent of firehose/localstack (firehose
    # client-cache behaviour is unit-tested).
    serialized = ModelEndpointConfig(
        endpoint_name="it-endpoint",
        bundle_name="it-bundle",
        post_inference_hooks=[],
        user_id="it-user",
        endpoint_id="it-endpoint-id",
        endpoint_type="async",
        bundle_id="it-bundle-id",
        labels={},
    ).serialize()
    with NamedTemporaryFile(mode="w+") as f:
        f.write(serialized)
        f.seek(0)
        yield f.name


def _start_forwarder(
    queue: str,
    pool: str,
    endpoint_config_location: str,
    concurrency: int = 4,
    broker: str = "redis",
    sqs_url: str = None,
) -> subprocess.Popen:
    env = {
        **os.environ,
        "USE_REDIS_LOCALHOST": "1",
        "CELERY_WORKER_POOL": pool,
        "ENDPOINT_CONFIG_LOCATION": endpoint_config_location,
        "BASE_PATH": os.getcwd(),
    }
    cmd = [
        sys.executable,
        "-m",
        "model_engine_server.inference.forwarding.celery_forwarder",
        "--config",
        CONFIG_PATH,
        "--queue",
        queue,
        "--task-visibility",
        "VISIBILITY_24H",
        "--num-workers",
        str(concurrency),
        "--broker-type",
        broker,
        "--backend-protocol",
        "redis",
        "--set",
        f"forwarder.async.user_port={MAIN_PORT}",
        "--set",
        "forwarder.async.predict_route=/predict",
        "--set",
        "forwarder.async.healthcheck_route=/readyz",
    ]
    if broker == "sqs":
        cmd += ["--sqs-url", sqs_url]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _redis_producer():
    return celery_app(
        None,
        task_visibility=TaskVisibility.VISIBILITY_24H,
        broker_type="redis",
        backend_protocol="redis",
    )


def _drain(proc: subprocess.Popen):
    proc.kill()
    out, _ = proc.communicate()
    if out:
        print(out.decode(errors="replace"))


@pytest.mark.parametrize("pool", ["prefork", "gevent"])
def test_forwarder_processes_task_under_pool(pool, stub_main, endpoint_config_location):
    queue = f"it-fwd-{pool}-{str(uuid4())[-8:]}"
    producer = _redis_producer()
    worker = _start_forwarder(queue, pool, endpoint_config_location)
    try:
        payload = {"url": None, "args": {"x": 1}, "return_pickled": False}
        result = producer.send_task(
            LIRA_CELERY_TASK_NAME, args=[payload, datetime.utcnow()], queue=queue
        )
        assert result.get(timeout=60) is not None
    finally:
        _drain(worker)


def test_gevent_handles_concurrent_tasks_in_one_process(stub_main, endpoint_config_location):
    # 25 tasks that each take ~0.5s in the model, concurrency 20 in ONE gevent process. Serial
    # would take ~12.5s; concurrent should be a few seconds. Proves gevent gives prefork-like
    # concurrency without N processes.
    queue = f"it-fwd-conc-{str(uuid4())[-8:]}"
    producer = _redis_producer()
    worker = _start_forwarder(queue, "gevent", endpoint_config_location, concurrency=20)
    try:
        # Warm up: wait until the worker is booted and consuming before timing, so boot time is
        # not charged against the concurrency budget.
        warmup = {"url": None, "args": {"x": 1}, "return_pickled": False}
        assert (
            producer.send_task(
                LIRA_CELERY_TASK_NAME, args=[warmup, datetime.utcnow()], queue=queue
            ).get(timeout=60)
            is not None
        )
        payload = {"url": None, "args": {"sleep_s": 0.5}, "return_pickled": False}
        start = time.monotonic()
        results = [
            producer.send_task(
                LIRA_CELERY_TASK_NAME, args=[payload, datetime.utcnow()], queue=queue
            )
            for _ in range(25)
        ]
        for r in results:
            assert r.get(timeout=60) is not None
        elapsed = time.monotonic() - start
        assert elapsed < 8.0, f"25x0.5s tasks took {elapsed:.1f}s; gevent not running concurrently"
    finally:
        _drain(worker)


def test_gevent_warm_shutdown_drains_inflight_task(stub_main, endpoint_config_location):
    # SIGTERM the worker while a task is in flight (pod scale-down). Warm shutdown must let the
    # task finish, so the result is still retrievable.
    queue = f"it-fwd-term-{str(uuid4())[-8:]}"
    producer = _redis_producer()
    worker = _start_forwarder(queue, "gevent", endpoint_config_location)
    try:
        # Warm up so the worker is booted and consuming; otherwise boot can eat the window where
        # the real task is observably STARTED.
        warmup = {"url": None, "args": {"x": 1}, "return_pickled": False}
        assert (
            producer.send_task(
                LIRA_CELERY_TASK_NAME, args=[warmup, datetime.utcnow()], queue=queue
            ).get(timeout=60)
            is not None
        )
        payload = {"url": None, "args": {"sleep_s": 5.0}, "return_pickled": False}
        result = producer.send_task(
            LIRA_CELERY_TASK_NAME, args=[payload, datetime.utcnow()], queue=queue
        )
        # Wait until the task is actually running (track_started=True) before signalling, so we
        # test draining an in-flight task, not one still queued during worker boot.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and result.state != "STARTED":
            time.sleep(0.2)
        assert (
            result.state == "STARTED"
        ), f"task not in-flight (state={result.state}); cannot test drain"
        worker.send_signal(signal.SIGTERM)  # warm shutdown
        assert result.get(timeout=30) is not None  # in-flight task still completed
    finally:
        _drain(worker)


@pytest.mark.skipif(
    not _localstack_sqs_available(), reason="needs localstack SQS at AWS_ENDPOINT_URL"
)
def test_forwarder_processes_task_over_sqs_gevent(stub_main, endpoint_config_location):
    # Prod's broker is SQS. Validate the kombu/boto SQS transport under gevent end to end.
    import boto3
    from celery import Celery

    queue = f"it-fwd-sqs-{str(uuid4())[-8:]}"
    sqs = boto3.client("sqs", endpoint_url=AWS_ENDPOINT_URL, region_name="us-west-2")
    sqs_url = sqs.create_queue(QueueName=queue)["QueueUrl"]
    worker = _start_forwarder(
        queue, "gevent", endpoint_config_location, broker="sqs", sqs_url=sqs_url
    )
    producer = Celery()
    producer.conf.broker_url = "sqs://"
    producer.conf.broker_transport_options = {
        "predefined_queues": {queue: {"url": sqs_url}},
        "region": "us-west-2",
    }
    producer.conf.result_backend = (
        "redis://localhost:6379/1"  # worker writes db 1 (get_redis_endpoint(1))
    )
    try:
        payload = {
            "url": None,
            "args": {"x": 1},
            "return_pickled": False,
        }  # keep < 256KB (SQS limit)
        result = producer.send_task(
            LIRA_CELERY_TASK_NAME, args=[payload, datetime.utcnow()], queue=queue
        )
        assert result.get(timeout=90) is not None
    finally:
        _drain(worker)
