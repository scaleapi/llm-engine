import json
import threading
from dataclasses import dataclass
from typing import Mapping
from unittest import mock

import pytest
import requests_mock
from aioresponses import aioresponses
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from model_engine_server.common.dtos.tasks import EndpointPredictV1Request
from model_engine_server.domain.entities.model_endpoint_entity import ModelEndpointConfig
from model_engine_server.inference.forwarding.forwarding import Forwarder
from model_engine_server.inference.forwarding.http_forwarder import (
    MultiprocessingConcurrencyLimiter,
    get_concurrency_limiter,
    get_forwarder_loader,
    get_stream_passthrough_forwarder_loader,
    get_streaming_forwarder_loader,
    get_sync_passthrough_forwarder_loader,
    init_app,
    predict,
)
from model_engine_server.inference.infra.gateways.datadog_inference_monitoring_metrics_gateway import (
    DatadogInferenceMonitoringMetricsGateway,
)
from model_engine_server.inference.post_inference_hooks import PostInferenceHooksHandler
from tests.unit.conftest import FakeStreamingStorageGateway

PAYLOAD: Mapping[str, str] = {"hello": "world"}


class ExceptionCapturedThread(threading.Thread):
    def __init__(self, target, args):
        super().__init__(target=target, args=args)
        self.ex = None

    def run(self):
        try:
            self._target(*self._args)
        except Exception as e:
            self.ex = e

    def join(self):
        super().join()
        if self.ex is not None:
            raise self.ex


def mocked_get(*args, **kwargs):  # noqa
    @dataclass
    class mocked_static_status_code:
        status_code: int = 200

    return mocked_static_status_code()


def mocked_post(*args, **kwargs):  # noqa
    @dataclass
    class mocked_static_json:
        status_code: int = 200

        def json(self) -> dict:
            return PAYLOAD  # type: ignore

    return mocked_static_json()


def mocked_get_config():
    return {
        "sync": {
            "user_port": 5005,
            "user_hostname": "localhost",
            "use_grpc": False,
            "predict_route": "/predict",
            "healthcheck_route": "/readyz",
            "batch_route": None,
            "model_engine_unwrap": True,
            "serialize_results_as_string": True,
            "forward_http_status": True,
        },
        "stream": {
            "user_port": 5005,
            "user_hostname": "localhost",
            "predict_route": "/stream",
            "healthcheck_route": "/readyz",
            "batch_route": None,
            "model_engine_unwrap": True,
            "serialize_results_as_string": False,
        },
        "max_concurrency": 42,
    }


@pytest.fixture
def post_inference_hooks_handler():
    handler = PostInferenceHooksHandler(
        endpoint_name="test_endpoint_name",
        bundle_name="test_bundle_name",
        post_inference_hooks=[],
        user_id="test_user_id",
        billing_queue="billing_queue",
        billing_tags=[],
        default_callback_url=None,
        default_callback_auth=None,
        monitoring_metrics_gateway=DatadogInferenceMonitoringMetricsGateway(),
        endpoint_id="test_endpoint_id",
        endpoint_type="sync",
        bundle_id="test_bundle_id",
        labels={},
        streaming_storage_gateway=FakeStreamingStorageGateway(),
    )
    return handler


@pytest.fixture
def post_inference_hooks_handler_with_logging():
    handler = PostInferenceHooksHandler(
        endpoint_name="test_endpoint_name",
        bundle_name="test_bundle_name",
        post_inference_hooks=["logging"],
        user_id="test_user_id",
        billing_queue="billing_queue",
        billing_tags=[],
        default_callback_url=None,
        default_callback_auth=None,
        monitoring_metrics_gateway=DatadogInferenceMonitoringMetricsGateway(),
        endpoint_id="test_endpoint_id",
        endpoint_type="sync",
        bundle_id="test_bundle_id",
        labels={},
        streaming_storage_gateway=FakeStreamingStorageGateway(),
    )
    return handler


@pytest.fixture
def mock_request():
    return EndpointPredictV1Request(
        url="test_url",
        return_pickled=False,
        args={"x": 1},
    )


@mock.patch(
    "model_engine_server.inference.forwarding.http_forwarder.get_config",
    mocked_get_config,
)
def test_get_forwarder_loader():
    loader = get_forwarder_loader()
    assert loader.predict_route == "/predict"

    loader = get_forwarder_loader("/v1/chat/completions")
    assert loader.predict_route == "/v1/chat/completions"


@mock.patch(
    "model_engine_server.inference.forwarding.http_forwarder.get_config",
    mocked_get_config,
)
def test_get_streaming_forwarder_loader():
    loader = get_streaming_forwarder_loader()
    assert loader.predict_route == "/stream"

    loader = get_streaming_forwarder_loader("/v1/chat/completions")
    assert loader.predict_route == "/v1/chat/completions"


@mock.patch(
    "model_engine_server.inference.forwarding.http_forwarder.get_config",
    mocked_get_config,
)
def test_get_concurrency_limiter():
    limiter = get_concurrency_limiter()
    assert isinstance(limiter, MultiprocessingConcurrencyLimiter)
    assert limiter.concurrency == 42


@mock.patch("requests.post", mocked_post)
@mock.patch("requests.get", mocked_get)
@pytest.mark.skip(reason="This test is flaky")
def test_http_service_429(mock_request, post_inference_hooks_handler):
    mock_forwarder = Forwarder(
        "ignored",
        model_engine_unwrap=True,
        serialize_results_as_string=False,
        post_inference_hooks_handler=post_inference_hooks_handler,
        wrap_response=True,
        forward_http_status=True,
    )
    limiter = MultiprocessingConcurrencyLimiter(1, True)
    t1 = ExceptionCapturedThread(
        target=predict, args=(mock_request, BackgroundTasks(), mock_forwarder, limiter)
    )
    t2 = ExceptionCapturedThread(
        target=predict, args=(mock_request, BackgroundTasks(), mock_forwarder, limiter)
    )
    t1.start()
    t2.start()
    t1.join()
    with pytest.raises(Exception):  # 429 thrown
        t2.join()


def test_handler_response(post_inference_hooks_handler):
    try:
        post_inference_hooks_handler.handle(
            request_payload=mock_request, response=PAYLOAD, task_id="test_task_id"
        )
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")


def test_handler_json_response(post_inference_hooks_handler):
    try:
        post_inference_hooks_handler.handle(
            request_payload=mock_request,
            response=JSONResponse(content=PAYLOAD),
            task_id="test_task_id",
        )
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")


def test_handler_with_logging(post_inference_hooks_handler_with_logging):
    try:
        post_inference_hooks_handler_with_logging.handle(
            request_payload=mock_request,
            response=JSONResponse(content=PAYLOAD),
            task_id="test_task_id",
        )
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")


# Test the fastapi app


def mocked_get_config_with_extra_paths():
    return {
        "sync": {
            "user_port": 5005,
            "user_hostname": "localhost",
            "use_grpc": False,
            "predict_route": "/predict",
            "healthcheck_route": "/readyz",
            "batch_route": None,
            "model_engine_unwrap": True,
            "serialize_results_as_string": True,
            "forward_http_status": True,
            "extra_routes": ["/v1/chat/completions"],
        },
        "stream": {
            "user_port": 5005,
            "user_hostname": "localhost",
            "predict_route": "/stream",
            "healthcheck_route": "/readyz",
            "batch_route": None,
            "model_engine_unwrap": True,
            "serialize_results_as_string": False,
            "extra_routes": ["/v1/chat/completions"],
        },
        "max_concurrency": 42,
    }


def mocked_get_config_stream_passthrough():
    return {
        "sync": {
            "user_port": 5005,
            "user_hostname": "localhost",
            "use_grpc": False,
            "predict_route": "/predict",
            "healthcheck_route": "/readyz",
            "batch_route": None,
            "model_engine_unwrap": True,
            "serialize_results_as_string": True,
            "forward_http_status": True,
            "extra_routes": ["/v1/chat/completions"],
        },
        "stream": {
            "user_port": 5005,
            "user_hostname": "localhost",
            "predict_route": "/stream",
            "healthcheck_route": "/readyz",
            "batch_route": None,
            "model_engine_unwrap": True,
            "serialize_results_as_string": False,
            "extra_routes": ["/v1/chat/completions"],
            "forwarder_type": "passthrough",
        },
        "max_concurrency": 42,
    }


def mocked_get_config_sync_passthrough():
    return {
        "sync": {
            "user_port": 5005,
            "user_hostname": "localhost",
            "use_grpc": False,
            "predict_route": "/predict",
            "healthcheck_route": "/readyz",
            "batch_route": None,
            "model_engine_unwrap": True,
            "serialize_results_as_string": True,
            "forward_http_status": True,
            "extra_routes": ["/v1/chat/completions"],
            "forwarder_type": "passthrough",
        },
        "stream": {
            "user_port": 5005,
            "user_hostname": "localhost",
            "predict_route": "/stream",
            "healthcheck_route": "/readyz",
            "batch_route": None,
            "model_engine_unwrap": True,
            "serialize_results_as_string": False,
            "extra_routes": ["/v1/chat/completions"],
        },
        "max_concurrency": 42,
    }


def get_predict_endpoint(config):
    cfg_sync = config["sync"]
    predict_endpoint = (
        f"http://{cfg_sync['user_hostname']}:{cfg_sync['user_port']}{cfg_sync['predict_route']}"
    )
    return predict_endpoint


def get_healthcheck_endpoint(config):
    cfg_sync = config["sync"]
    healthcheck_endpoint = (
        f"http://{cfg_sync['user_hostname']}:{cfg_sync['user_port']}{cfg_sync['healthcheck_route']}"
    )
    return healthcheck_endpoint


def get_stream_endpoint(config):
    cfg_stream = config["stream"]
    stream_endpoint = f"http://{cfg_stream['user_hostname']}:{cfg_stream['user_port']}{cfg_stream['predict_route']}"
    return stream_endpoint


def get_chat_endpoint(config):
    cfg_sync = config["sync"]
    chat_endpoint = (
        f"http://{cfg_sync['user_hostname']}:{cfg_sync['user_port']}{cfg_sync['extra_routes'][0]}"
    )
    return chat_endpoint


def mocked_get_endpoint_config():
    return ModelEndpointConfig(
        endpoint_name="test_endpoint_name",
        bundle_name="test_bundle_name",
    )


@pytest.fixture()
@mock.patch(
    "model_engine_server.inference.forwarding.http_forwarder.get_config",
    mocked_get_config_with_extra_paths,
)
@mock.patch(
    "model_engine_server.inference.forwarding.forwarding.get_endpoint_config",
    mocked_get_endpoint_config,
)
async def mocked_app() -> FastAPI:
    with requests_mock.Mocker() as req_mock:
        healthcheck_endpoint = get_healthcheck_endpoint(mocked_get_config_with_extra_paths())
        req_mock.get(
            healthcheck_endpoint,
            json={"status": "ok"},
        )
        app = await init_app()
        return app


@pytest.fixture()
@mock.patch(
    "model_engine_server.inference.forwarding.http_forwarder.get_config",
    mocked_get_config_stream_passthrough,
)
@mock.patch(
    "model_engine_server.inference.forwarding.forwarding.get_endpoint_config",
    mocked_get_endpoint_config,
)
async def mocked_stream_passthrough_app() -> FastAPI:
    with requests_mock.Mocker() as req_mock:
        healthcheck_endpoint = get_healthcheck_endpoint(mocked_get_config_stream_passthrough())
        req_mock.get(
            healthcheck_endpoint,
            json={"status": "ok"},
        )
        app = await init_app()
        return app


@pytest.fixture()
@mock.patch(
    "model_engine_server.inference.forwarding.http_forwarder.get_config",
    mocked_get_config_sync_passthrough,
)
@mock.patch(
    "model_engine_server.inference.forwarding.forwarding.get_endpoint_config",
    mocked_get_endpoint_config,
)
async def mocked_sync_passthrough_app() -> FastAPI:
    with requests_mock.Mocker() as req_mock:
        healthcheck_endpoint = get_healthcheck_endpoint(mocked_get_config_sync_passthrough())
        req_mock.get(
            healthcheck_endpoint,
            json={"status": "ok"},
        )
        app = await init_app()
        return app


def wrap_request(request):
    return {"url": "", "args": request}


def wrap_result(result):
    return {"result": result}


@pytest.mark.anyio
@mock.patch(
    "model_engine_server.inference.forwarding.http_forwarder.get_config",
    mocked_get_config_with_extra_paths,
)
@mock.patch(
    "model_engine_server.inference.forwarding.forwarding.get_endpoint_config",
    mocked_get_endpoint_config,
)
async def test_mocked_app_success(mocked_app):
    config = mocked_get_config_with_extra_paths()
    config_sync = config["sync"]
    # config_stream = config["stream"]

    predict_endpoint = get_predict_endpoint(config)
    healthcheck_endpoint = get_healthcheck_endpoint(config)

    # stream_endpoint = get_stream_endpoint(config)
    chat_endpoint = get_chat_endpoint(config)

    raw_payload = {"prompt": "Hello", "stream": False}
    raw_result = {"message": "Hello World"}

    payload = wrap_request(raw_payload)
    expected_result = wrap_result(
        json.dumps(raw_result) if config_sync["serialize_results_as_string"] else raw_result
    )
    with (
        TestClient(mocked_app) as client,
        aioresponses() as aio_mock,
        requests_mock.Mocker() as req_mock,
    ):
        req_mock.get(
            healthcheck_endpoint,
            json={"status": "ok"},
        )
        aio_mock.post(predict_endpoint, status=200, payload=raw_result)
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json() == expected_result

        aio_mock.post(chat_endpoint, status=200, payload=raw_result)
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert response.json() == expected_result

        # TODO: add tests for streaming; it's not as trivial as I'd hoped


# Tests for passthrough routes


@pytest.mark.anyio
@mock.patch(
    "model_engine_server.inference.forwarding.http_forwarder.get_config",
    mocked_get_config_stream_passthrough,
)
@mock.patch(
    "model_engine_server.inference.forwarding.forwarding.get_endpoint_config",
    mocked_get_endpoint_config,
)
async def test_stream_passthrough_routes_post(mocked_stream_passthrough_app):
    """Test stream passthrough route with POST method"""
    config = mocked_get_config_stream_passthrough()
    healthcheck_endpoint = get_healthcheck_endpoint(config)
    chat_endpoint = get_chat_endpoint(config)

    raw_payload = {"prompt": "Hello", "stream": True}
    raw_result = b'{"message": "Hello World"}\n'

    with (
        TestClient(mocked_stream_passthrough_app) as client,
        aioresponses() as aio_mock,
        requests_mock.Mocker() as req_mock,
    ):
        req_mock.get(healthcheck_endpoint, json={"status": "ok"})

        # Mock the passthrough endpoint
        aio_mock.post(
            chat_endpoint, status=200, body=raw_result, headers={"content-type": "application/json"}
        )

        response = client.post(
            "/v1/chat/completions",
            json=raw_payload,
            headers={"Authorization": "Bearer test-token", "User-Agent": "test-client"},
        )
        assert response.status_code == 200


@pytest.mark.anyio
@mock.patch(
    "model_engine_server.inference.forwarding.http_forwarder.get_config",
    mocked_get_config_stream_passthrough,
)
@mock.patch(
    "model_engine_server.inference.forwarding.forwarding.get_endpoint_config",
    mocked_get_endpoint_config,
)
async def test_stream_passthrough_routes_get(mocked_stream_passthrough_app):
    """Test stream passthrough route with GET method"""
    config = mocked_get_config_stream_passthrough()
    healthcheck_endpoint = get_healthcheck_endpoint(config)
    chat_endpoint = get_chat_endpoint(config)

    raw_result = b'{"models": ["model1", "model2"]}\n'

    with (
        TestClient(mocked_stream_passthrough_app) as client,
        aioresponses() as aio_mock,
        requests_mock.Mocker() as req_mock,
    ):
        req_mock.get(healthcheck_endpoint, json={"status": "ok"})

        # Mock the passthrough endpoint
        aio_mock.get(
            f"{chat_endpoint}?param=value",
            status=200,
            body=raw_result,
            headers={"content-type": "application/json"},
        )

        response = client.get(
            "/v1/chat/completions?param=value", headers={"Authorization": "Bearer test-token"}
        )
        assert response.status_code == 200


@pytest.mark.anyio
@mock.patch(
    "model_engine_server.inference.forwarding.http_forwarder.get_config",
    mocked_get_config_sync_passthrough,
)
@mock.patch(
    "model_engine_server.inference.forwarding.forwarding.get_endpoint_config",
    mocked_get_endpoint_config,
)
async def test_sync_passthrough_routes_post(mocked_sync_passthrough_app):
    """Test sync passthrough route with POST method"""
    config = mocked_get_config_sync_passthrough()
    healthcheck_endpoint = get_healthcheck_endpoint(config)
    chat_endpoint = get_chat_endpoint(config)

    raw_payload = {"prompt": "Hello", "stream": False}
    raw_result = {"message": "Hello World"}

    with (
        TestClient(mocked_sync_passthrough_app) as client,
        aioresponses() as aio_mock,
        requests_mock.Mocker() as req_mock,
    ):
        req_mock.get(healthcheck_endpoint, json={"status": "ok"})

        # Mock the passthrough endpoint
        aio_mock.post(
            chat_endpoint,
            status=200,
            payload=raw_result,
            headers={"content-type": "application/json"},
        )

        response = client.post(
            "/v1/chat/completions",
            json=raw_payload,
            headers={"Authorization": "Bearer test-token", "User-Agent": "test-client"},
        )
        assert response.status_code == 200


@pytest.mark.anyio
@mock.patch(
    "model_engine_server.inference.forwarding.http_forwarder.get_config",
    mocked_get_config_sync_passthrough,
)
@mock.patch(
    "model_engine_server.inference.forwarding.forwarding.get_endpoint_config",
    mocked_get_endpoint_config,
)
async def test_sync_passthrough_routes_put(mocked_sync_passthrough_app):
    """Test sync passthrough route with PUT method"""
    config = mocked_get_config_sync_passthrough()
    healthcheck_endpoint = get_healthcheck_endpoint(config)
    chat_endpoint = get_chat_endpoint(config)

    raw_payload = {"model": "updated-model", "params": {"temperature": 0.7}}
    raw_result = {"status": "updated"}

    with (
        TestClient(mocked_sync_passthrough_app) as client,
        aioresponses() as aio_mock,
        requests_mock.Mocker() as req_mock,
    ):
        req_mock.get(healthcheck_endpoint, json={"status": "ok"})

        # Mock the passthrough endpoint
        aio_mock.put(
            chat_endpoint,
            status=200,
            payload=raw_result,
            headers={"content-type": "application/json"},
        )

        response = client.put(
            "/v1/chat/completions", json=raw_payload, headers={"Authorization": "Bearer test-token"}
        )
        assert response.status_code == 200


@pytest.mark.anyio
@mock.patch(
    "model_engine_server.inference.forwarding.http_forwarder.get_config",
    mocked_get_config_sync_passthrough,
)
@mock.patch(
    "model_engine_server.inference.forwarding.forwarding.get_endpoint_config",
    mocked_get_endpoint_config,
)
async def test_sync_passthrough_routes_delete(mocked_sync_passthrough_app):
    """Test sync passthrough route with DELETE method"""
    config = mocked_get_config_sync_passthrough()
    healthcheck_endpoint = get_healthcheck_endpoint(config)
    chat_endpoint = get_chat_endpoint(config)

    raw_result = {"status": "deleted"}

    with (
        TestClient(mocked_sync_passthrough_app) as client,
        aioresponses() as aio_mock,
        requests_mock.Mocker() as req_mock,
    ):
        req_mock.get(healthcheck_endpoint, json={"status": "ok"})

        # Mock the passthrough endpoint
        aio_mock.delete(
            chat_endpoint,
            status=200,
            payload=raw_result,
            headers={"content-type": "application/json"},
        )

        response = client.delete(
            "/v1/chat/completions", headers={"Authorization": "Bearer test-token"}
        )
        assert response.status_code == 200


@pytest.mark.anyio
@mock.patch(
    "model_engine_server.inference.forwarding.http_forwarder.get_config",
    mocked_get_config_sync_passthrough,
)
@mock.patch(
    "model_engine_server.inference.forwarding.forwarding.get_endpoint_config",
    mocked_get_endpoint_config,
)
async def test_sync_passthrough_routes_patch(mocked_sync_passthrough_app):
    """Test sync passthrough route with PATCH method"""
    config = mocked_get_config_sync_passthrough()
    healthcheck_endpoint = get_healthcheck_endpoint(config)
    chat_endpoint = get_chat_endpoint(config)

    raw_payload = {"temperature": 0.9}
    raw_result = {"status": "patched"}

    with (
        TestClient(mocked_sync_passthrough_app) as client,
        aioresponses() as aio_mock,
        requests_mock.Mocker() as req_mock,
    ):
        req_mock.get(healthcheck_endpoint, json={"status": "ok"})

        # Mock the passthrough endpoint
        aio_mock.patch(
            chat_endpoint,
            status=200,
            payload=raw_result,
            headers={"content-type": "application/json"},
        )

        response = client.patch(
            "/v1/chat/completions", json=raw_payload, headers={"Authorization": "Bearer test-token"}
        )
        assert response.status_code == 200


@pytest.mark.anyio
@mock.patch(
    "model_engine_server.inference.forwarding.http_forwarder.get_config",
    mocked_get_config_sync_passthrough,
)
@mock.patch(
    "model_engine_server.inference.forwarding.forwarding.get_endpoint_config",
    mocked_get_endpoint_config,
)
async def test_sync_passthrough_routes_head(mocked_sync_passthrough_app):
    """Test sync passthrough route with HEAD method"""
    config = mocked_get_config_sync_passthrough()
    healthcheck_endpoint = get_healthcheck_endpoint(config)
    chat_endpoint = get_chat_endpoint(config)

    with (
        TestClient(mocked_sync_passthrough_app) as client,
        aioresponses() as aio_mock,
        requests_mock.Mocker() as req_mock,
    ):
        req_mock.get(healthcheck_endpoint, json={"status": "ok"})

        # Mock the passthrough endpoint
        aio_mock.head(
            chat_endpoint,
            status=200,
            headers={"content-type": "application/json", "content-length": "100"},
        )

        response = client.head(
            "/v1/chat/completions", headers={"Authorization": "Bearer test-token"}
        )
        assert response.status_code == 200


@pytest.mark.anyio
@mock.patch(
    "model_engine_server.inference.forwarding.http_forwarder.get_config",
    mocked_get_config_sync_passthrough,
)
@mock.patch(
    "model_engine_server.inference.forwarding.forwarding.get_endpoint_config",
    mocked_get_endpoint_config,
)
async def test_sync_passthrough_routes_options(mocked_sync_passthrough_app):
    """Test sync passthrough route with OPTIONS method"""
    config = mocked_get_config_sync_passthrough()
    healthcheck_endpoint = get_healthcheck_endpoint(config)
    chat_endpoint = get_chat_endpoint(config)

    with (
        TestClient(mocked_sync_passthrough_app) as client,
        aioresponses() as aio_mock,
        requests_mock.Mocker() as req_mock,
    ):
        req_mock.get(healthcheck_endpoint, json={"status": "ok"})

        # Mock the passthrough endpoint
        aio_mock.options(
            chat_endpoint,
            status=200,
            headers={
                "allow": "GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS",
                "access-control-allow-methods": "GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS",
            },
        )

        response = client.options("/v1/chat/completions")
        assert response.status_code == 200


def test_get_stream_passthrough_forwarder_loader():
    """Test getting stream passthrough forwarder loader"""
    with mock.patch(
        "model_engine_server.inference.forwarding.http_forwarder.get_config",
        mocked_get_config_stream_passthrough,
    ):
        loader = get_stream_passthrough_forwarder_loader()
        assert loader.user_port == 5005
        assert loader.user_hostname == "localhost"

        loader_with_path = get_stream_passthrough_forwarder_loader("/custom/route")
        assert loader_with_path.passthrough_route == "/custom/route"


def test_get_sync_passthrough_forwarder_loader():
    """Test getting sync passthrough forwarder loader"""
    with mock.patch(
        "model_engine_server.inference.forwarding.http_forwarder.get_config",
        mocked_get_config_sync_passthrough,
    ):
        loader = get_sync_passthrough_forwarder_loader()
        assert loader.user_port == 5005
        assert loader.user_hostname == "localhost"

        loader_with_path = get_sync_passthrough_forwarder_loader("/custom/route")
        assert loader_with_path.passthrough_route == "/custom/route"
