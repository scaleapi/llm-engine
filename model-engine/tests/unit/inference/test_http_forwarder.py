import threading
from dataclasses import dataclass
from typing import Mapping
from unittest import mock

import pytest
from fastapi import BackgroundTasks
from fastapi.responses import JSONResponse
from model_engine_server.common.dtos.tasks import EndpointPredictV1Request
from model_engine_server.inference.forwarding.forwarding import Forwarder
from model_engine_server.inference.forwarding.http_forwarder import (
    MultiprocessingConcurrencyLimiter,
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


@mock.patch("requests.post", mocked_post)
@mock.patch("requests.get", mocked_get)
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
