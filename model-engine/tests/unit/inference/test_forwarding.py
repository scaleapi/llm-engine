import json
from dataclasses import dataclass
from typing import List, Mapping, Tuple
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import pytest
from aioresponses import aioresponses
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from model_engine_server.core.utils.env import environment
from model_engine_server.domain.entities import ModelEndpointConfig
from model_engine_server.inference.domain.gateways.inference_monitoring_metrics_gateway import (
    InferenceMonitoringMetricsGateway,
)
from model_engine_server.inference.forwarding.forwarding import (
    ENV_SERIALIZE_RESULTS_AS_STRING,
    FORWARDER_OVERHEAD_METRIC,
    FORWARDER_TOTAL_METRIC,
    FORWARDER_TTFT_METRIC,
    FORWARDER_VLLM_ROUNDTRIP_METRIC,
    FORWARDER_VLLM_TTFT_METRIC,
    KEY_SERIALIZE_RESULTS_AS_STRING,
    Forwarder,
    LoadForwarder,
    LoadPassthroughForwarder,
    LoadStreamingForwarder,
    PassthroughForwarder,
    StreamingForwarder,
    load_named_config,
)
from model_engine_server.inference.infra.gateways.datadog_inference_monitoring_metrics_gateway import (
    DatadogInferenceMonitoringMetricsGateway,
)
from model_engine_server.inference.post_inference_hooks import PostInferenceHooksHandler
from tests.unit.conftest import FakeStreamingStorageGateway

PAYLOAD: Mapping[str, str] = {"hello": "world"}
PAYLOAD_END = "[DONE]"


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


def mocked_post_400(*args, **kwargs):  # noqa
    @dataclass
    class mocked_static_json:
        status_code: int = 400

        def json(self) -> dict:
            return PAYLOAD  # type: ignore

    return mocked_static_json()


def mocked_post_500(*args, **kwargs):  # noqa
    @dataclass
    class mocked_static_json:
        status_code: int = 500

        def json(self) -> dict:
            return PAYLOAD  # type: ignore

    return mocked_static_json()


def mocked_sse_client(*args, **kwargs):  # noqa
    @dataclass
    class Event:
        data: str

    @dataclass
    class mocked_static_events:
        def events(self) -> list:
            payload_json = json.dumps(PAYLOAD)
            return [
                Event(data=payload_json),
                Event(data=payload_json),
                Event(data=PAYLOAD_END),
            ]

    return mocked_static_events()


def mocked_get_endpoint_config():
    return ModelEndpointConfig(
        endpoint_name="test_endpoint_name",
        bundle_name="test_bundle_name",
    )


class MockRequest:
    """Mock request object for testing PassthroughForwarder"""

    def __init__(
        self,
        method="POST",
        path="/mcp/test",
        query="",
        headers=None,
        body_data=b'{"test": "data"}',
    ):
        self.method = method
        self.headers = headers or {
            "content-type": "application/json",
            "authorization": "Bearer token",
        }
        self.url = MagicMock()
        self.url.path = path
        self.url.query = query
        self._body_data = body_data

    async def body(self):
        return self._body_data


class MockContent:
    """Mock content object for aiohttp response"""

    def __init__(self, chunks=None):
        self.chunks = chunks or [b"chunk1", b"chunk2"]

    async def iter_chunks(self):
        """Mock async iterator for chunks"""
        for chunk in self.chunks:
            yield (chunk,)  # aiohttp yields tuples of (chunk, is_last)


class MockAiohttpResponse:
    """Mock aiohttp response for testing PassthroughForwarder"""

    def __init__(self, data=b'{"result": "success"}', headers=None, status=200, chunks=None):
        self.headers = headers or {"content-type": "application/json"}
        self.status = status
        self._data = data
        self.content = MockContent(chunks)

    async def read(self):
        return self._data


def mocked_aiohttp_client_session():
    """Mock aiohttp ClientSession for passthrough forwarder tests"""
    mock_response = MockAiohttpResponse()
    mock_client = AsyncMock()
    mock_client.request = AsyncMock(return_value=mock_response)
    return mock_client


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


def mocked_config_content():
    return {
        "forwarder": {
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
    }


def mocked_config_overrides():
    return [
        "forwarder.sync.extra_routes=['/v1/chat/completions']",
        "forwarder.stream.extra_routes=['/v1/chat/completions']",
        "forwarder.sync.healthcheck_route=/health",
        "forwarder.stream.healthcheck_route=/health",
    ]


# patch open(config_uri, "rt") and have output be mocked_config_content
@mock.patch("builtins.open", mock.mock_open(read_data=json.dumps(mocked_config_content())))
def test_load_named_config():
    output = load_named_config("dummy.yml", config_overrides=mocked_config_overrides())
    expected_output = {
        "name": "forwarder",
        "sync": {
            "user_port": 5005,
            "user_hostname": "localhost",
            "use_grpc": False,
            "predict_route": "/predict",
            "healthcheck_route": "/health",
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
            "healthcheck_route": "/health",
            "batch_route": None,
            "model_engine_unwrap": True,
            "serialize_results_as_string": False,
            "extra_routes": ["/v1/chat/completions"],
        },
        "max_concurrency": 42,
    }
    assert output == expected_output


@mock.patch("requests.post", mocked_post)
@mock.patch("requests.get", mocked_get)
def test_forwarders(post_inference_hooks_handler):
    fwd = Forwarder(
        "ignored",
        model_engine_unwrap=True,
        serialize_results_as_string=False,
        post_inference_hooks_handler=post_inference_hooks_handler,
        wrap_response=True,
        forward_http_status=True,
        forward_http_status_in_body=False,
    )
    json_response = fwd({"ignore": "me"})
    _check(json_response)


def _check(json_response) -> None:
    json_response = (
        json.loads(json_response.body.decode("utf-8"))
        if isinstance(json_response, JSONResponse)
        else json_response
    )
    assert json_response == {"result": PAYLOAD}


def _check_serialized_with_status_code_in_body(json_response, status_code: int) -> None:
    json_response = (
        json.loads(json_response.body.decode("utf-8"))
        if isinstance(json_response, JSONResponse)
        else json_response
    )
    assert isinstance(json_response["result"], str)
    assert (
        len(json_response) == 2
    ), f"expecting only 'result' and 'status_code' key, but got {json_response=}"
    assert json.loads(json_response["result"]) == PAYLOAD
    assert json_response["status_code"] == status_code


def _check_responses_not_wrapped(json_response) -> None:
    json_response = (
        json.loads(json_response.body.decode("utf-8"))
        if isinstance(json_response, JSONResponse)
        else json_response
    )
    assert json_response == PAYLOAD


def _check_streaming(streaming_response) -> None:
    streaming_response_list = list(streaming_response)
    assert len(streaming_response_list) == 3
    assert streaming_response_list[0] == {"result": PAYLOAD}
    assert streaming_response_list[1] == {"result": PAYLOAD}
    assert streaming_response_list[2] == {"result": PAYLOAD_END}


def _check_streaming_serialized(streaming_response) -> None:
    streaming_response_list = list(streaming_response)
    assert len(streaming_response_list) == 3
    assert streaming_response_list[0] == {"result": json.dumps(PAYLOAD)}
    assert streaming_response_list[1] == {"result": json.dumps(PAYLOAD)}
    assert streaming_response_list[2] == {"result": PAYLOAD_END}


async def _check_passthrough_response(passthrough_response_generator) -> None:
    """Check a passthrough forwarder response generator"""
    response_list = []
    async for response in passthrough_response_generator:
        response_list.append(response)

    # The refactored forward_stream yields:
    # 1. (headers, status) tuple
    # 2. chunks from content.iter_chunks()
    # 3. final read() result
    assert len(response_list) >= 3  # At least headers tuple + chunks + final read

    # First item should be (headers, status) tuple
    headers_and_status = response_list[0]
    assert isinstance(headers_and_status, tuple)
    assert len(headers_and_status) == 2
    headers, status = headers_and_status
    assert isinstance(headers, dict)
    assert status == 200

    # Last item should be the final read result
    final_data = response_list[-1]
    assert final_data == b'{"result": "success"}'

    # Middle items should be chunk data
    chunks = response_list[1:-1]
    assert len(chunks) == 2  # Our mock has 2 chunks
    assert chunks[0] == b"chunk1"
    assert chunks[1] == b"chunk2"


@mock.patch("requests.post", mocked_post)
@mock.patch("requests.get", mocked_get)
def test_forwarders_serialize_results_as_string(post_inference_hooks_handler):
    fwd = Forwarder(
        "ignored",
        model_engine_unwrap=True,
        serialize_results_as_string=True,
        post_inference_hooks_handler=post_inference_hooks_handler,
        wrap_response=True,
        forward_http_status=True,
        forward_http_status_in_body=False,
    )
    json_response = fwd({"ignore": "me"})
    _check_serialized(json_response)


def _check_serialized(json_response) -> None:
    json_response = (
        json.loads(json_response.body.decode("utf-8"))
        if isinstance(json_response, JSONResponse)
        else json_response
    )
    assert isinstance(json_response["result"], str)
    assert len(json_response) == 1, f"expecting only 'result' key, but got {json_response=}"
    assert json.loads(json_response["result"]) == PAYLOAD


@mock.patch("requests.post", mocked_post)
@mock.patch("requests.get", mocked_get)
def test_forwarders_override_serialize_results(post_inference_hooks_handler):
    fwd = Forwarder(
        "ignored",
        model_engine_unwrap=True,
        serialize_results_as_string=True,
        post_inference_hooks_handler=post_inference_hooks_handler,
        wrap_response=True,
        forward_http_status=True,
        forward_http_status_in_body=False,
    )
    json_response = fwd({"ignore": "me", KEY_SERIALIZE_RESULTS_AS_STRING: False})
    _check(json_response)

    fwd = Forwarder(
        "ignored",
        model_engine_unwrap=True,
        serialize_results_as_string=False,
        post_inference_hooks_handler=post_inference_hooks_handler,
        wrap_response=True,
        forward_http_status=True,
        forward_http_status_in_body=False,
    )
    json_response = fwd({"ignore": "me", KEY_SERIALIZE_RESULTS_AS_STRING: True})
    _check_serialized(json_response)


@mock.patch("requests.post", mocked_post)
@mock.patch("requests.get", mocked_get)
def test_forwarder_does_not_wrap_response(post_inference_hooks_handler):
    fwd = Forwarder(
        "ignored",
        model_engine_unwrap=True,
        serialize_results_as_string=False,
        post_inference_hooks_handler=post_inference_hooks_handler,
        wrap_response=False,
        forward_http_status=True,
        forward_http_status_in_body=False,
    )
    json_response = fwd({"ignore": "me"})
    _check_responses_not_wrapped(json_response)


@mock.patch("requests.post", mocked_post_500)
@mock.patch("requests.get", mocked_get)
def test_forwarder_return_status_code(post_inference_hooks_handler):
    fwd = Forwarder(
        "ignored",
        model_engine_unwrap=True,
        serialize_results_as_string=True,
        post_inference_hooks_handler=post_inference_hooks_handler,
        wrap_response=False,
        forward_http_status=True,
        forward_http_status_in_body=False,
    )
    json_response = fwd({"ignore": "me"})
    _check_responses_not_wrapped(json_response)
    assert json_response.status_code == 500


@mock.patch("requests.post", mocked_post_500)
@mock.patch("requests.get", mocked_get)
def test_forwarder_dont_return_status_code(post_inference_hooks_handler):
    fwd = Forwarder(
        "ignored",
        model_engine_unwrap=True,
        serialize_results_as_string=True,
        post_inference_hooks_handler=post_inference_hooks_handler,
        wrap_response=False,
        forward_http_status=False,
        forward_http_status_in_body=False,
    )
    json_response = fwd({"ignore": "me"})
    assert json_response == PAYLOAD


@mock.patch("requests.post", mocked_post_500)
@mock.patch("requests.get", mocked_get)
def test_forwarder_return_status_code_in_body(post_inference_hooks_handler):
    fwd = Forwarder(
        "ignored",
        model_engine_unwrap=True,
        serialize_results_as_string=True,
        post_inference_hooks_handler=post_inference_hooks_handler,
        wrap_response=True,
        forward_http_status=False,
        forward_http_status_in_body=True,
    )
    response = fwd({"ignore": "me"})
    _check_serialized_with_status_code_in_body(response, 500)


@mock.patch("requests.post", mocked_post)
@mock.patch("requests.get", mocked_get)
@mock.patch(
    "model_engine_server.inference.forwarding.forwarding.get_endpoint_config",
    mocked_get_endpoint_config,
)
def test_forwarder_loader():
    fwd = LoadForwarder(serialize_results_as_string=True).load(None, None)  # type: ignore
    json_response = fwd({"ignore": "me"})
    _check_serialized(json_response)

    fwd = LoadForwarder(serialize_results_as_string=False).load(None, None)  # type: ignore
    json_response = fwd({"ignore": "me"})
    _check(json_response)

    fwd = LoadForwarder(wrap_response=False).load(None, None)  # type: ignore
    json_response = fwd({"ignore": "me"})
    _check_responses_not_wrapped(json_response)


@mock.patch("requests.post", mocked_post)
@mock.patch("requests.get", mocked_get)
@mock.patch(
    "model_engine_server.inference.forwarding.forwarding.get_endpoint_config",
    mocked_get_endpoint_config,
)
def test_forwarder_loader_env_serialize_behavior(post_inference_hooks_handler):
    with environment(**{ENV_SERIALIZE_RESULTS_AS_STRING: "false"}):
        fwd = LoadForwarder(serialize_results_as_string=True).load(None, None)  # type: ignore
    json_response = fwd({"ignore": "me"})
    _check(json_response)

    with environment(**{ENV_SERIALIZE_RESULTS_AS_STRING: "true"}):
        fwd = LoadForwarder(serialize_results_as_string=False).load(None, None)  # type: ignore
    json_response = fwd({"ignore": "me"})
    _check_serialized(json_response)


@mock.patch("requests.post", mocked_post)
@mock.patch("requests.get", mocked_get)
def test_forwarder_serialize_within_args(post_inference_hooks_handler):
    # standard Launch-created forwarder
    fwd = Forwarder(
        "ignored",
        model_engine_unwrap=True,
        serialize_results_as_string=True,
        post_inference_hooks_handler=post_inference_hooks_handler,
        wrap_response=True,
        forward_http_status=True,
        forward_http_status_in_body=False,
    )
    # expected: no `serialize_results_as_string` at top-level nor in 'args'
    json_response = fwd({"something": "to ignore", "args": {"my": "payload", "is": "here"}})
    _check_serialized(json_response)
    # unwraps under "args" to find `serialize_results_as_string`
    payload = {
        "something": "to ignore",
        "args": {"my": "payload", "is": "here", "serialize_results_as_string": False},
    }
    json_response = fwd(payload)
    _check(json_response)
    # w/o unwrapping it won't "find" the `"serialize_results_as_string": False` directive
    fwd = Forwarder(
        "ignored",
        model_engine_unwrap=False,
        serialize_results_as_string=True,
        post_inference_hooks_handler=post_inference_hooks_handler,
        wrap_response=True,
        forward_http_status=True,
        forward_http_status_in_body=False,
    )
    json_response = fwd(payload)
    _check_serialized(json_response)


@mock.patch("requests.post", mocked_post)
@mock.patch("requests.get", mocked_get)
@mock.patch("sseclient.SSEClient", mocked_sse_client)
def test_streaming_forwarders(post_inference_hooks_handler):
    fwd = StreamingForwarder(
        "ignored",
        model_engine_unwrap=True,
        serialize_results_as_string=False,
        post_inference_hooks_handler=post_inference_hooks_handler,
    )
    response = fwd({"ignore": "me"})
    _check_streaming(response)


@mock.patch("requests.post", mocked_post_400)
@mock.patch("requests.get", mocked_get)
@mock.patch("sseclient.SSEClient", mocked_sse_client)
def test_streaming_forwarder_400_upstream(post_inference_hooks_handler):
    fwd = StreamingForwarder(
        "ignored",
        model_engine_unwrap=True,
        serialize_results_as_string=False,
        post_inference_hooks_handler=post_inference_hooks_handler,
    )
    with pytest.raises(HTTPException) as e:
        fwd({"ignore": "me"})

    assert e.value.status_code == 400


@mock.patch("requests.post", mocked_post)
@mock.patch("requests.get", mocked_get)
@mock.patch("sseclient.SSEClient", mocked_sse_client)
@mock.patch(
    "model_engine_server.inference.forwarding.forwarding.get_endpoint_config",
    mocked_get_endpoint_config,
)
def test_streaming_forwarder_loader():
    fwd = LoadStreamingForwarder(serialize_results_as_string=True).load(None, None)  # type: ignore
    json_response = fwd({"ignore": "me"})
    _check_streaming_serialized(json_response)

    fwd = LoadStreamingForwarder(serialize_results_as_string=False).load(None, None)  # type: ignore
    response = fwd({"ignore": "me"})
    _check_streaming(response)


@pytest.mark.asyncio
async def test_streaming_forward_relays_chunks():
    """Async StreamingForwarder.forward() relays each upstream SSE chunk unchanged."""
    endpoint = "http://localhost:5005/stream"
    fwd = StreamingForwarder(
        endpoint,
        model_engine_unwrap=False,
        serialize_results_as_string=False,
        post_inference_hooks_handler=None,
    )
    sse_body = f"data: {json.dumps(PAYLOAD)}\n\ndata: {PAYLOAD_END}\n\n"
    with aioresponses() as aio_mock:
        aio_mock.post(
            endpoint, status=200, body=sse_body, headers={"content-type": "text/event-stream"}
        )
        chunks = [chunk async for chunk in fwd.forward(dict(PAYLOAD))]
    assert chunks == [{"result": PAYLOAD}, {"result": PAYLOAD_END}]


# Passthrough forwarder tests
@pytest.mark.asyncio
async def test_passthrough_forwarder():
    """Test basic PassthroughForwarder functionality"""
    fwd = PassthroughForwarder(passthrough_endpoint="http://localhost:5005/mcp/test")
    mock_request = MockRequest(method="POST", path="/mcp/test", query="param=value")

    with mock.patch("aiohttp.ClientSession") as mock_session:
        mock_client = mocked_aiohttp_client_session()
        mock_session.return_value.__aenter__.return_value = mock_client

        response_generator = fwd.forward_stream(mock_request)
        await _check_passthrough_response(response_generator)

        # Verify the correct endpoint was called
        mock_client.request.assert_called_once_with(
            method="POST",
            url="http://localhost:5005/mcp/test?param=value",
            data=b'{"test": "data"}',
            headers={
                "content-type": "application/json",
                "authorization": "Bearer token",
            },
        )


@pytest.mark.asyncio
async def test_passthrough_forwarder_get_request():
    """Test PassthroughForwarder with GET request (no body)"""
    fwd = PassthroughForwarder(passthrough_endpoint="http://localhost:5005/mcp/status")
    mock_request = MockRequest(method="GET", path="/mcp/status", query="", body_data=b"")

    with mock.patch("aiohttp.ClientSession") as mock_session:
        mock_client = mocked_aiohttp_client_session()
        mock_session.return_value.__aenter__.return_value = mock_client

        response_generator = fwd.forward_stream(mock_request)
        await _check_passthrough_response(response_generator)

        # Verify GET request has no data
        mock_client.request.assert_called_once_with(
            method="GET",
            url="http://localhost:5005/mcp/status",
            data=None,
            headers={
                "content-type": "application/json",
                "authorization": "Bearer token",
            },
        )


@pytest.mark.asyncio
async def test_passthrough_forwarder_header_filtering():
    """Test that PassthroughForwarder filters out excluded headers"""
    fwd = PassthroughForwarder(passthrough_endpoint="http://localhost:5005")

    # Include both allowed and excluded headers
    headers_with_excluded = {
        "content-type": "application/json",
        "authorization": "Bearer token",
        "host": "original-host",  # Should be excluded
        "content-length": "123",  # Should be excluded
        "connection": "keep-alive",  # Should be excluded
        "custom-header": "keep-me",  # Should be kept
    }

    mock_request = MockRequest(method="POST", path="/mcp/test", headers=headers_with_excluded)

    with mock.patch("aiohttp.ClientSession") as mock_session:
        mock_client = mocked_aiohttp_client_session()
        mock_session.return_value.__aenter__.return_value = mock_client

        response_generator = fwd.forward_stream(mock_request)
        await _check_passthrough_response(response_generator)

        # Check that excluded headers were filtered out
        call_args = mock_client.request.call_args
        actual_headers = call_args.kwargs["headers"]

        expected_headers = {
            "content-type": "application/json",
            "authorization": "Bearer token",
            "custom-header": "keep-me",
        }

        assert actual_headers == expected_headers
        assert "host" not in actual_headers
        assert "content-length" not in actual_headers
        assert "connection" not in actual_headers


@mock.patch("requests.get", mocked_get)
def test_load_passthrough_forwarder():
    """Test LoadPassthroughForwarder.load() method"""
    loader = LoadPassthroughForwarder(
        user_port=5005,
        user_hostname="localhost",
        healthcheck_route="/health",
    )

    forwarder = loader.load(None, None)  # type: ignore

    assert isinstance(forwarder, PassthroughForwarder)


def test_load_passthrough_forwarder_validation():
    """Test LoadPassthroughForwarder validation similar to streaming tests"""
    # Test invalid port
    with pytest.raises(ValueError, match="Invalid port value"):
        LoadPassthroughForwarder(user_port=0).load(None, None)  # type: ignore

    # Test empty hostname
    with pytest.raises(ValueError, match="hostname must be non-empty"):
        LoadPassthroughForwarder(user_hostname="").load(None, None)  # type: ignore

    # Test non-localhost hostname
    with pytest.raises(NotImplementedError, match="localhost-based user-code services"):
        LoadPassthroughForwarder(user_hostname="remote-host").load(None, None)  # type: ignore

    # Test empty healthcheck route
    with pytest.raises(ValueError, match="healthcheck route must be non-empty"):
        LoadPassthroughForwarder(healthcheck_route="").load(None, None)  # type: ignore


# ---- Forwarder timing instrumentation ----


class FakeInferenceMonitoringMetricsGateway(InferenceMonitoringMetricsGateway):
    """Records emitted timing metrics so tests can assert on them."""

    def __init__(self):
        self.timing_calls: List[Tuple[str, float, List[str]]] = []

    def emit_attempted_post_inference_hook(self, hook: str):
        pass

    def emit_successful_post_inference_hook(self, hook: str):
        pass

    def emit_async_task_received_metric(self, queue_name: str):
        pass

    def emit_async_task_stuck_metric(self, queue_name: str):
        pass

    def emit_timing_metric(self, metric_name: str, value_ms: float, tags: List[str]):
        self.timing_calls.append((metric_name, value_ms, tags))

    def by_name(self) -> Mapping[str, Tuple[float, List[str]]]:
        return {name: (value_ms, tags) for name, value_ms, tags in self.timing_calls}


SYNC_ROUTE = "/v1/chat/completions"
SYNC_ENDPOINT = f"http://localhost:5005{SYNC_ROUTE}"
SYNC_TAGS = ["request_type:sync", f"route:{SYNC_ROUTE}", "endpoint_name:test-endpoint"]
STREAM_TAGS = ["request_type:stream", f"route:{SYNC_ROUTE}", "endpoint_name:test-endpoint"]


@pytest.mark.asyncio
async def test_forward_emits_sync_timing_metrics():
    metrics_gateway = FakeInferenceMonitoringMetricsGateway()
    fwd = Forwarder(
        SYNC_ENDPOINT,
        model_engine_unwrap=False,
        serialize_results_as_string=False,
        post_inference_hooks_handler=None,
        wrap_response=True,
        forward_http_status=False,
        forward_http_status_in_body=False,
        monitoring_metrics_gateway=metrics_gateway,
        metric_tags=SYNC_TAGS,
    )

    # Scripted clock: t0=0, t_vllm_start=1, vllm end=3, total end=5 (seconds).
    # roundtrip brackets only the post+json (1->3); total brackets entry->exit (0->5).
    with (
        aioresponses() as aio_mock,
        mock.patch(
            "model_engine_server.inference.forwarding.forwarding.perf_counter",
            side_effect=[0.0, 1.0, 3.0, 5.0],
        ),
    ):
        aio_mock.post(SYNC_ENDPOINT, status=200, payload=dict(PAYLOAD))
        response = await fwd.forward(dict(PAYLOAD))

    assert response == {"result": PAYLOAD}  # happy-path response unchanged

    emitted = metrics_gateway.by_name()
    assert emitted[FORWARDER_TOTAL_METRIC][0] == 5000.0
    assert emitted[FORWARDER_VLLM_ROUNDTRIP_METRIC][0] == 2000.0
    assert emitted[FORWARDER_OVERHEAD_METRIC][0] == 3000.0
    # overhead == total - roundtrip
    assert (
        emitted[FORWARDER_OVERHEAD_METRIC][0]
        == emitted[FORWARDER_TOTAL_METRIC][0] - emitted[FORWARDER_VLLM_ROUNDTRIP_METRIC][0]
    )
    assert emitted[FORWARDER_OVERHEAD_METRIC][1] == SYNC_TAGS


@pytest.mark.asyncio
async def test_forward_emits_streaming_ttft_metrics():
    metrics_gateway = FakeInferenceMonitoringMetricsGateway()
    fwd = StreamingForwarder(
        SYNC_ENDPOINT,
        model_engine_unwrap=False,
        serialize_results_as_string=False,
        post_inference_hooks_handler=None,
        monitoring_metrics_gateway=metrics_gateway,
        metric_tags=STREAM_TAGS,
    )

    sse_body = 'data: {"hello": "world"}\n\ndata: [DONE]\n\n'
    # Scripted clock: t0=0, t_vllm_start=1, first-event=4 (seconds).
    with (
        aioresponses() as aio_mock,
        mock.patch(
            "model_engine_server.inference.forwarding.forwarding.perf_counter",
            side_effect=[0.0, 1.0, 4.0],
        ),
    ):
        aio_mock.post(
            SYNC_ENDPOINT,
            status=200,
            body=sse_body,
            headers={"content-type": "text/event-stream"},
        )
        chunks = [chunk async for chunk in fwd.forward(dict(PAYLOAD))]

    assert len(chunks) == 2  # sanity: stream was driven (contents asserted elsewhere)

    emitted = metrics_gateway.by_name()
    assert emitted[FORWARDER_TTFT_METRIC][0] == 4000.0
    assert emitted[FORWARDER_VLLM_TTFT_METRIC][0] == 3000.0
    assert emitted[FORWARDER_OVERHEAD_METRIC][0] == 1000.0
    # overhead == forwarder_ttft - vllm_ttft
    assert (
        emitted[FORWARDER_OVERHEAD_METRIC][0]
        == emitted[FORWARDER_TTFT_METRIC][0] - emitted[FORWARDER_VLLM_TTFT_METRIC][0]
    )
    assert emitted[FORWARDER_OVERHEAD_METRIC][1] == STREAM_TAGS


@pytest.mark.asyncio
async def test_forward_without_metrics_gateway_is_noop():
    """A forwarder with no metrics gateway must still serve the happy path."""
    fwd = Forwarder(
        SYNC_ENDPOINT,
        model_engine_unwrap=False,
        serialize_results_as_string=False,
        post_inference_hooks_handler=None,
        wrap_response=True,
        forward_http_status=False,
        forward_http_status_in_body=False,
        monitoring_metrics_gateway=None,
    )
    with aioresponses() as aio_mock:
        aio_mock.post(SYNC_ENDPOINT, status=200, payload=dict(PAYLOAD))
        response = await fwd.forward(dict(PAYLOAD))
    assert response == {"result": PAYLOAD}


@mock.patch("requests.post", mocked_post)
@mock.patch("requests.get", mocked_get)
@mock.patch(
    "model_engine_server.inference.forwarding.forwarding.get_endpoint_config",
    mocked_get_endpoint_config,
)
def test_loaders_set_metrics_gateway_and_tags():
    fwd = LoadForwarder().load(None, None)  # type: ignore
    assert fwd.monitoring_metrics_gateway is not None
    assert fwd.metric_tags == [
        "request_type:sync",
        "route:/predict",
        "endpoint_name:test_endpoint_name",
    ]

    stream_fwd = LoadStreamingForwarder().load(None, None)  # type: ignore
    assert stream_fwd.monitoring_metrics_gateway is not None
    assert stream_fwd.metric_tags == [
        "request_type:stream",
        "route:/predict",
        "endpoint_name:test_endpoint_name",
    ]


@mock.patch("requests.post", mocked_post)
@mock.patch("requests.get", mocked_get)
@mock.patch(
    "model_engine_server.inference.forwarding.forwarding.get_endpoint_config",
    side_effect=RuntimeError("no endpoint config"),
)
def test_loader_survives_endpoint_config_failure(_mock_config):
    # Even when endpoint config (and thus the hooks handler) fails, the forwarder still gets a
    # metrics gateway; tags degrade to omit endpoint_name.
    fwd = LoadForwarder().load(None, None)  # type: ignore
    assert fwd.monitoring_metrics_gateway is not None
    assert fwd.post_inference_hooks_handler is None
    assert fwd.metric_tags == ["request_type:sync", "route:/predict"]
