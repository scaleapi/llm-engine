import json
from dataclasses import dataclass
from typing import Mapping
from unittest import mock

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from model_engine_server.core.utils.env import environment
from model_engine_server.domain.entities import ModelEndpointConfig
from model_engine_server.inference.forwarding.forwarding import (
    ENV_SERIALIZE_RESULTS_AS_STRING,
    KEY_SERIALIZE_RESULTS_AS_STRING,
    Forwarder,
    LoadForwarder,
    LoadStreamingForwarder,
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
