import json
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llm_engine_server.common.dtos.tasks import (
    EndpointPredictV1Request,
    SyncEndpointPredictV1Response,
)
from llm_engine_server.domain.exceptions import UpstreamServiceError
from llm_engine_server.infra.gateways.live_streaming_model_endpoint_inference_gateway import (
    LiveStreamingModelEndpointInferenceGateway,
)


@dataclass
class FakeIterator:
    content: bytes = b'{"test": "content"}'
    count: int = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.count = self.count + 1
        if self.count == 1:
            return b"data: " + self.content
        if self.count in {2, 3}:
            return b"\n"
        if self.count == 4:
            raise StopAsyncIteration


@dataclass
class FakeResponse:
    def __init__(self, status: int, message_content: bytes = b'{"test": "content"}'):
        self.status = status
        self.message_content = message_content
        self.content = FakeIterator(content=message_content)

    async def read(self):
        return self.message_content


def _get_mock_client_session(fake_response: FakeResponse):
    mock_post = AsyncMock(return_value=fake_response)
    mock_client_session_val = AsyncMock()
    mock_client_session_val.post = mock_post
    mock_client_session_val.__aenter__ = AsyncMock(return_value=mock_client_session_val)
    mock_client_session_val.__aexit__ = AsyncMock()
    mock_client_session = MagicMock(return_value=mock_client_session_val)
    return mock_client_session


@pytest.mark.asyncio
async def test_make_request_with_retries_success():
    gateway = LiveStreamingModelEndpointInferenceGateway(use_asyncio=True)

    fake_response = FakeResponse(status=200)
    mock_client_session = _get_mock_client_session(fake_response)

    with patch(
        "llm_engine_server.infra.gateways.live_streaming_model_endpoint_inference_gateway.aiohttp.ClientSession",
        mock_client_session,
    ):
        response = gateway.make_request_with_retries("test_request_url", {}, 0.05, 2)
        count = 0
        async for message in response:
            assert message == {"test": "content"}
            count += 1
        assert count == 1


@pytest.mark.asyncio
async def test_make_request_with_retries_failed_429():
    gateway = LiveStreamingModelEndpointInferenceGateway(use_asyncio=True)

    fake_response = FakeResponse(status=429)
    mock_client_session = _get_mock_client_session(fake_response)

    with pytest.raises(UpstreamServiceError), patch(
        "llm_engine_server.infra.gateways.live_streaming_model_endpoint_inference_gateway.aiohttp.ClientSession",
        mock_client_session,
    ):
        async for response in gateway.make_request_with_retries("test_request_url", {}, 0.05, 2):
            response


@pytest.mark.asyncio
async def test_make_request_with_retries_failed_traceback():
    gateway = LiveStreamingModelEndpointInferenceGateway(use_asyncio=True)

    fake_response = FakeResponse(status=500)
    mock_client_session = _get_mock_client_session(fake_response)

    with pytest.raises(UpstreamServiceError), patch(
        "llm_engine_server.infra.gateways.live_streaming_model_endpoint_inference_gateway.aiohttp.ClientSession",
        mock_client_session,
    ):
        async for response in gateway.make_request_with_retries("test_request_url", {}, 0.05, 2):
            response


@pytest.mark.asyncio
async def test_streaming_predict_success(
    endpoint_predict_request_1: Tuple[EndpointPredictV1Request, Dict[str, Any]]
):
    gateway = LiveStreamingModelEndpointInferenceGateway(use_asyncio=True)

    fake_response = FakeResponse(status=200)
    mock_client_session = _get_mock_client_session(fake_response)
    with patch(
        "llm_engine_server.infra.gateways.live_streaming_model_endpoint_inference_gateway.aiohttp.ClientSession",
        mock_client_session,
    ):
        response = gateway.streaming_predict(
            topic="test_topic", predict_request=endpoint_predict_request_1[0]
        )
        count = 0
        async for message in response:
            assert isinstance(message, SyncEndpointPredictV1Response)
            assert message.dict() == {
                "status": "SUCCESS",
                "result": {"test": "content"},
                "traceback": None,
            }
            count += 1
        assert count == 1


@pytest.mark.asyncio
async def test_predict_raises_traceback_json(
    endpoint_predict_request_1: Tuple[EndpointPredictV1Request, Dict[str, Any]]
):
    gateway = LiveStreamingModelEndpointInferenceGateway(use_asyncio=True)

    content = json.dumps({"detail": {"traceback": "test_traceback"}}).encode("utf-8")
    fake_response = FakeResponse(status=500, message_content=content)
    mock_client_session = _get_mock_client_session(fake_response)
    with patch(
        "llm_engine_server.infra.gateways.live_streaming_model_endpoint_inference_gateway.aiohttp.ClientSession",
        mock_client_session,
    ):
        response = gateway.streaming_predict(
            topic="test_topic", predict_request=endpoint_predict_request_1[0]
        )
        count = 0
        async for message in response:
            assert isinstance(message, SyncEndpointPredictV1Response)
            assert message.dict() == {
                "status": "FAILURE",
                "result": None,
                "traceback": "test_traceback",
            }
            count += 1
        assert count == 1


@pytest.mark.asyncio
async def test_predict_raises_traceback_not_json(
    endpoint_predict_request_1: Tuple[EndpointPredictV1Request, Dict[str, Any]]
):
    gateway = LiveStreamingModelEndpointInferenceGateway(use_asyncio=True)

    content = b"Test traceback content"
    fake_response = FakeResponse(status=500, message_content=content)
    mock_client_session = _get_mock_client_session(fake_response)
    with patch(
        "llm_engine_server.infra.gateways.live_streaming_model_endpoint_inference_gateway.aiohttp.ClientSession",
        mock_client_session,
    ):
        response = gateway.streaming_predict(
            topic="test_topic", predict_request=endpoint_predict_request_1[0]
        )
        count = 0
        async for message in response:
            assert isinstance(message, SyncEndpointPredictV1Response)
            assert message.dict() == {
                "status": "FAILURE",
                "result": None,
                "traceback": "Test traceback content",
            }
            count += 1
        assert count == 1
