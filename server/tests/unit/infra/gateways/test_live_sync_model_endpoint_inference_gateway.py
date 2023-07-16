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
from llm_engine_server.infra.gateways.live_sync_model_endpoint_inference_gateway import (
    LiveSyncModelEndpointInferenceGateway,
)


@dataclass
class FakeResponse:
    status: int
    content: bytes = b"test_content"
    body: Any = None

    async def read(self):
        return self.content

    async def json(self):
        return self.body if self.body else {"test_key": "test_value"}


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
    gateway = LiveSyncModelEndpointInferenceGateway(use_asyncio=True)

    fake_response = FakeResponse(status=200)
    mock_client_session = _get_mock_client_session(fake_response)

    with patch(
        "llm_engine_server.infra.gateways.live_sync_model_endpoint_inference_gateway.aiohttp.ClientSession",
        mock_client_session,
    ):
        response = await gateway.make_request_with_retries("test_request_url", {}, 0.05, 2)
    assert response == {"test_key": "test_value"}


@pytest.mark.asyncio
async def test_make_request_with_retries_failed_429():
    gateway = LiveSyncModelEndpointInferenceGateway(use_asyncio=True)

    fake_response = FakeResponse(status=429)
    mock_client_session = _get_mock_client_session(fake_response)

    with pytest.raises(UpstreamServiceError), patch(
        "llm_engine_server.infra.gateways.live_sync_model_endpoint_inference_gateway.aiohttp.ClientSession",
        mock_client_session,
    ):
        await gateway.make_request_with_retries("test_request_url", {}, 0.05, 2)


@pytest.mark.asyncio
async def test_make_request_with_retries_failed_traceback():
    gateway = LiveSyncModelEndpointInferenceGateway(use_asyncio=True)

    fake_response = FakeResponse(status=500)
    mock_client_session = _get_mock_client_session(fake_response)

    with pytest.raises(UpstreamServiceError), patch(
        "llm_engine_server.infra.gateways.live_sync_model_endpoint_inference_gateway.aiohttp.ClientSession",
        mock_client_session,
    ):
        await gateway.make_request_with_retries("test_request_url", {}, 0.05, 2)


@pytest.mark.asyncio
async def test_predict_success(
    endpoint_predict_request_1: Tuple[EndpointPredictV1Request, Dict[str, Any]]
):
    gateway = LiveSyncModelEndpointInferenceGateway(use_asyncio=True)

    fake_response = FakeResponse(status=200, body={"test_key": "test_value"})
    mock_client_session = _get_mock_client_session(fake_response)
    with patch(
        "llm_engine_server.infra.gateways.live_sync_model_endpoint_inference_gateway.aiohttp.ClientSession",
        mock_client_session,
    ):
        response = await gateway.predict(
            topic="test_topic", predict_request=endpoint_predict_request_1[0]
        )
        assert isinstance(response, SyncEndpointPredictV1Response)
        assert response.dict() == {
            "status": "SUCCESS",
            "result": {"test_key": "test_value"},
            "traceback": None,
        }


@pytest.mark.asyncio
async def test_predict_raises_traceback_json(
    endpoint_predict_request_1: Tuple[EndpointPredictV1Request, Dict[str, Any]]
):
    gateway = LiveSyncModelEndpointInferenceGateway(use_asyncio=True)

    content = json.dumps({"detail": {"traceback": "test_traceback"}}).encode("utf-8")
    fake_response = FakeResponse(status=500, content=content)
    mock_client_session = _get_mock_client_session(fake_response)
    with patch(
        "llm_engine_server.infra.gateways.live_sync_model_endpoint_inference_gateway.aiohttp.ClientSession",
        mock_client_session,
    ):
        response = await gateway.predict(
            topic="test_topic", predict_request=endpoint_predict_request_1[0]
        )
        assert isinstance(response, SyncEndpointPredictV1Response)
        assert response.dict() == {
            "status": "FAILURE",
            "result": None,
            "traceback": "test_traceback",
        }


@pytest.mark.asyncio
async def test_predict_raises_traceback_not_json(
    endpoint_predict_request_1: Tuple[EndpointPredictV1Request, Dict[str, Any]]
):
    gateway = LiveSyncModelEndpointInferenceGateway(use_asyncio=True)

    content = b"Test traceback content"
    fake_response = FakeResponse(status=500, content=content)
    mock_client_session = _get_mock_client_session(fake_response)
    with patch(
        "llm_engine_server.infra.gateways.live_sync_model_endpoint_inference_gateway.aiohttp.ClientSession",
        mock_client_session,
    ):
        response = await gateway.predict(
            topic="test_topic", predict_request=endpoint_predict_request_1[0]
        )
        assert isinstance(response, SyncEndpointPredictV1Response)
        assert response.dict() == {
            "status": "FAILURE",
            "result": None,
            "traceback": "Test traceback content",
        }
