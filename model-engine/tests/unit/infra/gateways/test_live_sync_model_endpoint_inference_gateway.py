import json
from dataclasses import dataclass
from typing import Any, Dict, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from model_engine_server.common.dtos.tasks import (
    SyncEndpointPredictV1Request,
    SyncEndpointPredictV1Response,
)
from model_engine_server.domain.exceptions import InvalidRequestException, UpstreamServiceError
from model_engine_server.domain.gateways.monitoring_metrics_gateway import MonitoringMetricsGateway
from model_engine_server.infra.gateways.live_sync_model_endpoint_inference_gateway import (
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


def _get_mock_client_session_with_client_connector_error():
    mock_post = AsyncMock(
        side_effect=aiohttp.ClientConnectorError(connection_key=None, os_error=OSError())
    )
    mock_client_session_val = AsyncMock()
    mock_client_session_val.post = mock_post
    mock_client_session_val.__aenter__ = AsyncMock(return_value=mock_client_session_val)

    async def _aexit(*exc):
        pass

    mock_client_session_val.__aexit__ = AsyncMock(side_effect=_aexit)
    mock_client_session = MagicMock(return_value=mock_client_session_val)
    return mock_client_session


@pytest.mark.asyncio
async def test_make_request_with_retries_success(
    fake_monitoring_metrics_gateway: MonitoringMetricsGateway,
):
    gateway = LiveSyncModelEndpointInferenceGateway(
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway, use_asyncio=True
    )

    fake_response = FakeResponse(status=200)
    mock_client_session = _get_mock_client_session(fake_response)

    with patch(
        "model_engine_server.infra.gateways.live_sync_model_endpoint_inference_gateway.aiohttp.ClientSession",
        mock_client_session,
    ):
        response = await gateway.make_request_with_retries(
            "test_request_url", {}, 0.05, 2, "test_endpoint_name"
        )
    assert response == {"test_key": "test_value"}


@pytest.mark.asyncio
async def test_make_request_with_retries_failed_429(
    fake_monitoring_metrics_gateway: MonitoringMetricsGateway,
):
    gateway = LiveSyncModelEndpointInferenceGateway(
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway, use_asyncio=True
    )

    fake_response = FakeResponse(status=429)
    mock_client_session = _get_mock_client_session(fake_response)

    with pytest.raises(UpstreamServiceError), patch(
        "model_engine_server.infra.gateways.live_sync_model_endpoint_inference_gateway.aiohttp.ClientSession",
        mock_client_session,
    ):
        await gateway.make_request_with_retries(
            "test_request_url", {}, 0.05, 2, "test_endpoint_name"
        )


@pytest.mark.asyncio
async def test_make_request_with_retries_failed_traceback(
    fake_monitoring_metrics_gateway: MonitoringMetricsGateway,
):
    gateway = LiveSyncModelEndpointInferenceGateway(
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway, use_asyncio=True
    )

    fake_response = FakeResponse(status=500)
    mock_client_session = _get_mock_client_session(fake_response)

    with pytest.raises(UpstreamServiceError), patch(
        "model_engine_server.infra.gateways.live_sync_model_endpoint_inference_gateway.aiohttp.ClientSession",
        mock_client_session,
    ):
        await gateway.make_request_with_retries(
            "test_request_url", {}, 0.05, 2, "test_endpoint_name"
        )


@pytest.mark.asyncio
async def test_make_request_with_retries_failed_with_client_connector_error(
    fake_monitoring_metrics_gateway: MonitoringMetricsGateway,
):
    gateway = LiveSyncModelEndpointInferenceGateway(
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway, use_asyncio=True
    )

    mock_client_session = _get_mock_client_session_with_client_connector_error()

    with pytest.raises(UpstreamServiceError), patch(
        "model_engine_server.infra.gateways.live_sync_model_endpoint_inference_gateway.aiohttp.ClientSession",
        mock_client_session,
    ):
        await gateway.make_request_with_retries(
            "test_request_url", {}, 0.05, 2, "test_endpoint_name"
        )


@pytest.mark.asyncio
async def test_predict_success(
    sync_endpoint_predict_request_1: Tuple[SyncEndpointPredictV1Request, Dict[str, Any]],
    fake_monitoring_metrics_gateway: MonitoringMetricsGateway,
):
    gateway = LiveSyncModelEndpointInferenceGateway(
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway, use_asyncio=True
    )

    fake_response = FakeResponse(status=200, body={"test_key": "test_value"})
    mock_client_session = _get_mock_client_session(fake_response)
    with patch(
        "model_engine_server.infra.gateways.live_sync_model_endpoint_inference_gateway.aiohttp.ClientSession",
        mock_client_session,
    ):
        response = await gateway.predict(
            topic="test_topic",
            predict_request=sync_endpoint_predict_request_1[0],
            endpoint_name="test_name",
        )
        assert isinstance(response, SyncEndpointPredictV1Response)
        assert response.dict() == {
            "status": "SUCCESS",
            "result": {"test_key": "test_value"},
            "traceback": None,
        }


@pytest.mark.asyncio
async def test_predict_raises_traceback_json(
    sync_endpoint_predict_request_1: Tuple[SyncEndpointPredictV1Request, Dict[str, Any]],
    fake_monitoring_metrics_gateway: MonitoringMetricsGateway,
):
    gateway = LiveSyncModelEndpointInferenceGateway(
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway, use_asyncio=True
    )

    content = json.dumps({"detail": {"traceback": "test_traceback"}}).encode("utf-8")
    fake_response = FakeResponse(status=500, content=content)
    mock_client_session = _get_mock_client_session(fake_response)
    with patch(
        "model_engine_server.infra.gateways.live_sync_model_endpoint_inference_gateway.aiohttp.ClientSession",
        mock_client_session,
    ):
        response = await gateway.predict(
            topic="test_topic",
            predict_request=sync_endpoint_predict_request_1[0],
            endpoint_name="test_name",
        )
        assert isinstance(response, SyncEndpointPredictV1Response)
        assert response.dict() == {
            "status": "FAILURE",
            "result": None,
            "traceback": "test_traceback",
        }


@pytest.mark.asyncio
async def test_predict_raises_traceback_not_json(
    sync_endpoint_predict_request_1: Tuple[SyncEndpointPredictV1Request, Dict[str, Any]],
    fake_monitoring_metrics_gateway: MonitoringMetricsGateway,
):
    gateway = LiveSyncModelEndpointInferenceGateway(
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway, use_asyncio=True
    )

    content = b"Test traceback content"
    fake_response = FakeResponse(status=500, content=content)
    mock_client_session = _get_mock_client_session(fake_response)
    with patch(
        "model_engine_server.infra.gateways.live_sync_model_endpoint_inference_gateway.aiohttp.ClientSession",
        mock_client_session,
    ):
        response = await gateway.predict(
            topic="test_topic",
            predict_request=sync_endpoint_predict_request_1[0],
            endpoint_name="test_name",
        )
        assert isinstance(response, SyncEndpointPredictV1Response)
        assert response.dict() == {
            "status": "FAILURE",
            "result": None,
            "traceback": "Test traceback content",
        }


@pytest.mark.asyncio
async def test_predict_raises_traceback_wrapped(
    sync_endpoint_predict_request_1: Tuple[SyncEndpointPredictV1Request, Dict[str, Any]],
    fake_monitoring_metrics_gateway: MonitoringMetricsGateway,
):
    gateway = LiveSyncModelEndpointInferenceGateway(
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway, use_asyncio=True
    )

    content = json.dumps(
        {"result": json.dumps({"detail": {"traceback": "test_traceback"}})}
    ).encode("utf-8")
    fake_response = FakeResponse(status=500, content=content)
    mock_client_session = _get_mock_client_session(fake_response)
    with patch(
        "model_engine_server.infra.gateways.live_sync_model_endpoint_inference_gateway.aiohttp.ClientSession",
        mock_client_session,
    ):
        response = await gateway.predict(
            topic="test_topic",
            predict_request=sync_endpoint_predict_request_1[0],
            endpoint_name="test_name",
        )
        assert isinstance(response, SyncEndpointPredictV1Response)
        assert response.dict() == {
            "status": "FAILURE",
            "result": None,
            "traceback": "test_traceback",
        }


@pytest.mark.asyncio
async def test_predict_raises_traceback_wrapped_detail_array(
    sync_endpoint_predict_request_1: Tuple[SyncEndpointPredictV1Request, Dict[str, Any]],
    fake_monitoring_metrics_gateway: MonitoringMetricsGateway,
):
    gateway = LiveSyncModelEndpointInferenceGateway(
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway, use_asyncio=True
    )

    content = json.dumps({"result": json.dumps({"detail": [{"error": "error"}]})}).encode("utf-8")
    fake_response = FakeResponse(status=500, content=content)
    mock_client_session = _get_mock_client_session(fake_response)
    with patch(
        "model_engine_server.infra.gateways.live_sync_model_endpoint_inference_gateway.aiohttp.ClientSession",
        mock_client_session,
    ):
        response = await gateway.predict(
            topic="test_topic",
            predict_request=sync_endpoint_predict_request_1[0],
            endpoint_name="test_name",
        )
        assert isinstance(response, SyncEndpointPredictV1Response)
        assert response.dict() == {
            "status": "FAILURE",
            "result": None,
            "traceback": """{"detail":[{"error":"error"}]}""",
        }


@pytest.mark.asyncio
async def test_predict_upstream_raises_400(
    sync_endpoint_predict_request_1: Tuple[SyncEndpointPredictV1Request, Dict[str, Any]],
    fake_monitoring_metrics_gateway: MonitoringMetricsGateway,
):
    gateway = LiveSyncModelEndpointInferenceGateway(
        monitoring_metrics_gateway=fake_monitoring_metrics_gateway, use_asyncio=True
    )

    content = json.dumps({"result": json.dumps({"error": "error"})}).encode("utf-8")
    fake_response = FakeResponse(status=400, content=content)
    mock_client_session = _get_mock_client_session(fake_response)
    with patch(
        "model_engine_server.infra.gateways.live_sync_model_endpoint_inference_gateway.aiohttp.ClientSession",
        mock_client_session,
    ):
        # assert that the exception is raised
        with pytest.raises(InvalidRequestException):
            await gateway.predict(
                topic="test_topic",
                predict_request=sync_endpoint_predict_request_1[0],
                endpoint_name="test_name",
            )
