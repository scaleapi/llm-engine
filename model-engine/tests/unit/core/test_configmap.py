"""Unit tests for configmap.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kubernetes_asyncio.client.rest import ApiException
from kubernetes_asyncio.config.config_exception import ConfigException

from model_engine_server.core.configmap import read_config_map


@pytest.fixture
def mock_config_map_data():
    return {"key1": "value1", "key2": "value2"}


@pytest.mark.asyncio
async def test_read_config_map_incluster(mock_config_map_data):
    """Test read_config_map succeeds using in-cluster config."""
    mock_cm = MagicMock()
    mock_cm.data = mock_config_map_data

    with patch("model_engine_server.core.configmap.client.Configuration") as mock_cfg_cls, patch(
        "model_engine_server.core.configmap.kube_config.load_incluster_config"
    ) as mock_incluster, patch(
        "model_engine_server.core.configmap.client.ApiClient"
    ) as mock_api_client_cls, patch(
        "model_engine_server.core.configmap.client.CoreV1Api"
    ) as mock_core_v1:
        mock_configuration = MagicMock()
        mock_cfg_cls.return_value = mock_configuration

        mock_api_client = AsyncMock()
        mock_api_client.__aenter__ = AsyncMock(return_value=mock_api_client)
        mock_api_client.__aexit__ = AsyncMock(return_value=False)
        mock_api_client_cls.return_value = mock_api_client

        mock_core_api = AsyncMock()
        mock_core_api.read_namespaced_config_map = AsyncMock(return_value=mock_cm)
        mock_core_v1.return_value = mock_core_api

        result = await read_config_map("my-configmap", namespace="default")

    assert result == mock_config_map_data
    mock_incluster.assert_called_once_with(client_configuration=mock_configuration)
    mock_api_client_cls.assert_called_once_with(mock_configuration)
    mock_core_v1.assert_called_once_with(mock_api_client)
    mock_core_api.read_namespaced_config_map.assert_called_once_with(
        name="my-configmap", namespace="default"
    )


@pytest.mark.asyncio
async def test_read_config_map_falls_back_to_kube_config(mock_config_map_data):
    """Test read_config_map falls back to load_kube_config when not in-cluster."""
    mock_cm = MagicMock()
    mock_cm.data = mock_config_map_data

    with patch("model_engine_server.core.configmap.client.Configuration") as mock_cfg_cls, patch(
        "model_engine_server.core.configmap.kube_config.load_incluster_config",
        side_effect=ConfigException("not in cluster"),
    ), patch(
        "model_engine_server.core.configmap.kube_config.load_kube_config", new_callable=AsyncMock
    ) as mock_load_kube, patch(
        "model_engine_server.core.configmap.client.ApiClient"
    ) as mock_api_client_cls, patch(
        "model_engine_server.core.configmap.client.CoreV1Api"
    ) as mock_core_v1:
        mock_configuration = MagicMock()
        mock_cfg_cls.return_value = mock_configuration

        mock_api_client = AsyncMock()
        mock_api_client.__aenter__ = AsyncMock(return_value=mock_api_client)
        mock_api_client.__aexit__ = AsyncMock(return_value=False)
        mock_api_client_cls.return_value = mock_api_client

        mock_core_api = AsyncMock()
        mock_core_api.read_namespaced_config_map = AsyncMock(return_value=mock_cm)
        mock_core_v1.return_value = mock_core_api

        result = await read_config_map("my-configmap", namespace="default")

    assert result == mock_config_map_data
    mock_load_kube.assert_called_once_with(client_configuration=mock_configuration)


@pytest.mark.asyncio
async def test_read_config_map_raises_api_exception():
    """Test read_config_map propagates ApiException from the k8s API."""
    with patch("model_engine_server.core.configmap.client.Configuration"), patch(
        "model_engine_server.core.configmap.kube_config.load_incluster_config"
    ), patch(
        "model_engine_server.core.configmap.client.ApiClient"
    ) as mock_api_client_cls, patch(
        "model_engine_server.core.configmap.client.CoreV1Api"
    ) as mock_core_v1:
        mock_api_client = AsyncMock()
        mock_api_client.__aenter__ = AsyncMock(return_value=mock_api_client)
        mock_api_client.__aexit__ = AsyncMock(return_value=False)
        mock_api_client_cls.return_value = mock_api_client

        mock_core_api = AsyncMock()
        mock_core_api.read_namespaced_config_map = AsyncMock(
            side_effect=ApiException(status=404, reason="Not Found")
        )
        mock_core_v1.return_value = mock_core_api

        with pytest.raises(ApiException):
            await read_config_map("missing-configmap", namespace="default")
