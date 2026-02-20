from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest
from azure.core.exceptions import ResourceExistsError
from model_engine_server.infra.gateways.resources.asb_queue_endpoint_resource_delegate import (
    ASB_MAXIMUM_LOCK_DURATION,
    ASBQueueEndpointResourceDelegate,
)

MODULE_PATH = "model_engine_server.infra.gateways.resources.asb_queue_endpoint_resource_delegate"


@pytest.fixture
def mock_servicebus_client():
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    with patch(
        f"{MODULE_PATH}._get_servicebus_administration_client",
        return_value=mock_client,
    ):
        yield mock_client


@pytest.mark.asyncio
async def test_asb_create_queue_clamps_lock_duration_to_max(mock_servicebus_client):
    """Test that queue_message_timeout_seconds > 300 is clamped to ASB max (300s)."""
    mock_queue_props = MagicMock()
    mock_queue_props.lock_duration = timedelta(seconds=60)  # current value
    mock_servicebus_client.get_queue.return_value = mock_queue_props

    delegate = ASBQueueEndpointResourceDelegate()
    await delegate.create_queue_if_not_exists(
        endpoint_id="test_endpoint_id",
        endpoint_name="test_endpoint",
        endpoint_created_by="test_user",
        endpoint_labels={"team": "test_team"},
        queue_message_timeout_seconds=600,  # above 300 limit
    )

    # lock_duration should be clamped to 300
    assert mock_queue_props.lock_duration == timedelta(seconds=ASB_MAXIMUM_LOCK_DURATION)
    mock_servicebus_client.update_queue.assert_called_once_with(mock_queue_props)


@pytest.mark.asyncio
async def test_asb_create_queue_sets_lock_duration_within_limit(mock_servicebus_client):
    """Test that queue_message_timeout_seconds <= 300 is used as-is."""
    mock_queue_props = MagicMock()
    mock_queue_props.lock_duration = timedelta(seconds=60)  # current value
    mock_servicebus_client.get_queue.return_value = mock_queue_props

    delegate = ASBQueueEndpointResourceDelegate()
    await delegate.create_queue_if_not_exists(
        endpoint_id="test_endpoint_id",
        endpoint_name="test_endpoint",
        endpoint_created_by="test_user",
        endpoint_labels={"team": "test_team"},
        queue_message_timeout_seconds=120,
    )

    assert mock_queue_props.lock_duration == timedelta(seconds=120)
    mock_servicebus_client.update_queue.assert_called_once_with(mock_queue_props)


@pytest.mark.asyncio
async def test_asb_create_queue_no_update_when_timeout_is_none(mock_servicebus_client):
    """Test that lock_duration is not updated when queue_message_timeout_seconds is None."""
    delegate = ASBQueueEndpointResourceDelegate()
    await delegate.create_queue_if_not_exists(
        endpoint_id="test_endpoint_id",
        endpoint_name="test_endpoint",
        endpoint_created_by="test_user",
        endpoint_labels={"team": "test_team"},
    )

    mock_servicebus_client.get_queue.assert_not_called()
    mock_servicebus_client.update_queue.assert_not_called()


@pytest.mark.asyncio
async def test_asb_create_queue_skips_update_when_duration_unchanged(mock_servicebus_client):
    """Test that update_queue is not called if lock_duration already matches."""
    mock_queue_props = MagicMock()
    mock_queue_props.lock_duration = timedelta(seconds=200)
    mock_servicebus_client.get_queue.return_value = mock_queue_props

    delegate = ASBQueueEndpointResourceDelegate()
    await delegate.create_queue_if_not_exists(
        endpoint_id="test_endpoint_id",
        endpoint_name="test_endpoint",
        endpoint_created_by="test_user",
        endpoint_labels={"team": "test_team"},
        queue_message_timeout_seconds=200,
    )

    mock_servicebus_client.update_queue.assert_not_called()


@pytest.mark.asyncio
async def test_asb_create_queue_handles_existing_queue(mock_servicebus_client):
    """Test that ResourceExistsError is handled gracefully."""
    mock_servicebus_client.create_queue.side_effect = ResourceExistsError("Queue already exists")

    delegate = ASBQueueEndpointResourceDelegate()
    result = await delegate.create_queue_if_not_exists(
        endpoint_id="test_endpoint_id",
        endpoint_name="test_endpoint",
        endpoint_created_by="test_user",
        endpoint_labels={"team": "test_team"},
    )

    assert result.queue_name == "launch-endpoint-id-test_endpoint_id"
    assert result.queue_url is None
