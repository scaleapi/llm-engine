import pytest
from unittest.mock import patch, MagicMock
from datetime import timedelta

from model_engine_server.infra.gateways.resources.asb_queue_endpoint_resource_delegate import (
    ASBQueueEndpointResourceDelegate,
)


class TestASBQueueEndpointResourceDelegate:
    @pytest.fixture
    def delegate(self):
        return ASBQueueEndpointResourceDelegate()

    @pytest.fixture
    def mock_servicebus_client(self):
        with patch(
            "model_engine_server.infra.gateways.resources.asb_queue_endpoint_resource_delegate._get_servicebus_administration_client"
        ) as mock_client:
            yield mock_client

    @pytest.mark.asyncio
    async def test_create_queue_with_default_timeout(self, delegate, mock_servicebus_client):
        """Test queue creation with default 60-second timeout"""
        mock_client = MagicMock()
        mock_servicebus_client.return_value.__enter__.return_value = mock_client

        result = await delegate.create_queue_if_not_exists(
            endpoint_id="test-endpoint",
            endpoint_name="test-endpoint",
            endpoint_created_by="test-user",
            endpoint_labels={"team": "test"},
            queue_message_timeout_duration=60,  # Default
        )

        # Verify queue creation was called with correct properties
        mock_client.create_queue.assert_called_once()
        args, kwargs = mock_client.create_queue.call_args
        
        assert "queue_properties" in kwargs
        queue_properties = kwargs["queue_properties"]
        assert queue_properties.lock_duration == timedelta(seconds=60)
        assert result.queue_name == "launch-endpoint-id-test-endpoint"

    @pytest.mark.asyncio
    async def test_create_queue_with_custom_timeout(self, delegate, mock_servicebus_client):
        """Test queue creation with custom timeout duration"""
        mock_client = MagicMock()
        mock_servicebus_client.return_value.__enter__.return_value = mock_client

        await delegate.create_queue_if_not_exists(
            endpoint_id="test-endpoint",
            endpoint_name="test-endpoint",
            endpoint_created_by="test-user",
            endpoint_labels={"team": "test"},
            queue_message_timeout_duration=180,  # 3 minutes
        )

        # Verify queue creation was called with custom timeout
        args, kwargs = mock_client.create_queue.call_args
        queue_properties = kwargs["queue_properties"]
        assert queue_properties.lock_duration == timedelta(seconds=180)

    @pytest.mark.asyncio
    async def test_create_queue_timeout_validation_error(self, delegate, mock_servicebus_client):
        """Test that timeout > 300 seconds raises ValidationError"""
        with pytest.raises(ValueError, match="exceeds Azure Service Bus maximum of 300 seconds"):
            await delegate.create_queue_if_not_exists(
                endpoint_id="test-endpoint",
                endpoint_name="test-endpoint",
                endpoint_created_by="test-user",
                endpoint_labels={"team": "test"},
                queue_message_timeout_duration=400,  # > 300 seconds
            )

    @pytest.mark.asyncio
    async def test_create_queue_max_allowed_timeout(self, delegate, mock_servicebus_client):
        """Test queue creation with maximum allowed timeout (300s)"""
        mock_client = MagicMock()
        mock_servicebus_client.return_value.__enter__.return_value = mock_client

        await delegate.create_queue_if_not_exists(
            endpoint_id="test-endpoint",
            endpoint_name="test-endpoint",
            endpoint_created_by="test-user",
            endpoint_labels={"team": "test"},
            queue_message_timeout_duration=300,  # Exactly 5 minutes
        )

        # Should succeed without error
        args, kwargs = mock_client.create_queue.call_args
        queue_properties = kwargs["queue_properties"]
        assert queue_properties.lock_duration == timedelta(seconds=300)
