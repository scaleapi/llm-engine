from unittest import mock
from unittest.mock import AsyncMock

import pytest
from model_engine_server.infra.gateways.resources.onprem_queue_endpoint_resource_delegate import (
    OnPremQueueEndpointResourceDelegate,
)


@pytest.fixture
def mock_redis_client():
    client = mock.Mock()
    client.llen = AsyncMock(return_value=5)
    return client


@pytest.fixture
def onprem_queue_delegate():
    return OnPremQueueEndpointResourceDelegate()


@pytest.fixture
def onprem_queue_delegate_with_redis(mock_redis_client):
    return OnPremQueueEndpointResourceDelegate(redis_client=mock_redis_client)


@pytest.mark.asyncio
async def test_create_queue_if_not_exists(onprem_queue_delegate):
    result = await onprem_queue_delegate.create_queue_if_not_exists(
        endpoint_id="test-endpoint-123",
        endpoint_name="test-endpoint",
        endpoint_created_by="test-user",
        endpoint_labels={"team": "test-team"},
    )

    assert result.queue_name == "launch-endpoint-id-test-endpoint-123"
    assert result.queue_url == "launch-endpoint-id-test-endpoint-123"


@pytest.mark.asyncio
async def test_delete_queue(onprem_queue_delegate):
    await onprem_queue_delegate.delete_queue(endpoint_id="test-endpoint-123")


@pytest.mark.asyncio
async def test_get_queue_attributes_no_redis(onprem_queue_delegate):
    result = await onprem_queue_delegate.get_queue_attributes(endpoint_id="test-endpoint-123")

    assert "Attributes" in result
    assert result["Attributes"]["ApproximateNumberOfMessages"] == "0"
    assert result["Attributes"]["QueueName"] == "launch-endpoint-id-test-endpoint-123"
    assert result["ResponseMetadata"]["HTTPStatusCode"] == 200


@pytest.mark.asyncio
async def test_get_queue_attributes_with_redis(onprem_queue_delegate_with_redis, mock_redis_client):
    result = await onprem_queue_delegate_with_redis.get_queue_attributes(
        endpoint_id="test-endpoint-123"
    )

    assert "Attributes" in result
    assert result["Attributes"]["ApproximateNumberOfMessages"] == "5"
    assert result["Attributes"]["QueueName"] == "launch-endpoint-id-test-endpoint-123"
    assert result["ResponseMetadata"]["HTTPStatusCode"] == 200
    mock_redis_client.llen.assert_called_once_with("launch-endpoint-id-test-endpoint-123")
