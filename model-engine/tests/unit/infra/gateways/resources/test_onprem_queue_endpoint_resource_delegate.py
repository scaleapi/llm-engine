import pytest
from model_engine_server.infra.gateways.resources.onprem_queue_endpoint_resource_delegate import (
    OnPremQueueEndpointResourceDelegate,
)


@pytest.fixture
def onprem_queue_delegate():
    return OnPremQueueEndpointResourceDelegate()


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
async def test_get_queue_attributes(onprem_queue_delegate):
    result = await onprem_queue_delegate.get_queue_attributes(endpoint_id="test-endpoint-123")

    assert "Attributes" in result
    assert result["Attributes"]["ApproximateNumberOfMessages"] == "0"
    assert result["Attributes"]["QueueName"] == "launch-endpoint-id-test-endpoint-123"
    assert result["ResponseMetadata"]["HTTPStatusCode"] == 200

