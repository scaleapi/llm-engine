from unittest.mock import MagicMock, patch

import pytest
from google.api_core import exceptions as gcp_exceptions
from model_engine_server.infra.gateways.resources.gcp_pubsub_queue_endpoint_resource_delegate import (
    GcpPubSubQueueEndpointResourceDelegate,
)

MODULE_PATH = "model_engine_server.infra.gateways.resources.gcp_pubsub_queue_endpoint_resource_delegate"

ENDPOINT_ID = "test_endpoint_id"
QUEUE_NAME = f"launch-endpoint-id-{ENDPOINT_ID}"
PROJECT_ID = "test-project"


@pytest.fixture
def mock_publisher():
    with patch(f"{MODULE_PATH}.pubsub_v1.PublisherClient") as mock_cls:
        yield mock_cls.return_value


@pytest.fixture
def mock_subscriber():
    with patch(f"{MODULE_PATH}.pubsub_v1.SubscriberClient") as mock_cls:
        yield mock_cls.return_value


@pytest.fixture
def delegate():
    return GcpPubSubQueueEndpointResourceDelegate(project_id=PROJECT_ID)


@pytest.mark.asyncio
async def test_create_queue_if_not_exists_new(mock_publisher, mock_subscriber, delegate):
    """Both topic and subscription are created when neither exists."""
    result = await delegate.create_queue_if_not_exists(
        endpoint_id=ENDPOINT_ID,
        endpoint_name="test_endpoint",
        endpoint_created_by="test_user",
        endpoint_labels={"team": "test"},
    )

    topic_path = f"projects/{PROJECT_ID}/topics/{QUEUE_NAME}"
    subscription_path = f"projects/{PROJECT_ID}/subscriptions/{QUEUE_NAME}"

    mock_publisher.create_topic.assert_called_once_with(name=topic_path)
    mock_subscriber.create_subscription.assert_called_once_with(
        name=subscription_path,
        topic=topic_path,
        ack_deadline_seconds=60,  # default when timeout is None
    )
    assert result.queue_name == QUEUE_NAME
    assert result.queue_url is None


@pytest.mark.asyncio
async def test_create_queue_if_not_exists_topic_already_exists(
    mock_publisher, mock_subscriber, delegate
):
    """AlreadyExists on topic creation is silenced; subscription still attempts creation."""
    mock_publisher.create_topic.side_effect = gcp_exceptions.AlreadyExists("topic exists")

    result = await delegate.create_queue_if_not_exists(
        endpoint_id=ENDPOINT_ID,
        endpoint_name="test_endpoint",
        endpoint_created_by="test_user",
        endpoint_labels={},
    )

    mock_subscriber.create_subscription.assert_called_once()
    assert result.queue_name == QUEUE_NAME


@pytest.mark.asyncio
async def test_create_queue_if_not_exists_subscription_already_exists(
    mock_publisher, mock_subscriber, delegate
):
    """AlreadyExists on subscription creation is silenced."""
    mock_subscriber.create_subscription.side_effect = gcp_exceptions.AlreadyExists(
        "subscription exists"
    )

    result = await delegate.create_queue_if_not_exists(
        endpoint_id=ENDPOINT_ID,
        endpoint_name="test_endpoint",
        endpoint_created_by="test_user",
        endpoint_labels={},
        queue_message_timeout_seconds=120,
    )

    mock_publisher.create_topic.assert_called_once()
    assert result.queue_name == QUEUE_NAME


@pytest.mark.asyncio
async def test_delete_queue_subscription_not_found_silent(
    mock_publisher, mock_subscriber, delegate
):
    """NotFound on subscription deletion is silenced; topic deletion still attempts."""
    mock_subscriber.delete_subscription.side_effect = gcp_exceptions.NotFound("sub not found")

    await delegate.delete_queue(endpoint_id=ENDPOINT_ID)

    mock_subscriber.delete_subscription.assert_called_once()
    mock_publisher.delete_topic.assert_called_once()


@pytest.mark.asyncio
async def test_delete_queue_topic_not_found_silent(mock_publisher, mock_subscriber, delegate):
    """NotFound on topic deletion is silenced."""
    mock_publisher.delete_topic.side_effect = gcp_exceptions.NotFound("topic not found")

    await delegate.delete_queue(endpoint_id=ENDPOINT_ID)

    mock_subscriber.delete_subscription.assert_called_once()
    mock_publisher.delete_topic.assert_called_once()


@pytest.mark.asyncio
async def test_get_queue_attributes_returns_expected_shape(delegate):
    """get_queue_attributes returns a dict with 'name' and 'num_undelivered_messages'."""
    attrs = await delegate.get_queue_attributes(endpoint_id=ENDPOINT_ID)

    assert attrs["name"] == QUEUE_NAME
    assert "num_undelivered_messages" in attrs
    assert attrs["num_undelivered_messages"] == -1
