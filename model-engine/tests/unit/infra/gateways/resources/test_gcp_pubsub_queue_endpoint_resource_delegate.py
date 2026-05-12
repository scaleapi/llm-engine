from unittest.mock import MagicMock, patch

import pytest
from google.api_core import exceptions as gcp_exceptions
from model_engine_server.domain.exceptions import EndpointResourceInfraException
from model_engine_server.infra.gateways.resources.gcp_pubsub_queue_endpoint_resource_delegate import (
    GcpPubSubQueueEndpointResourceDelegate,
)

MODULE_PATH = "model_engine_server.infra.gateways.resources.gcp_pubsub_queue_endpoint_resource_delegate"

ENDPOINT_ID = "test_endpoint_id"
PROJECT_ID = "test-project"
TOPIC_PREFIX = "launch-endpoint-id-"
SUBSCRIPTION_PREFIX = "launch-endpoint-id-"
QUEUE_NAME = f"{TOPIC_PREFIX}{ENDPOINT_ID}"


@pytest.fixture
def mock_publisher():
    with patch(f"{MODULE_PATH}.pubsub_v1.PublisherClient") as mock_cls:
        yield mock_cls.return_value


@pytest.fixture
def mock_subscriber():
    with patch(f"{MODULE_PATH}.pubsub_v1.SubscriberClient") as mock_cls:
        yield mock_cls.return_value


@pytest.fixture
def delegate(mock_publisher, mock_subscriber):
    return GcpPubSubQueueEndpointResourceDelegate(project_id=PROJECT_ID)


def test_init_empty_project_id_raises():
    with pytest.raises(ValueError, match="non-empty project_id"):
        GcpPubSubQueueEndpointResourceDelegate(project_id="")


@pytest.mark.asyncio
async def test_create_queue_if_not_exists_new(
    mock_publisher, mock_subscriber, delegate
):
    """Both topic and subscription are created when neither exists."""
    result = await delegate.create_queue_if_not_exists(
        endpoint_id=ENDPOINT_ID,
        endpoint_name="test_endpoint",
        endpoint_created_by="test_user",
        endpoint_labels={"team": "test"},
    )

    topic_path = f"projects/{PROJECT_ID}/topics/{TOPIC_PREFIX}{ENDPOINT_ID}"
    subscription_path = (
        f"projects/{PROJECT_ID}/subscriptions/{SUBSCRIPTION_PREFIX}{ENDPOINT_ID}"
    )

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
    mock_publisher.create_topic.side_effect = gcp_exceptions.AlreadyExists(
        "topic exists"
    )

    result = await delegate.create_queue_if_not_exists(
        endpoint_id=ENDPOINT_ID,
        endpoint_name="test_endpoint",
        endpoint_created_by="test_user",
        endpoint_labels={},
    )

    mock_subscriber.create_subscription.assert_called_once()
    assert result.queue_name == QUEUE_NAME


@pytest.mark.asyncio
async def test_create_queue_if_not_exists_subscription_already_exists_updates_ack_deadline(
    mock_publisher, mock_subscriber, delegate
):
    """AlreadyExists on subscription triggers an update_subscription call."""
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
    mock_subscriber.update_subscription.assert_called_once()
    assert result.queue_name == QUEUE_NAME


@pytest.mark.asyncio
async def test_create_queue_subscription_already_exists_update_failure_is_warned(
    mock_publisher, mock_subscriber, delegate
):
    """update_subscription GoogleAPIError is swallowed with a warning (not raised)."""
    mock_subscriber.create_subscription.side_effect = gcp_exceptions.AlreadyExists(
        "exists"
    )
    mock_subscriber.update_subscription.side_effect = gcp_exceptions.GoogleAPIError(
        "boom"
    )

    # Should not raise
    result = await delegate.create_queue_if_not_exists(
        endpoint_id=ENDPOINT_ID,
        endpoint_name="test_endpoint",
        endpoint_created_by="test_user",
        endpoint_labels={},
    )
    assert result.queue_name == QUEUE_NAME


@pytest.mark.asyncio
async def test_delete_queue_subscription_not_found_silent(
    mock_publisher, mock_subscriber, delegate
):
    """NotFound on subscription deletion is silenced; topic deletion still attempts."""
    mock_subscriber.delete_subscription.side_effect = gcp_exceptions.NotFound(
        "sub not found"
    )

    await delegate.delete_queue(endpoint_id=ENDPOINT_ID)

    mock_subscriber.delete_subscription.assert_called_once()
    mock_publisher.delete_topic.assert_called_once()


@pytest.mark.asyncio
async def test_delete_queue_topic_not_found_silent(
    mock_publisher, mock_subscriber, delegate
):
    """NotFound on topic deletion is silenced."""
    mock_publisher.delete_topic.side_effect = gcp_exceptions.NotFound("topic not found")

    await delegate.delete_queue(endpoint_id=ENDPOINT_ID)

    mock_subscriber.delete_subscription.assert_called_once()
    mock_publisher.delete_topic.assert_called_once()


@pytest.mark.asyncio
async def test_delete_queue_subscription_api_error_raises(
    mock_publisher, mock_subscriber, delegate
):
    """Non-NotFound GoogleAPIError on subscription deletion raises EndpointResourceInfraException."""
    mock_subscriber.delete_subscription.side_effect = gcp_exceptions.GoogleAPIError(
        "network error"
    )

    with pytest.raises(
        EndpointResourceInfraException, match="Failed to delete Pub/Sub subscription"
    ):
        await delegate.delete_queue(endpoint_id=ENDPOINT_ID)


@pytest.mark.asyncio
async def test_delete_queue_topic_api_error_raises(
    mock_publisher, mock_subscriber, delegate
):
    """Non-NotFound GoogleAPIError on topic deletion raises EndpointResourceInfraException."""
    mock_publisher.delete_topic.side_effect = gcp_exceptions.GoogleAPIError(
        "network error"
    )

    with pytest.raises(
        EndpointResourceInfraException, match="Failed to delete Pub/Sub topic"
    ):
        await delegate.delete_queue(endpoint_id=ENDPOINT_ID)


@pytest.mark.asyncio
async def test_delete_queue_subscription_deleted_before_topic(
    mock_publisher, mock_subscriber, delegate
):
    """Subscription must be deleted before topic (Pub/Sub requirement to avoid race)."""
    parent = MagicMock()
    parent.attach_mock(mock_subscriber.delete_subscription, "sub_del")
    parent.attach_mock(mock_publisher.delete_topic, "topic_del")

    await delegate.delete_queue(endpoint_id=ENDPOINT_ID)

    call_order = [c[0] for c in parent.mock_calls]
    assert call_order == ["sub_del", "topic_del"]


@pytest.mark.asyncio
async def test_get_queue_attributes_returns_expected_shape(delegate):
    """get_queue_attributes returns a dict with 'name' and 'num_undelivered_messages'."""
    attrs = await delegate.get_queue_attributes(endpoint_id=ENDPOINT_ID)

    assert attrs["name"] == QUEUE_NAME
    assert "num_undelivered_messages" in attrs
    assert attrs["num_undelivered_messages"] == -1
