from typing import Any, Dict, Optional

from google.api_core import exceptions as gcp_exceptions
from google.cloud import pubsub_v1
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.infra.gateways.resources.queue_endpoint_resource_delegate import (
    QueueEndpointResourceDelegate,
    QueueInfo,
)

logger = make_logger(logger_name())

GCP_PUBSUB_MAX_ACK_DEADLINE_SECONDS = 600  # Pub/Sub hard limit


class GcpPubSubQueueEndpointResourceDelegate(QueueEndpointResourceDelegate):
    """
    Using GCP Pub/Sub (topic + subscription per endpoint).
    """

    def __init__(
        self,
        project_id: str,
        topic_prefix: str = "launch-endpoint-id-",
        subscription_prefix: str = "launch-endpoint-id-",
    ) -> None:
        self.project_id = project_id
        self.topic_prefix = topic_prefix
        self.subscription_prefix = subscription_prefix

    async def create_queue_if_not_exists(
        self,
        endpoint_id: str,
        endpoint_name: str,
        endpoint_created_by: str,
        endpoint_labels: Dict[str, Any],
        queue_message_timeout_seconds: Optional[int] = None,
    ) -> QueueInfo:
        queue_name = QueueEndpointResourceDelegate.endpoint_id_to_queue_name(endpoint_id)
        topic_path = f"projects/{self.project_id}/topics/{queue_name}"
        subscription_path = f"projects/{self.project_id}/subscriptions/{queue_name}"
        ack_deadline = min(queue_message_timeout_seconds or 60, GCP_PUBSUB_MAX_ACK_DEADLINE_SECONDS)

        publisher = pubsub_v1.PublisherClient()
        subscriber = pubsub_v1.SubscriberClient()

        try:
            publisher.create_topic(name=topic_path)
        except gcp_exceptions.AlreadyExists:
            pass

        try:
            subscriber.create_subscription(
                name=subscription_path,
                topic=topic_path,
                ack_deadline_seconds=ack_deadline,
            )
        except gcp_exceptions.AlreadyExists:
            pass

        # Pub/Sub has no URL concept analogous to SQS queue URLs
        return QueueInfo(queue_name, queue_url=None)

    async def delete_queue(self, endpoint_id: str) -> None:
        queue_name = QueueEndpointResourceDelegate.endpoint_id_to_queue_name(endpoint_id)
        subscription_path = f"projects/{self.project_id}/subscriptions/{queue_name}"
        topic_path = f"projects/{self.project_id}/topics/{queue_name}"

        subscriber = pubsub_v1.SubscriberClient()
        publisher = pubsub_v1.PublisherClient()

        try:
            subscriber.delete_subscription(subscription=subscription_path)
        except gcp_exceptions.NotFound:
            logger.info(
                f"Could not find Pub/Sub subscription {subscription_path} for endpoint {endpoint_id}"
            )

        try:
            publisher.delete_topic(topic=topic_path)
        except gcp_exceptions.NotFound:
            logger.info(
                f"Could not find Pub/Sub topic {topic_path} for endpoint {endpoint_id}"
            )

    async def get_queue_attributes(self, endpoint_id: str) -> Dict[str, Any]:
        queue_name = QueueEndpointResourceDelegate.endpoint_id_to_queue_name(endpoint_id)
        return {
            "name": queue_name,
            # Pub/Sub does not expose a synchronous undelivered message count;
            # real observability requires the Cloud Monitoring API as a separate concern.
            "num_undelivered_messages": -1,
        }
