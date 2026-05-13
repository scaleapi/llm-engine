from typing import Any, Dict, Optional

from google.api_core import exceptions as gcp_exceptions
from google.cloud import pubsub_v1
from google.protobuf import field_mask_pb2
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.exceptions import EndpointResourceInfraException
from model_engine_server.infra.gateways.resources.queue_endpoint_resource_delegate import (
    QueueEndpointResourceDelegate,
    QueueInfo,
)

logger = make_logger(logger_name())

GCP_PUBSUB_MAX_ACK_DEADLINE_SECONDS = 600  # Pub/Sub hard limit


class GcpPubSubQueueEndpointResourceDelegate(QueueEndpointResourceDelegate):
    """
    Using GCP Pub/Sub (topic + subscription per endpoint).

    topic_prefix and subscription_prefix control the GCP resource name prefix.
    The logical queue_name returned to callers always uses the canonical
    QueueEndpointResourceDelegate.endpoint_id_to_queue_name format, independent
    of these prefixes.
    """

    def __init__(
        self,
        project_id: str,
        topic_prefix: str = "launch-endpoint-id-",
        subscription_prefix: str = "launch-endpoint-id-",
    ) -> None:
        if not project_id:
            raise ValueError(
                "GcpPubSubQueueEndpointResourceDelegate requires a non-empty project_id; "
                "set infra.gcp_project_id in the service config."
            )
        self.project_id = project_id
        self.topic_prefix = topic_prefix
        self.subscription_prefix = subscription_prefix
        # Lazily-initialized gRPC clients. Construction calls Google ADC which is
        # unavailable in unit-test environments, so defer until first real use.
        # The clients are then cached for the lifetime of the delegate.
        self._publisher_client: Optional[pubsub_v1.PublisherClient] = None
        self._subscriber_client: Optional[pubsub_v1.SubscriberClient] = None

    @property
    def _publisher(self) -> pubsub_v1.PublisherClient:
        if self._publisher_client is None:
            self._publisher_client = pubsub_v1.PublisherClient()
        return self._publisher_client

    @property
    def _subscriber(self) -> pubsub_v1.SubscriberClient:
        if self._subscriber_client is None:
            self._subscriber_client = pubsub_v1.SubscriberClient()
        return self._subscriber_client

    def _topic_id(self, endpoint_id: str) -> str:
        return f"{self.topic_prefix}{endpoint_id}"

    def _subscription_id(self, endpoint_id: str) -> str:
        return f"{self.subscription_prefix}{endpoint_id}"

    async def create_queue_if_not_exists(
        self,
        endpoint_id: str,
        endpoint_name: str,
        endpoint_created_by: str,
        endpoint_labels: Dict[str, Any],
        queue_message_timeout_seconds: Optional[int] = None,
    ) -> QueueInfo:
        queue_name = QueueEndpointResourceDelegate.endpoint_id_to_queue_name(endpoint_id)
        topic_path = f"projects/{self.project_id}/topics/{self._topic_id(endpoint_id)}"
        subscription_path = (
            f"projects/{self.project_id}/subscriptions/{self._subscription_id(endpoint_id)}"
        )
        ack_deadline = min(queue_message_timeout_seconds or 60, GCP_PUBSUB_MAX_ACK_DEADLINE_SECONDS)

        try:
            self._publisher.create_topic(name=topic_path)
        except gcp_exceptions.AlreadyExists:
            pass

        try:
            self._subscriber.create_subscription(
                name=subscription_path,
                topic=topic_path,
                ack_deadline_seconds=ack_deadline,
            )
        except gcp_exceptions.AlreadyExists:
            try:
                self._subscriber.update_subscription(
                    subscription=pubsub_v1.types.Subscription(
                        name=subscription_path,
                        ack_deadline_seconds=ack_deadline,
                    ),
                    update_mask=field_mask_pb2.FieldMask(paths=["ack_deadline_seconds"]),
                )
            except gcp_exceptions.GoogleAPIError as e:
                logger.warning(
                    f"Failed to update ack_deadline for Pub/Sub subscription {subscription_path}: {e}"
                )

        # Pub/Sub has no URL concept analogous to SQS queue URLs
        return QueueInfo(queue_name, queue_url=None)

    async def delete_queue(self, endpoint_id: str) -> None:
        subscription_path = (
            f"projects/{self.project_id}/subscriptions/{self._subscription_id(endpoint_id)}"
        )
        topic_path = f"projects/{self.project_id}/topics/{self._topic_id(endpoint_id)}"

        # Always attempt BOTH deletions so a failure on one doesn't leave the other resource
        # orphaned (Greptile P1). NotFound is silent. Other GoogleAPIErrors are collected and
        # surfaced together at the end so callers see every cleanup failure, not just the first.
        errors: list[tuple[str, str, gcp_exceptions.GoogleAPIError]] = []

        try:
            self._subscriber.delete_subscription(subscription=subscription_path)
        except gcp_exceptions.NotFound:
            logger.info(
                f"Could not find Pub/Sub subscription {subscription_path} for endpoint {endpoint_id}"
            )
        except gcp_exceptions.GoogleAPIError as e:
            errors.append(("subscription", subscription_path, e))

        try:
            self._publisher.delete_topic(topic=topic_path)
        except gcp_exceptions.NotFound:
            logger.info(f"Could not find Pub/Sub topic {topic_path} for endpoint {endpoint_id}")
        except gcp_exceptions.GoogleAPIError as e:
            errors.append(("topic", topic_path, e))

        if errors:
            details = "; ".join(
                f"Failed to delete Pub/Sub {kind} {path}: {err}" for kind, path, err in errors
            )
            raise EndpointResourceInfraException(
                f"Cleanup errors for endpoint {endpoint_id}: {details}"
            ) from errors[0][2]

    async def get_queue_attributes(self, endpoint_id: str) -> Dict[str, Any]:
        queue_name = QueueEndpointResourceDelegate.endpoint_id_to_queue_name(endpoint_id)
        return {
            "name": queue_name,
            # Pub/Sub does not expose a synchronous undelivered message count;
            # real observability requires the Cloud Monitoring API as a separate concern.
            "num_undelivered_messages": -1,
        }
