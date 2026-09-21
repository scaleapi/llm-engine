from datetime import datetime, timezone
from typing import Optional

from aiobotocore.session import AioSession
from model_engine_server.core.aws.roles import session
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.infra.gateways.resources.queue_endpoint_resource_delegate import (
    QueueEndpointResourceDelegate,
)
from model_engine_server.infra.services.endpoint_gc_service import QueueActivityGateway

logger = make_logger(logger_name())

_ONE_DAY_SECONDS = 86400


class SQSQueueActivityGateway(QueueActivityGateway):
    """Reads AWS/SQS NumberOfMessagesSent from CloudWatch for an endpoint's queue."""

    def __init__(self, sqs_profile: Optional[str]):
        self.sqs_profile = sqs_profile

    async def messages_sent_since(self, endpoint_id: str, since: datetime) -> Optional[int]:
        queue_name = QueueEndpointResourceDelegate.endpoint_id_to_queue_name(endpoint_id)
        try:
            async with session(role=self.sqs_profile, session_type=AioSession).create_client(
                "cloudwatch", region_name=infra_config().default_region
            ) as cloudwatch:
                response = await cloudwatch.get_metric_statistics(
                    Namespace="AWS/SQS",
                    MetricName="NumberOfMessagesSent",
                    Dimensions=[{"Name": "QueueName", "Value": queue_name}],
                    StartTime=since,
                    EndTime=datetime.now(timezone.utc),
                    Period=_ONE_DAY_SECONDS,
                    Statistics=["Sum"],
                )
        except Exception:
            logger.exception(f"CloudWatch lookup failed for queue {queue_name}")
            return None
        return int(sum(point.get("Sum", 0) for point in response.get("Datapoints", [])))


class UnknownQueueActivityGateway(QueueActivityGateway):
    """For clouds without a CloudWatch equivalent wired up: async endpoints are never collected."""

    async def messages_sent_since(self, endpoint_id: str, since: datetime) -> Optional[int]:
        return None
