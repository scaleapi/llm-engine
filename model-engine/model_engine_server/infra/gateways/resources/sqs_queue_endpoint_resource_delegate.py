import json
from string import Template
from typing import Any, Dict, Optional, Sequence

import botocore.exceptions
from aioboto3 import Session as AioSession
from aiobotocore.client import AioBaseClient
from model_engine_server.common.config import hmi_config
from model_engine_server.core.aws.roles import session
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.exceptions import EndpointResourceInfraException
from model_engine_server.infra.gateways.resources.queue_endpoint_resource_delegate import (
    QueueEndpointResourceDelegate,
    QueueInfo,
)

logger = make_logger(logger_name())

__all__: Sequence[str] = ("SQSQueueEndpointResourceDelegate",)


def _create_async_sqs_client(sqs_profile: Optional[str]) -> AioBaseClient:
    return session(role=sqs_profile, session_type=AioSession).client(
        "sqs", region_name=infra_config().default_region
    )


def _get_queue_policy(queue_name: str) -> str:
    queue_policy_template = Template(hmi_config.sqs_queue_policy_template)
    return queue_policy_template.substitute(queue_name=queue_name)


def _get_queue_tags(
    team: str, endpoint_id: str, endpoint_name: str, endpoint_created_by: str
) -> Dict[str, str]:
    queue_tag_template = Template(hmi_config.sqs_queue_tag_template)
    return json.loads(
        queue_tag_template.substitute(
            team=team,
            endpoint_id=endpoint_id,
            endpoint_name=endpoint_name,
            endpoint_created_by=endpoint_created_by,
        )
    )


class SQSQueueEndpointResourceDelegate(QueueEndpointResourceDelegate):
    def __init__(self, sqs_profile: Optional[str]):
        self.sqs_profile = sqs_profile

    async def create_queue_if_not_exists(
        self,
        endpoint_id: str,
        endpoint_name: str,
        endpoint_created_by: str,
        endpoint_labels: Dict[str, Any],
        queue_message_timeout_duration: Optional[int] = None,
    ) -> QueueInfo:
        # Use provided timeout or default to 43200 (12 hours, max SQS visibility)
        timeout_duration = queue_message_timeout_duration or 43200
        
        async with _create_async_sqs_client(sqs_profile=self.sqs_profile) as sqs_client:
            queue_name = QueueEndpointResourceDelegate.endpoint_id_to_queue_name(endpoint_id)

            try:
                get_queue_url_response = await sqs_client.get_queue_url(QueueName=queue_name)
                return QueueInfo(
                    queue_name=queue_name,
                    queue_url=get_queue_url_response["QueueUrl"],
                )
            except botocore.exceptions.ClientError:
                logger.info("Queue does not exist, creating it")
                pass

            try:
                create_response = await sqs_client.create_queue(
                    QueueName=queue_name,
                    Attributes=dict(
                        VisibilityTimeout=str(timeout_duration),
                        Policy=_get_queue_policy(queue_name=queue_name),
                    ),
                    tags=_get_queue_tags(
                        team=endpoint_labels["team"],
                        endpoint_id=endpoint_id,
                        endpoint_name=endpoint_name,
                        endpoint_created_by=endpoint_created_by,
                    ),
                )
            except botocore.exceptions.ClientError as e:
                raise EndpointResourceInfraException("Failed to create SQS queue") from e

            if create_response["ResponseMetadata"]["HTTPStatusCode"] != 200:
                raise EndpointResourceInfraException(
                    f"Creating SQS queue got non-200 response: {create_response}"
                )

            return QueueInfo(queue_name, create_response["QueueUrl"])

    async def delete_queue(self, endpoint_id: str) -> None:
        queue_name = QueueEndpointResourceDelegate.endpoint_id_to_queue_name(endpoint_id)
        async with _create_async_sqs_client(self.sqs_profile) as sqs_client:
            try:
                queue_url = (await sqs_client.get_queue_url(QueueName=queue_name))["QueueUrl"]
            except botocore.exceptions.ClientError:
                logger.info(
                    f"Could not get queue url for queue_name={queue_name}, endpoint_id={endpoint_id}, "
                    "skipping delete"
                )
                return

            try:
                delete_response = await sqs_client.delete_queue(QueueUrl=queue_url)
            except botocore.exceptions.ClientError as e:
                raise EndpointResourceInfraException("Failed to delete SQS queue") from e

            # Example failed delete:
            # botocore.errorfactory.QueueDoesNotExist:
            #   An error occurred (AWS.SimpleQueueService.NonExistentQueue) when calling the GetQueueUrl operation:
            #   The specified queue does not exist for this wsdl version.
            if delete_response["ResponseMetadata"]["HTTPStatusCode"] != 200:
                raise EndpointResourceInfraException(
                    f"Deleting SQS queue got non-200 response: {delete_response}"
                )

    async def get_queue_attributes(self, endpoint_id: str) -> Dict[str, Any]:
        queue_name = QueueEndpointResourceDelegate.endpoint_id_to_queue_name(endpoint_id)
        async with _create_async_sqs_client(self.sqs_profile) as sqs_client:
            try:
                queue_url = (await sqs_client.get_queue_url(QueueName=queue_name))["QueueUrl"]
            except botocore.exceptions.ClientError as e:
                raise EndpointResourceInfraException(
                    f"Could not find queue {queue_name} for endpoint {endpoint_id}"
                ) from e

            try:
                attributes_response = await sqs_client.get_queue_attributes(
                    QueueUrl=queue_url, AttributeNames=["All"]
                )
            except botocore.exceptions.ClientError as e:
                raise EndpointResourceInfraException("Failed to get SQS queue attributes") from e

            return attributes_response
