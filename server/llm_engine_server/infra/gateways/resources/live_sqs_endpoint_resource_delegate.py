import json
from string import Template
from typing import Any, Dict, Optional, Sequence

import botocore.exceptions
from aioboto3 import Session as AioSession
from aiobotocore.client import AioBaseClient
from llm_engine_server.common.config import hmi_config
from llm_engine_server.core.aws.roles import session
from llm_engine_server.core.loggers import filename_wo_ext, make_logger
from llm_engine_server.domain.exceptions import EndpointResourceInfraException
from llm_engine_server.infra.gateways.resources.sqs_endpoint_resource_delegate import (
    SQSEndpointResourceDelegate,
    SQSQueueInfo,
)
from mypy_boto3_sqs.type_defs import GetQueueAttributesResultTypeDef

logger = make_logger(filename_wo_ext(__file__))

__all__: Sequence[str] = ("LiveSQSEndpointResourceDelegate",)


def _create_async_sqs_client(sqs_profile: Optional[str]) -> AioBaseClient:
    return session(role=sqs_profile, session_type=AioSession).client("sqs", region_name="us-west-2")


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


class LiveSQSEndpointResourceDelegate(SQSEndpointResourceDelegate):
    def __init__(self, sqs_profile: Optional[str]):
        self.sqs_profile = sqs_profile

    async def create_queue_if_not_exists(
        self,
        endpoint_id: str,
        endpoint_name: str,
        endpoint_created_by: str,
        endpoint_labels: Dict[str, Any],
    ) -> SQSQueueInfo:
        async with _create_async_sqs_client(sqs_profile=self.sqs_profile) as sqs_client:
            queue_name = SQSEndpointResourceDelegate.endpoint_id_to_queue_name(endpoint_id)

            try:
                get_queue_url_response = await sqs_client.get_queue_url(QueueName=queue_name)
                return SQSQueueInfo(
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
                        VisibilityTimeout="43200",
                        # To match current hardcoded Celery timeout of 24hr
                        # However, the max SQS visibility is 12hrs.
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

            return SQSQueueInfo(queue_name, create_response["QueueUrl"])

    async def delete_queue(self, endpoint_id: str) -> None:
        queue_name = SQSEndpointResourceDelegate.endpoint_id_to_queue_name(endpoint_id)
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

    async def get_queue_attributes(self, endpoint_id: str) -> GetQueueAttributesResultTypeDef:
        queue_name = SQSEndpointResourceDelegate.endpoint_id_to_queue_name(endpoint_id)
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
