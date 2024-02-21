import json
from typing import Any, Dict

import boto3
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.exceptions import StreamPutException
from model_engine_server.inference.domain.gateways.streaming_storage_gateway import (
    StreamingStorageGateway,
)

logger = make_logger(logger_name())


class FirehoseStreamingStorageGateway(StreamingStorageGateway):
    """
    A gateway that stores data through the AWS Kinesis Firehose streaming mechanism.
    """

    def __init__(self):
        pass

    """
    Creates a new firehose client.

    Streams with Snowflake as a destination and the AWS profile live in different 
    accounts. Firehose doesn't support resource-based policies, so we need to assume 
    a new role to write to the stream.
    """

    def _get_firehose_client(self):
        sts_client = boto3.client("sts", region_name=infra_config().default_region)
        assumed_role_object = sts_client.assume_role(
            RoleArn=infra_config().firehose_role_arn,
            RoleSessionName="AssumeMlLoggingRoleSession",
        )
        credentials = assumed_role_object["Credentials"]
        session = boto3.Session(
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
        )
        firehose_client = session.client("firehose", region_name=infra_config().default_region)
        return firehose_client

    def put_record(self, stream_name: str, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Put a record into a Firehose stream.

        Args:
            stream_name: The name of the stream.
            record: The record to put into the stream.
        """
        firehose_response = self._get_firehose_client().put_record(
            DeliveryStreamName=stream_name, Record={"Data": json.dumps(record).encode("utf-8")}
        )
        if firehose_response["ResponseMetadata"]["HTTPStatusCode"] != 200:
            raise StreamPutException(
                f"Failed to put record into firehose stream {stream_name}. Response metadata {firehose_response['ResponseMetadata']}."
            )
        logger.info(
            f"Logged to firehose stream {stream_name}. Record ID: {firehose_response['RecordId']}. Task ID: {record['RESPONSE_BODY']['task_id']}"
        )
        return firehose_response
