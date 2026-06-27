import json
import os
import threading
from typing import Any, Dict

import boto3
from botocore.config import Config
from botocore.credentials import RefreshableCredentials
from botocore.session import get_session
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.exceptions import StreamPutException
from model_engine_server.inference.domain.gateways.streaming_storage_gateway import (
    StreamingStorageGateway,
)

logger = make_logger(logger_name())

# Shared client is reused across greenlets under the gevent pool; size the pool to concurrency.
_FIREHOSE_MAX_POOL_CONNECTIONS = int(os.getenv("FIREHOSE_CLIENT_MAX_POOL_CONNECTIONS", "50"))


class FirehoseStreamingStorageGateway(StreamingStorageGateway):
    """Stores records via AWS Kinesis Firehose.

    MLI-7328: client is built once and cached. It is called per task by the logging hook;
    rebuilding per call (STS assume_role + new boto clients) leaked memory and OOM-killed the
    forwarder. Do not revert to per-call construction. Assumed-role creds are temporary, so the
    cache uses RefreshableCredentials to re-assume before expiry.
    """

    def __init__(self):
        self._client = None
        # gevent: greenlets share this client and build() yields on assume_role; lock stops two
        # greenlets both building. threading.Lock is greenlet-aware after monkey.patch_all().
        self._lock = threading.Lock()

    def _make_refreshable_client(self):
        # Firehose stream is cross-account, so we assume a role; RefreshableCredentials lets
        # botocore re-assume before the temporary creds expire (no client rebuild needed).
        region = infra_config().default_region
        role_arn = infra_config().firehose_role_arn

        def _refresh():
            sts_client = boto3.Session(region_name=region).client("sts")
            credentials = sts_client.assume_role(
                RoleArn=role_arn,
                RoleSessionName="AssumeMlLoggingRoleSession",
            )["Credentials"]
            return {
                "access_key": credentials["AccessKeyId"],
                "secret_key": credentials["SecretAccessKey"],
                "token": credentials["SessionToken"],
                "expiry_time": credentials["Expiration"].isoformat(),
            }

        creds = RefreshableCredentials.create_from_metadata(
            metadata=_refresh(),
            refresh_using=_refresh,
            method="sts-assume-role",
        )
        botocore_session = get_session()
        botocore_session._credentials = creds
        return boto3.Session(botocore_session=botocore_session).client(
            "firehose",
            region_name=region,
            config=Config(max_pool_connections=_FIREHOSE_MAX_POOL_CONNECTIONS),
        )

    def _get_firehose_client(self):
        # Double-checked lock: hot path is lock-free once built; lock only guards first build.
        if self._client is None:
            with self._lock:
                if self._client is None:
                    self._client = self._make_refreshable_client()
        return self._client

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
