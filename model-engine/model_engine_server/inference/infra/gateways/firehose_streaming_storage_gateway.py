import json
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import boto3
from botocore.config import Config
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.exceptions import StreamPutException
from model_engine_server.inference.domain.gateways.streaming_storage_gateway import (
    StreamingStorageGateway,
)

logger = make_logger(logger_name())

# Rebuild the cached client this long before the assumed-role token actually expires.
_CLIENT_REFRESH_LEEWAY = timedelta(minutes=10)


def _firehose_max_pool_connections() -> int:
    # Shared client pool size under gevent; set >= worker concurrency. Bad/non-positive values
    # fall back to the default instead of crash-looping the forwarder.
    default = 50
    raw = os.getenv("FIREHOSE_CLIENT_MAX_POOL_CONNECTIONS", str(default))
    try:
        value = int(raw)
        if value >= 1:
            return value
    except ValueError:
        pass
    logger.warning(
        "Invalid FIREHOSE_CLIENT_MAX_POOL_CONNECTIONS=%r; using default %d", raw, default
    )
    return default


_FIREHOSE_MAX_POOL_CONNECTIONS = _firehose_max_pool_connections()


class FirehoseStreamingStorageGateway(StreamingStorageGateway):
    """Stores records via AWS Kinesis Firehose.

    MLI-7328: the client is cached, not rebuilt per call. The logging hook calls put_record on
    every async task; rebuilding per call (STS assume_role + new boto clients) leaked memory and
    OOM-killed the forwarder. Do not revert to per-call construction. The Firehose stream is
    cross-account, so creds come from assume_role and are temporary; the cached client is rebuilt
    just before the token expires (~once per token lifetime, not per task) using only public
    boto3 APIs.
    """

    def __init__(self):
        self._client = None
        self._expiry = None  # assumed-role token expiry; rebuild before it
        # gevent: greenlets share this client and the build yields on assume_role; the lock stops
        # two greenlets both building. threading.Lock is greenlet-aware after monkey.patch_all().
        self._lock = threading.Lock()

    def _build_client(self):
        # Cross-account: assume the firehose role, then build a client with those temporary creds.
        # Pure: returns (client, expiry) and does not touch self, so the caller swaps both in
        # atomically and a failed (re)build never advances _expiry past a client we did not install.
        # Public boto3 API only (no botocore-internal credential injection).
        region = infra_config().default_region
        creds = boto3.client("sts", region_name=region).assume_role(
            RoleArn=infra_config().firehose_role_arn,
            RoleSessionName="AssumeMlLoggingRoleSession",
        )["Credentials"]
        client = boto3.client(
            "firehose",
            region_name=region,
            aws_access_key_id=creds["AccessKeyId"],
            aws_secret_access_key=creds["SecretAccessKey"],
            aws_session_token=creds["SessionToken"],
            config=Config(max_pool_connections=_FIREHOSE_MAX_POOL_CONNECTIONS),
        )
        return client, creds["Expiration"]  # botocore returns a tz-aware datetime

    def _needs_rebuild(self) -> bool:
        if self._expiry is None:
            return True
        return datetime.now(timezone.utc) >= self._expiry - _CLIENT_REFRESH_LEEWAY

    def _get_firehose_client(self):
        # Rebuild only when the assumed-role token is near expiry (~once per token lifetime), never
        # per task. Assign client + expiry together so a failed (re)build leaves the prior client
        # and expiry intact and the next call retries. Double-checked lock guards the build.
        if self._client is None or self._needs_rebuild():
            with self._lock:
                if self._client is None or self._needs_rebuild():
                    self._client, self._expiry = self._build_client()
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
