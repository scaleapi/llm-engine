import os
import time
from typing import IO, Callable, Iterable, Optional, Sequence

import boto3
import smart_open
from botocore.client import BaseClient
from model_engine_server.core.aws.roles import session
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger

logger = make_logger(logger_name())

__all__: Sequence[str] = (
    "sync_storage_client",
    # `open` should be used, but so as to not shadow the built-in, the preferred import is:
    #   >>>  storage_client.open
    # Thus, it's not included in the wildcard imports.
    "sync_storage_client_keepalive",
    "s3_fileobj_exists",
)


def sync_storage_client(**kwargs) -> BaseClient:
    # Support custom endpoints for S3-compatible storage
    endpoint_url = kwargs.get("endpoint_url") or os.getenv("AWS_ENDPOINT_URL")
    if endpoint_url and "endpoint_url" not in kwargs:
        kwargs["endpoint_url"] = endpoint_url
    
    if infra_config().cloud_provider == "onprem":
        # For onprem, use explicit credentials from environment variables
        boto3_session = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=infra_config().default_region
        )
        return boto3_session.client("s3", **kwargs)
    else:
        return session(infra_config().profile_ml_worker).client("s3", **kwargs)  # type: ignore


def open(uri: str, mode: str = "rt", **kwargs) -> IO:  # pylint: disable=redefined-builtin
    if "transport_params" not in kwargs:
        kwargs["transport_params"] = {"client": sync_storage_client()}
    return smart_open.open(uri, mode, **kwargs)


def sync_storage_client_keepalive(
    s3_client: BaseClient, buckets: Iterable[str], interval: int, is_cancelled: Callable[[], bool]
) -> None:
    """Keeps connection pool warmed up for access on list of S3 buckets.

    NOTE: :param:`is_cancelled` **MUST BE THREADSAFE**.
    """
    while True:
        if is_cancelled():
            logger.info("Ending S3 client keepalive: cancel invoked.")
            return
        for bucket in buckets:
            try:
                s3_client.head_bucket(Bucket=bucket)
            except Exception:  # pylint:disable=broad-except
                logger.exception(
                    f"Unexpected error in keepalive loop on HeadBucket(bucket={bucket})"
                )
        time.sleep(interval)


def s3_fileobj_exists(bucket: str, key: str, s3: Optional[BaseClient] = None) -> bool:
    """
    Test if file exists in s3
    :param bucket: S3 bucket
    :param key: The rest of the file's path, e.g. "x/y/z" for a file located at
        f"s3://{bucket}/x/y/z"
    :param s3: A boto3 S3 client
    :return: Whether the file exists on s3 or not
    """
    if s3 is None:
        s3 = sync_storage_client()
    try:
        # https://stackoverflow.com/questions/33842944/check-if-a-key-exists-in-a-bucket-in-s3-using-boto3
        # Retrieves metadata from an object without returning the object itself (Most efficient)
        s3.head_object(Bucket=bucket, Key=key)
    except Exception as e:  # type: ignore
        try:
            # pylint: disable=no-member
            error_code = e.response["Error"]["Code"].strip()  # type: ignore
            if error_code in ("404", "NoSuchKey"):
                return False
        except (NameError, KeyError):
            pass
        raise e
    else:
        return True
