import time
from typing import IO, Callable, Iterable, Optional, Sequence

import smart_open
from google.cloud import storage
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


def sync_storage_client(**kwargs) -> storage.Client:
    # Optionally inject a project id from configuration if available.
    config = infra_config()
    if hasattr(config, "gcp_project_id"):
        kwargs.setdefault("project", config.gcp_project_id)
    return storage.Client(**kwargs)


def open(uri: str, mode: str = "rt", **kwargs) -> IO:  # pylint: disable=redefined-builtin
    if "transport_params" not in kwargs:
        kwargs["transport_params"] = {"client": sync_storage_client()}
    return smart_open.open(uri, mode, **kwargs)


def sync_storage_client_keepalive(
    gcp_client: storage.Client, buckets: Iterable[str], interval: int, is_cancelled: Callable[[], bool]
) -> None:
    """Keeps connection pool warmed up for access on list of GCP buckets.

    NOTE: :param:`is_cancelled` **MUST BE THREADSAFE**.
    """
    while True:
        if is_cancelled():
            logger.info("Ending GCP client keepalive: cancel invoked.")
            return
        for bucket in buckets:
            try:
                # Instead of head_bucket, for GCP we obtain the bucket object and reload it.
                bucket_obj = gcp_client.bucket(bucket)
                bucket_obj.reload()  # refreshes metadata and validates connectivity
            except Exception:  # pylint:disable=broad-except
                logger.exception(
                    f"Unexpected error in keepalive loop on accessing bucket: {bucket}"
                )
        time.sleep(interval)


def s3_fileobj_exists(bucket: str, key: str, client: Optional[storage.Client] = None) -> bool:
    """
    Test if file exists in GCP storage.
    :param bucket: GCP bucket name
    :param key: Blob name or file's path within the bucket
    :param client: A google.cloud.storage.Client instance
    :return: Whether the file exists on GCP or not
    """
    if client is None:
        client = sync_storage_client()
    try:
        bucket_obj = client.bucket(bucket)
        # get_blob returns None if the blob does not exist.
        blob = bucket_obj.get_blob(key)
    except Exception as e:
        logger.exception(f"Error checking file existence in bucket {bucket} for key {key}")
        raise e
    else:
        return blob is not None
