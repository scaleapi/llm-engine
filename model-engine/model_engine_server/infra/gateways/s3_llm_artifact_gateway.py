import json
import os
from typing import Any, Dict, List

from botocore.exceptions import ClientError
from model_engine_server.common.config import get_model_cache_directory_name, hmi_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.core.utils.url import parse_attachment_url
from model_engine_server.domain.exceptions import ObjectHasInvalidValueException
from model_engine_server.domain.gateways import LLMArtifactGateway
from model_engine_server.infra.gateways.s3_utils import get_s3_resource

logger = make_logger(logger_name())

# S3 client error codes that mean the configured checkpoint path is unusable (bad bucket
# or the deployment's role can't read it) rather than a transient/internal failure. We map
# these to a user-facing message so the deploy reports an actionable cause instead of an
# opaque 500. The bucket/path is surfaced; the IAM role ARN is never exposed.
_S3_CHECKPOINT_ACCESS_ERROR_CODES = {"AccessDenied", "NoSuchBucket", "403", "404"}


class S3LLMArtifactGateway(LLMArtifactGateway):
    """
    Concrete implementation for interacting with a filesystem backed by S3.
    """

    def list_files(self, path: str, **kwargs) -> List[str]:
        s3 = get_s3_resource(kwargs)
        parsed_remote = parse_attachment_url(path, clean_key=False)
        bucket = parsed_remote.bucket
        key = parsed_remote.key

        s3_bucket = s3.Bucket(bucket)
        try:
            files = [obj.key for obj in s3_bucket.objects.filter(Prefix=key)]
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code in _S3_CHECKPOINT_ACCESS_ERROR_CODES:
                # Surface the path (not the IAM role ARN, which is in the raw error) so the
                # failure is actionable. Logged in full for server-side debugging.
                logger.warning(f"Cannot access checkpoint path {path}: {error_code}")
                raise ObjectHasInvalidValueException(
                    f"Could not read model weights at '{path}'. Check that the path exists "
                    f"and that the deployment has read access to the bucket."
                ) from e
            raise
        logger.debug(f"Listed {len(files)} files from {path}")
        return files

    def download_files(self, path: str, target_path: str, overwrite=False, **kwargs) -> List[str]:
        s3 = get_s3_resource(kwargs)
        parsed_remote = parse_attachment_url(path, clean_key=False)
        bucket = parsed_remote.bucket
        key = parsed_remote.key

        s3_bucket = s3.Bucket(bucket)
        downloaded_files: List[str] = []

        for obj in s3_bucket.objects.filter(Prefix=key):
            file_path_suffix = obj.key.replace(key, "").lstrip("/")
            local_path = os.path.join(target_path, file_path_suffix).rstrip("/")

            if not overwrite and os.path.exists(local_path):
                logger.debug(f"Skipping existing file: {local_path}")
                downloaded_files.append(local_path)
                continue

            local_dir = "/".join(local_path.split("/")[:-1])
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)

            logger.info(f"Downloading {obj.key} to {local_path}")
            s3_bucket.download_file(obj.key, local_path)
            downloaded_files.append(local_path)

        logger.info(f"Downloaded {len(downloaded_files)} files to {target_path}")
        return downloaded_files

    def get_model_weights_urls(self, owner: str, model_name: str, **kwargs) -> List[str]:
        s3 = get_s3_resource(kwargs)
        parsed_remote = parse_attachment_url(
            hmi_config.hf_user_fine_tuned_weights_prefix, clean_key=False
        )
        bucket = parsed_remote.bucket
        fine_tuned_weights_prefix = parsed_remote.key

        s3_bucket = s3.Bucket(bucket)
        model_files: List[str] = []
        model_cache_name = get_model_cache_directory_name(model_name)
        prefix = f"{fine_tuned_weights_prefix}/{owner}/{model_cache_name}"

        for obj in s3_bucket.objects.filter(Prefix=prefix):
            model_files.append(f"s3://{bucket}/{obj.key}")

        logger.debug(f"Found {len(model_files)} model weight files for {owner}/{model_name}")
        return model_files

    def get_model_config(self, path: str, **kwargs) -> Dict[str, Any]:
        s3 = get_s3_resource(kwargs)
        parsed_remote = parse_attachment_url(path, clean_key=False)
        bucket = parsed_remote.bucket
        key = os.path.join(parsed_remote.key, "config.json")

        s3_bucket = s3.Bucket(bucket)
        filepath = os.path.join("/tmp", key.replace("/", "_"))

        logger.debug(f"Downloading config from {bucket}/{key} to {filepath}")
        s3_bucket.download_file(key, filepath)

        with open(filepath, "r") as f:
            config = json.load(f)

        logger.debug(f"Loaded model config from {path}")
        return config
