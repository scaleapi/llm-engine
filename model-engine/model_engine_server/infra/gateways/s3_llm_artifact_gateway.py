import json
import os
from typing import Any, Dict, List

import boto3
from model_engine_server.common.config import get_model_cache_directory_name, hmi_config
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.core.utils.url import parse_attachment_url
from model_engine_server.domain.gateways import LLMArtifactGateway

logger = make_logger(logger_name())


class S3LLMArtifactGateway(LLMArtifactGateway):
    """
    Concrete implementation for interacting with a filesystem backed by S3.
    """

    def _get_s3_resource(self, kwargs):
        if infra_config().cloud_provider == "onprem":
            # For onprem, use explicit credentials from environment variables
            session = boto3.Session(
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=infra_config().default_region
            )
        else:
            profile_name = kwargs.get("aws_profile", os.getenv("AWS_PROFILE"))
            session = boto3.Session(profile_name=profile_name)
        
        # Support custom endpoints for S3-compatible storage (like Scality)
        # Uses standard boto3 environment variables
        endpoint_url = kwargs.get("endpoint_url") or os.getenv("AWS_ENDPOINT_URL")
        
        resource_kwargs = {}
        if endpoint_url:
            resource_kwargs["endpoint_url"] = endpoint_url
            # For custom endpoints, boto3 automatically uses AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
        
        resource = session.resource("s3", **resource_kwargs)
        return resource

    def list_files(self, path: str, **kwargs) -> List[str]:
        s3 = self._get_s3_resource(kwargs)
        parsed_remote = parse_attachment_url(path, clean_key=False)
        bucket = parsed_remote.bucket
        key = parsed_remote.key

        s3_bucket = s3.Bucket(bucket)
        files = [obj.key for obj in s3_bucket.objects.filter(Prefix=key)]
        return files

    def download_files(self, path: str, target_path: str, overwrite=False, **kwargs) -> List[str]:
        s3 = self._get_s3_resource(kwargs)
        parsed_remote = parse_attachment_url(path, clean_key=False)
        bucket = parsed_remote.bucket
        key = parsed_remote.key

        s3_bucket = s3.Bucket(bucket)
        downloaded_files: List[str] = []
        for obj in s3_bucket.objects.filter(Prefix=key):
            file_path_suffix = obj.key.replace(key, "").lstrip("/")
            local_path = os.path.join(target_path, file_path_suffix).rstrip("/")

            if not overwrite and os.path.exists(local_path):
                downloaded_files.append(local_path)
                continue

            local_dir = "/".join(local_path.split("/")[:-1])
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)

            logger.info(f"Downloading {obj.key} to {local_path}")
            s3_bucket.download_file(obj.key, local_path)
            downloaded_files.append(local_path)
        return downloaded_files

    def get_model_weights_urls(self, owner: str, model_name: str, **kwargs) -> List[str]:
        s3 = self._get_s3_resource(kwargs)
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
        return model_files

    def get_model_config(self, path: str, **kwargs) -> Dict[str, Any]:
        s3 = self._get_s3_resource(kwargs)
        parsed_remote = parse_attachment_url(path, clean_key=False)
        bucket = parsed_remote.bucket
        key = os.path.join(parsed_remote.key, "config.json")
        s3_bucket = s3.Bucket(bucket)
        filepath = os.path.join("/tmp", key).replace("/", "_")
        s3_bucket.download_file(key, filepath)
        with open(filepath, "r") as f:
            return json.load(f)
