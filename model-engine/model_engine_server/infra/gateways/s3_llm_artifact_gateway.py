import os
from typing import List

import boto3
from model_engine_server.common.config import get_model_cache_directory_name, hmi_config
from model_engine_server.core.utils.url import parse_attachment_url
from model_engine_server.domain.gateways import LLMArtifactGateway


class S3LLMArtifactGateway(LLMArtifactGateway):
    """
    Concrete implemention for interacting with a filesystem backed by S3.
    """

    def _get_s3_resource(self, kwargs):
        profile_name = kwargs.get("aws_profile", os.getenv("AWS_PROFILE"))
        session = boto3.Session(profile_name=profile_name)
        resource = session.resource("s3")
        return resource

    def list_files(self, path: str, **kwargs) -> List[str]:
        s3 = self._get_s3_resource(kwargs)
        parsed_remote = parse_attachment_url(path)
        bucket = parsed_remote.bucket
        key = parsed_remote.key

        s3_bucket = s3.Bucket(bucket)
        files = [obj.key for obj in s3_bucket.objects.filter(Prefix=key)]
        return files

    def download_files(self, path: str, target_path: str, overwrite=False, **kwargs) -> List[str]:
        s3 = self._get_s3_resource(kwargs)
        parsed_remote = parse_attachment_url(path)
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

            s3_bucket.download_file(obj.key, local_path)
            downloaded_files.append(local_path)
        return downloaded_files

    def get_model_weights_urls(self, owner: str, model_name: str, **kwargs) -> List[str]:
        s3 = self._get_s3_resource(kwargs)
        # parsing prefix to get S3 bucket name
        bucket_name = hmi_config.hf_user_fine_tuned_weights_prefix.replace("s3://", "").split("/")[
            0
        ]
        bucket = s3.Bucket(bucket_name)
        model_files: List[str] = []
        model_cache_name = get_model_cache_directory_name(model_name)
        # parsing prefix to get /hosted-model-inference/fine_tuned_weights
        fine_tuned_weights_prefix = "/".join(
            hmi_config.hf_user_fine_tuned_weights_prefix.split("/")[-2:]
        )
        prefix = f"{fine_tuned_weights_prefix}/{owner}/{model_cache_name}"
        for obj in bucket.objects.filter(Prefix=prefix):
            model_files.append(f"s3://{bucket_name}/{obj.key}")
        return model_files
