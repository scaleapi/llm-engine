import json
import os
from typing import Any, Dict, List

from google.cloud import storage
from model_engine_server.common.config import get_model_cache_directory_name, hmi_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.core.utils.url import parse_attachment_url
from model_engine_server.domain.gateways import LLMArtifactGateway

logger = make_logger(logger_name())


class GCSLLMArtifactGateway(LLMArtifactGateway):
    """
    Concrete implementation for interacting with a filesystem backed by GCS.
    """

    def _get_gcs_client(self, kwargs) -> storage.Client:
        """
        Returns a GCS client. If desired, you could pass in project info
        or credentials via `kwargs`.
        """
        project = kwargs.get("gcp_project", os.getenv("GCP_PROJECT"))
        return storage.Client(project=project)

    def list_files(self, path: str, **kwargs) -> List[str]:
        """
        Lists all files under the path argument in GCS. The path is expected
        to be in the form 'gs://bucket/prefix'.
        """
        gcs = self._get_gcs_client(kwargs)
        parsed_remote = parse_attachment_url(path, clean_key=False)
        bucket_name = parsed_remote.bucket
        prefix = parsed_remote.key

        bucket = gcs.bucket(bucket_name)
        files = [blob.name for blob in bucket.list_blobs(prefix=prefix)]
        return files

    def download_files(self, path: str, target_path: str, overwrite=False, **kwargs) -> List[str]:
        """
        Downloads all files under the given path to the local target_path directory.
        """
        gcs = self._get_gcs_client(kwargs)
        parsed_remote = parse_attachment_url(path, clean_key=False)
        bucket_name = parsed_remote.bucket
        prefix = parsed_remote.key

        bucket = gcs.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        downloaded_files = []

        for blob in blobs:
            # Remove prefix and leading slash to derive local name
            file_path_suffix = blob.name.replace(prefix, "").lstrip("/")
            local_path = os.path.join(target_path, file_path_suffix).rstrip("/")

            if not overwrite and os.path.exists(local_path):
                downloaded_files.append(local_path)
                continue

            local_dir = os.path.dirname(local_path)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)

            logger.info(f"Downloading {blob.name} to {local_path}")
            blob.download_to_filename(local_path)
            downloaded_files.append(local_path)

        return downloaded_files

    def get_model_weights_urls(self, owner: str, model_name: str, **kwargs) -> List[str]:
        """
        Retrieves URLs for all model weight artifacts stored under the 
        prefix: hmi_config.hf_user_fine_tuned_weights_prefix / {owner} / {model_cache_name}
        """
        gcs = self._get_gcs_client(kwargs)
        prefix_base = hmi_config.hf_user_fine_tuned_weights_prefix
        if prefix_base.startswith("gs://"):
            # Strip "gs://" for prefix logic below
            prefix_base = prefix_base[5:]
        bucket_name, prefix_base = prefix_base.split("/", 1)

        model_cache_name = get_model_cache_directory_name(model_name)
        prefix = f"{prefix_base}/{owner}/{model_cache_name}"

        bucket = gcs.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)

        model_files = [f"gs://{bucket_name}/{blob.name}" for blob in blobs]
        return model_files

    def get_model_config(self, path: str, **kwargs) -> Dict[str, Any]:
        """
        Downloads a 'config.json' file from GCS located at path/config.json
        and returns it as a dictionary.
        """
        gcs = self._get_gcs_client(kwargs)
        parsed_remote = parse_attachment_url(path, clean_key=False)
        bucket_name = parsed_remote.bucket
        # The key from parse_attachment_url might be e.g. "weight_prefix/model_dir"
        # so we append "/config.json" and build a local path to download it.
        key_with_config = os.path.join(parsed_remote.key, "config.json")

        bucket = gcs.bucket(bucket_name)
        blob = bucket.blob(key_with_config)

        # Download to a tmp path and load
        filepath = os.path.join("/tmp", key_with_config.replace("/", "_"))
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        blob.download_to_filename(filepath)

        with open(filepath, "r") as f:
            return json.load(f) 