import os
from typing import List

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContainerClient
from model_engine_server.common.config import get_model_cache_directory_name, hmi_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.core.utils.url import parse_attachment_url
from model_engine_server.domain.gateways import LLMArtifactGateway

logger = make_logger(logger_name())


def _get_abs_container_client(bucket: str) -> ContainerClient:
    blob_service_client = BlobServiceClient(
        f"https://{os.getenv('ABS_ACCOUNT_NAME')}.blob.core.windows.net", DefaultAzureCredential()
    )
    return blob_service_client.get_container_client(container=bucket)


class ABSLLMArtifactGateway(LLMArtifactGateway):
    """
    Concrete implemention using Azure Blob Storage.
    """

    def list_files(self, path: str, **kwargs) -> List[str]:
        parsed_remote = parse_attachment_url(path, clean_key=False)
        bucket = parsed_remote.bucket
        key = parsed_remote.key

        container_client = _get_abs_container_client(bucket)
        return list(container_client.list_blob_names(name_starts_with=key))

    def download_files(self, path: str, target_path: str, overwrite=False, **kwargs) -> List[str]:
        parsed_remote = parse_attachment_url(path, clean_key=False)
        bucket = parsed_remote.bucket
        key = parsed_remote.key

        container_client = _get_abs_container_client(bucket)

        downloaded_files: List[str] = []
        for blob in container_client.list_blobs(name_starts_with=key):
            file_path_suffix = blob.name.replace(key, "").lstrip("/")
            local_path = os.path.join(target_path, file_path_suffix).rstrip("/")

            if not overwrite and os.path.exists(local_path):
                downloaded_files.append(local_path)
                continue

            local_dir = "/".join(local_path.split("/")[:-1])
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)

            logger.info(f"Downloading {blob.name} to {local_path}")
            with open(file=local_path, mode="wb") as f:
                f.write(container_client.download_blob(blob.name).readall())
            downloaded_files.append(local_path)
        return downloaded_files

    def get_model_weights_urls(self, owner: str, model_name: str, **kwargs) -> List[str]:
        parsed_remote = parse_attachment_url(
            hmi_config.hf_user_fine_tuned_weights_prefix, clean_key=False
        )
        account = parsed_remote.account
        bucket = parsed_remote.bucket
        fine_tuned_weights_prefix = parsed_remote.key

        container_client = _get_abs_container_client(bucket)

        model_files: List[str] = []
        model_cache_name = get_model_cache_directory_name(model_name)
        prefix = f"{fine_tuned_weights_prefix}/{owner}/{model_cache_name}"
        for blob_name in container_client.list_blob_names(name_starts_with=prefix):
            model_files.append(f"https://{account}.blob.core.windows.net/{bucket}/{blob_name}")
        return model_files
