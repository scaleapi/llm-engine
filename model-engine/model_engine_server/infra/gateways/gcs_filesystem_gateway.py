import re
from datetime import timedelta
from typing import IO

import smart_open
from google.auth import default
from google.cloud import storage
from model_engine_server.infra.gateways.filesystem_gateway import FilesystemGateway


class GCSFilesystemGateway(FilesystemGateway):
    """
    Concrete implementation for interacting with a filesystem backed by Google Cloud Storage.
    """

    def _get_storage_client(self) -> storage.Client:
        credentials, project = default()
        return storage.Client(credentials=credentials, project=project)

    def open(self, uri: str, mode: str = "rt", **kwargs) -> IO:
        client = self._get_storage_client()
        transport_params = {"client": client}
        return smart_open.open(uri, mode, transport_params=transport_params)

    def generate_signed_url(self, uri: str, expiration: int = 3600, **kwargs) -> str:
        # Parse gs://bucket/key format
        match = re.search(r"^gs://([^/]+)/(.*?)$", uri)
        if not match:
            # Try https://storage.googleapis.com/bucket/key format
            match = re.search(r"^https://storage\.googleapis\.com/([^/]+)/(.*?)$", uri)
        assert match, f"Invalid GCS URI: {uri}"

        bucket_name, blob_name = match.group(1), match.group(2)

        client = self._get_storage_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        signed_url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(seconds=expiration),
            method="GET",
            **kwargs,
        )
        return signed_url
