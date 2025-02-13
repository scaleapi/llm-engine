import os
import re
from typing import IO, Optional, Dict

import smart_open
from google.cloud import storage
from model_engine_server.infra.gateways.filesystem_gateway import FilesystemGateway


class GCSFilesystemGateway(FilesystemGateway):
    """
    Concrete implementation for interacting with Google Cloud Storage.
    """

    def get_storage_client(self, kwargs: Optional[Dict]) -> storage.Client:
        """
        Retrieve or create a Google Cloud Storage client. Could optionally
        utilize environment variables or passed-in credentials.
        """
        project = kwargs.get("gcp_project", os.getenv("GCP_PROJECT"))
        return storage.Client(project=project)

    def open(self, uri: str, mode: str = "rt", **kwargs) -> IO:
        """
        Uses smart_open to handle reading/writing to GCS.
        """
        # The `transport_params` is how smart_open passes in the storage client
        client = self.get_storage_client(kwargs)
        transport_params = {"client": client}
        return smart_open.open(uri, mode, transport_params=transport_params)

    def generate_signed_url(self, uri: str, expiration: int = 3600, **kwargs) -> str:
        """
        Generate a signed URL for the given GCS URI, valid for `expiration` seconds.
        """
        # Expecting URIs in the form: 'gs://bucket_name/some_key'
        match = re.search(r"^gs://([^/]+)/(.+)$", uri)
        if not match:
            raise ValueError(f"Invalid GCS URI: {uri}")

        bucket_name, blob_name = match.groups()
        client = self.get_storage_client(kwargs)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        return blob.generate_signed_url(expiration=expiration) 