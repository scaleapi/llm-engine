from azure.core.exceptions import ResourceExistsError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from celery.backends.azureblockblob import AzureBlockBlobBackend as DefaultAzureBlockBlobBackend
from kombu.utils import cached_property


class AzureBlockBlobBackend(DefaultAzureBlockBlobBackend):
    @cached_property
    def _blob_service_client(self):
        client = BlobServiceClient(
            f"https://{self._connection_string}.blob.core.windows.net",
            credential=DefaultAzureCredential(),
            connection_timeout=self._connection_timeout,
            read_timeout=self._read_timeout,
        )

        try:
            client.create_container(name=self._container_name)
        except ResourceExistsError:
            pass

        return client
