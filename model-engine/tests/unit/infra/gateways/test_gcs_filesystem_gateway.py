from datetime import timedelta
from unittest import mock
from unittest.mock import AsyncMock

import pytest
from model_engine_server.infra.gateways.gcs_filesystem_gateway import GCSFilesystemGateway


@pytest.fixture
def gateway():
    return GCSFilesystemGateway()


@mock.patch("model_engine_server.infra.gateways.gcs_filesystem_gateway.smart_open")
@mock.patch("model_engine_server.infra.gateways.gcs_filesystem_gateway.get_gcs_sync_client")
def test_open(mock_get_client, mock_smart_open, gateway):
    mock_client = mock.Mock()
    mock_get_client.return_value = mock_client

    gateway.open("gs://bucket/key", mode="rb")

    mock_smart_open.open.assert_called_once_with(
        "gs://bucket/key", "rb", transport_params={"client": mock_client}
    )


@mock.patch("model_engine_server.infra.gateways.gcs_filesystem_gateway.get_gcs_sync_client")
def test_generate_signed_url(mock_get_client, gateway):
    mock_client = mock.Mock()
    mock_bucket = mock.Mock()
    mock_blob = mock.Mock()
    mock_blob.generate_signed_url.return_value = "https://signed.example.com/blob"

    mock_get_client.return_value = mock_client
    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    result = gateway.generate_signed_url("gs://my-bucket/my-key", expiration=7200)

    assert result == "https://signed.example.com/blob"
    mock_client.bucket.assert_called_once_with("my-bucket")
    mock_bucket.blob.assert_called_once_with("my-key")
    mock_blob.generate_signed_url.assert_called_once_with(
        version="v4",
        expiration=timedelta(seconds=7200),
        method="GET",
    )


@pytest.mark.asyncio
@mock.patch("model_engine_server.infra.gateways.gcs_filesystem_gateway.Storage")
async def test_async_read(mock_storage_cls, gateway):
    mock_storage = AsyncMock()
    mock_storage.__aenter__.return_value = mock_storage
    mock_storage.__aexit__.return_value = None
    mock_storage.download.return_value = b"blob content"
    mock_storage_cls.return_value = mock_storage

    result = await gateway.async_read("gs://bucket/key")
    assert result == b"blob content"
    mock_storage.download.assert_called_once_with("bucket", "key")


@pytest.mark.asyncio
@mock.patch("model_engine_server.infra.gateways.gcs_filesystem_gateway.Storage")
async def test_async_write(mock_storage_cls, gateway):
    mock_storage = AsyncMock()
    mock_storage.__aenter__.return_value = mock_storage
    mock_storage.__aexit__.return_value = None
    mock_storage_cls.return_value = mock_storage

    await gateway.async_write("gs://bucket/key", b"new content")
    mock_storage.upload.assert_called_once_with("bucket", "key", b"new content")


@pytest.mark.asyncio
@mock.patch("model_engine_server.infra.gateways.gcs_filesystem_gateway.get_gcs_sync_client")
async def test_async_generate_signed_url(mock_get_client, gateway):
    mock_client = mock.Mock()
    mock_bucket = mock.Mock()
    mock_blob = mock.Mock()
    mock_blob.generate_signed_url.return_value = "https://signed.example.com/async"

    mock_get_client.return_value = mock_client
    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    result = await gateway.async_generate_signed_url("gs://my-bucket/my-key")
    assert result == "https://signed.example.com/async"
