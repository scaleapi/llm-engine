from unittest import mock
from unittest.mock import AsyncMock

import pytest
from model_engine_server.infra.gateways.gcs_file_storage_gateway import GCSFileStorageGateway

FAKE_BUCKET = "test-bucket"


@pytest.fixture
def gateway():
    return GCSFileStorageGateway()


def _mock_storage():
    """Create a mock gcloud.aio.storage.Storage that works as an async context manager."""
    mock_storage = AsyncMock()
    mock_storage.__aenter__.return_value = mock_storage
    mock_storage.__aexit__.return_value = None
    return mock_storage


@pytest.mark.asyncio
@mock.patch(
    "model_engine_server.infra.gateways.gcs_file_storage_gateway._generate_signed_url_sync",
    return_value="https://signed-url.example.com/file",
)
@mock.patch("model_engine_server.infra.gateways.gcs_file_storage_gateway.infra_config")
async def test_get_url_from_id(mock_infra_config, mock_signed_url, gateway):
    mock_infra_config.return_value.s3_bucket = FAKE_BUCKET
    result = await gateway.get_url_from_id("owner1", "file1.txt")
    assert result == "https://signed-url.example.com/file"
    mock_signed_url.assert_called_once()


@pytest.mark.asyncio
@mock.patch("model_engine_server.infra.gateways.gcs_file_storage_gateway.infra_config")
@mock.patch("model_engine_server.infra.gateways.gcs_file_storage_gateway.Storage")
async def test_get_file_success(mock_storage_cls, mock_infra_config, gateway):
    mock_infra_config.return_value.s3_bucket = FAKE_BUCKET
    storage = _mock_storage()
    storage.download_metadata.return_value = {
        "size": "2048",
        "updated": "2024-06-15T10:30:00.000Z",
    }
    mock_storage_cls.return_value = storage

    result = await gateway.get_file("owner1", "file1.txt")

    assert result is not None
    assert result.id == "file1.txt"
    assert result.filename == "file1.txt"
    assert result.size == 2048
    assert result.owner == "owner1"
    storage.download_metadata.assert_called_once_with(FAKE_BUCKET, "owner1/file1.txt")


@pytest.mark.asyncio
@mock.patch("model_engine_server.infra.gateways.gcs_file_storage_gateway.infra_config")
@mock.patch("model_engine_server.infra.gateways.gcs_file_storage_gateway.Storage")
async def test_get_file_not_found(mock_storage_cls, mock_infra_config, gateway):
    mock_infra_config.return_value.s3_bucket = FAKE_BUCKET
    storage = _mock_storage()
    storage.download_metadata.side_effect = Exception("Not Found")
    mock_storage_cls.return_value = storage

    result = await gateway.get_file("owner1", "nonexistent.txt")
    assert result is None


@pytest.mark.asyncio
@mock.patch("model_engine_server.infra.gateways.gcs_file_storage_gateway.infra_config")
@mock.patch("model_engine_server.infra.gateways.gcs_file_storage_gateway.Storage")
async def test_get_file_content_success(mock_storage_cls, mock_infra_config, gateway):
    mock_infra_config.return_value.s3_bucket = FAKE_BUCKET
    storage = _mock_storage()
    storage.download.return_value = b"file content here"
    mock_storage_cls.return_value = storage

    result = await gateway.get_file_content("owner1", "file1.txt")
    assert result == "file content here"
    storage.download.assert_called_once_with(FAKE_BUCKET, "owner1/file1.txt")


@pytest.mark.asyncio
@mock.patch("model_engine_server.infra.gateways.gcs_file_storage_gateway.infra_config")
@mock.patch("model_engine_server.infra.gateways.gcs_file_storage_gateway.Storage")
async def test_get_file_content_not_found(mock_storage_cls, mock_infra_config, gateway):
    mock_infra_config.return_value.s3_bucket = FAKE_BUCKET
    storage = _mock_storage()
    storage.download.side_effect = Exception("Not Found")
    mock_storage_cls.return_value = storage

    result = await gateway.get_file_content("owner1", "nonexistent.txt")
    assert result is None


@pytest.mark.asyncio
@mock.patch("model_engine_server.infra.gateways.gcs_file_storage_gateway.infra_config")
@mock.patch("model_engine_server.infra.gateways.gcs_file_storage_gateway.Storage")
async def test_upload_file(mock_storage_cls, mock_infra_config, gateway):
    mock_infra_config.return_value.s3_bucket = FAKE_BUCKET
    storage = _mock_storage()
    mock_storage_cls.return_value = storage

    result = await gateway.upload_file("owner1", "upload.txt", b"upload data")
    assert result == "upload.txt"
    storage.upload.assert_called_once_with(FAKE_BUCKET, "owner1/upload.txt", b"upload data")


@pytest.mark.asyncio
@mock.patch("model_engine_server.infra.gateways.gcs_file_storage_gateway.infra_config")
@mock.patch("model_engine_server.infra.gateways.gcs_file_storage_gateway.Storage")
async def test_delete_file_success(mock_storage_cls, mock_infra_config, gateway):
    mock_infra_config.return_value.s3_bucket = FAKE_BUCKET
    storage = _mock_storage()
    mock_storage_cls.return_value = storage

    result = await gateway.delete_file("owner1", "file1.txt")
    assert result is True
    storage.delete.assert_called_once_with(FAKE_BUCKET, "owner1/file1.txt")


@pytest.mark.asyncio
@mock.patch("model_engine_server.infra.gateways.gcs_file_storage_gateway.infra_config")
@mock.patch("model_engine_server.infra.gateways.gcs_file_storage_gateway.Storage")
async def test_delete_file_not_found(mock_storage_cls, mock_infra_config, gateway):
    mock_infra_config.return_value.s3_bucket = FAKE_BUCKET
    storage = _mock_storage()
    storage.delete.side_effect = Exception("Not Found")
    mock_storage_cls.return_value = storage

    result = await gateway.delete_file("owner1", "nonexistent.txt")
    assert result is False


@pytest.mark.asyncio
@mock.patch("model_engine_server.infra.gateways.gcs_file_storage_gateway.infra_config")
@mock.patch("model_engine_server.infra.gateways.gcs_file_storage_gateway.Storage")
async def test_list_files(mock_storage_cls, mock_infra_config, gateway):
    mock_infra_config.return_value.s3_bucket = FAKE_BUCKET
    storage = _mock_storage()
    storage.list_objects.return_value = {
        "items": [
            {"name": "owner1/file1.txt", "size": "100", "updated": "2024-01-01T00:00:00Z"},
            {"name": "owner1/file2.txt", "size": "200", "updated": "2024-01-02T00:00:00Z"},
        ]
    }
    mock_storage_cls.return_value = storage

    result = await gateway.list_files("owner1")
    assert len(result) == 2
    assert result[0].id == "file1.txt"
    assert result[0].size == 100
    assert result[1].id == "file2.txt"
    assert result[1].size == 200
    storage.list_objects.assert_called_once_with(FAKE_BUCKET, params={"prefix": "owner1"})


@pytest.mark.asyncio
@mock.patch("model_engine_server.infra.gateways.gcs_file_storage_gateway.infra_config")
@mock.patch("model_engine_server.infra.gateways.gcs_file_storage_gateway.Storage")
async def test_list_files_pagination(mock_storage_cls, mock_infra_config, gateway):
    mock_infra_config.return_value.s3_bucket = FAKE_BUCKET
    storage = _mock_storage()
    storage.list_objects.side_effect = [
        {
            "items": [
                {"name": "owner1/file1.txt", "size": "100", "updated": "2024-01-01T00:00:00Z"},
            ],
            "nextPageToken": "token123",
        },
        {
            "items": [
                {"name": "owner1/file2.txt", "size": "200", "updated": "2024-01-02T00:00:00Z"},
            ],
        },
    ]
    mock_storage_cls.return_value = storage

    result = await gateway.list_files("owner1")
    assert len(result) == 2
    assert result[0].id == "file1.txt"
    assert result[1].id == "file2.txt"
    assert storage.list_objects.call_count == 2
    storage.list_objects.assert_any_call(FAKE_BUCKET, params={"prefix": "owner1"})
    storage.list_objects.assert_any_call(
        FAKE_BUCKET, params={"prefix": "owner1", "pageToken": "token123"}
    )


@pytest.mark.asyncio
@mock.patch("model_engine_server.infra.gateways.gcs_file_storage_gateway.infra_config")
@mock.patch("model_engine_server.infra.gateways.gcs_file_storage_gateway.Storage")
async def test_list_files_empty(mock_storage_cls, mock_infra_config, gateway):
    mock_infra_config.return_value.s3_bucket = FAKE_BUCKET
    storage = _mock_storage()
    storage.list_objects.return_value = {}
    mock_storage_cls.return_value = storage

    result = await gateway.list_files("owner1")
    assert result == []
