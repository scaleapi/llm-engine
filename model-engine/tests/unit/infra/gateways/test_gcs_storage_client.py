from unittest import mock

import pytest
from model_engine_server.infra.gateways.gcs_storage_client import get_gcs_sync_client, parse_gcs_uri


def test_parse_gcs_uri_gs_scheme():
    bucket, key = parse_gcs_uri("gs://my-bucket/path/to/object")
    assert bucket == "my-bucket"
    assert key == "path/to/object"


def test_parse_gcs_uri_https_scheme():
    bucket, key = parse_gcs_uri("https://storage.googleapis.com/my-bucket/some/key")
    assert bucket == "my-bucket"
    assert key == "some/key"


def test_parse_gcs_uri_invalid():
    with pytest.raises(ValueError, match="Invalid GCS URI"):
        parse_gcs_uri("http://example.com/bucket/key")


def test_parse_gcs_uri_no_key():
    with pytest.raises(ValueError, match="Invalid GCS URI"):
        parse_gcs_uri("gs://bucket-only")


@mock.patch("model_engine_server.infra.gateways.gcs_storage_client.storage.Client")
@mock.patch("model_engine_server.infra.gateways.gcs_storage_client.default")
def test_get_gcs_sync_client(mock_default, mock_client_cls):
    mock_creds = mock.Mock()
    mock_default.return_value = (mock_creds, "my-project")

    client = get_gcs_sync_client()

    mock_default.assert_called_once()
    mock_client_cls.assert_called_once_with(credentials=mock_creds, project="my-project")
    assert client == mock_client_cls.return_value
