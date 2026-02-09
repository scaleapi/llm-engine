from typing import List
from unittest import mock

import pytest
from model_engine_server.common.config import hmi_config
from model_engine_server.infra.gateways.gcs_llm_artifact_gateway import GCSLLMArtifactGateway

MODULE = "model_engine_server.infra.gateways.gcs_llm_artifact_gateway"


@pytest.fixture
def gcs_artifact_gateway():
    return GCSLLMArtifactGateway()


@pytest.fixture
def fake_files():
    return [
        "fake-prefix/fake1",
        "fake-prefix/fake2",
        "fake-prefix/fake3",
        "fake-prefix-ext/fake1",
    ]


def mock_gcs_client(fake_files: List[str]):
    """Create a mock GCS client that simulates bucket.list_blobs with prefix filtering."""
    mock_client = mock.Mock()
    mock_bucket = mock.Mock()

    def list_blobs_side_effect(prefix=None):
        matched = [f for f in fake_files if f.startswith(prefix or "")]
        blobs = []
        for name in matched:
            blob = mock.Mock()
            blob.name = name
            blob.download_to_filename = mock.Mock()
            blobs.append(blob)
        return blobs

    mock_bucket.list_blobs.side_effect = list_blobs_side_effect
    mock_client.bucket.return_value = mock_bucket
    return mock_client


@mock.patch(f"{MODULE}.os.makedirs", lambda *args, **kwargs: None)
def test_gcs_llm_artifact_gateway_download_folder(gcs_artifact_gateway, fake_files):
    prefix = "/".join(fake_files[0].split("/")[:-1]) + "/"
    uri_prefix = f"gs://fake-bucket/{prefix}"
    target_dir = "fake-target"

    expected_files = [
        f"{target_dir}/{file.split('/')[-1]}" for file in fake_files if file.startswith(prefix)
    ]
    with mock.patch(f"{MODULE}.get_gcs_sync_client", return_value=mock_gcs_client(fake_files)):
        assert gcs_artifact_gateway.download_files(uri_prefix, target_dir) == expected_files


@mock.patch(f"{MODULE}.os.makedirs", lambda *args, **kwargs: None)
def test_gcs_llm_artifact_gateway_download_file(gcs_artifact_gateway, fake_files):
    file = fake_files[1]
    uri = f"gs://fake-bucket/{file}"
    target = f"fake-target/{file}"

    with mock.patch(f"{MODULE}.get_gcs_sync_client", return_value=mock_gcs_client(fake_files)):
        assert gcs_artifact_gateway.download_files(uri, target) == [target]


def test_gcs_llm_artifact_gateway_get_model_weights(gcs_artifact_gateway):
    owner = "fakeuser"
    model_name = "fakemodel"
    fake_files = [f"{owner}/models--{model_name}/fake1", f"{owner}/models--{model_name}/fake2"]

    gcs_prefix = hmi_config.hf_user_fine_tuned_weights_prefix
    # Convert s3:// prefix to gs:// for GCS and extract the key portion
    weights_prefix = "/".join(gcs_prefix.replace("s3://", "").split("/")[1:])
    fake_model_weights = [f"{weights_prefix}/{file}" for file in fake_files]
    expected_model_files = [f"{gcs_prefix}/{file}" for file in fake_files]
    # Replace s3:// with gs:// in expected URLs since GCS gateway produces gs:// URLs
    expected_model_files = [f.replace("s3://", "gs://") for f in expected_model_files]

    with mock.patch(
        f"{MODULE}.get_gcs_sync_client", return_value=mock_gcs_client(fake_model_weights)
    ):
        assert (
            gcs_artifact_gateway.get_model_weights_urls(owner, model_name) == expected_model_files
        )


def test_gcs_llm_artifact_gateway_get_model_config(gcs_artifact_gateway):
    config_data = '{"model_type": "llama", "hidden_size": 4096}'
    mock_client = mock.Mock()
    mock_bucket = mock.Mock()
    mock_blob = mock.Mock()
    mock_blob.download_as_text.return_value = config_data

    mock_client.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    with mock.patch(f"{MODULE}.get_gcs_sync_client", return_value=mock_client):
        result = gcs_artifact_gateway.get_model_config("gs://fake-bucket/models/llama/")
        assert result == {"model_type": "llama", "hidden_size": 4096}
