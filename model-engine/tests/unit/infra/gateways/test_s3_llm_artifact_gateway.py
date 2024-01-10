from typing import List
from unittest import mock

import pytest
from model_engine_server.common.config import hmi_config
from model_engine_server.infra.gateways.s3_llm_artifact_gateway import S3LLMArtifactGateway


@pytest.fixture
def llm_artifact_gateway():
    gateway = S3LLMArtifactGateway()
    return gateway


@pytest.fixture
def fake_files():
    return ["fake-prefix/fake1", "fake-prefix/fake2", "fake-prefix/fake3", "fake-prefix-ext/fake1"]


def mock_boto3_session(fake_files: List[str]):
    mock_session = mock.Mock()
    mock_bucket = mock.Mock()
    mock_objects = mock.Mock()

    def filter_files(*args, **kwargs):
        prefix = kwargs["Prefix"]
        return [mock.Mock(key=file) for file in fake_files if file.startswith(prefix)]

    mock_session.return_value.resource.return_value.Bucket.return_value = mock_bucket
    mock_bucket.objects = mock_objects
    mock_objects.filter.side_effect = filter_files

    mock_bucket.download_file.return_value = None
    return mock_session


@mock.patch(
    "model_engine_server.infra.gateways.s3_llm_artifact_gateway.os.makedirs",
    lambda *args, **kwargs: None,  # noqa
)
def test_s3_llm_artifact_gateway_download_folder(llm_artifact_gateway, fake_files):
    prefix = "/".join(fake_files[0].split("/")[:-1]) + "/"
    uri_prefix = f"s3://fake-bucket/{prefix}"
    target_dir = "fake-target"

    expected_files = [
        f"{target_dir}/{file.split('/')[-1]}" for file in fake_files if file.startswith(prefix)
    ]
    with mock.patch(
        "model_engine_server.infra.gateways.s3_llm_artifact_gateway.boto3.Session",
        mock_boto3_session(fake_files),
    ):
        assert llm_artifact_gateway.download_files(uri_prefix, target_dir) == expected_files


@mock.patch(
    "model_engine_server.infra.gateways.s3_llm_artifact_gateway.os.makedirs",
    lambda *args, **kwargs: None,  # noqa
)
def test_s3_llm_artifact_gateway_download_file(llm_artifact_gateway, fake_files):
    file = fake_files[1]
    uri = f"s3://fake-bucket/{file}"
    target = f"fake-target/{file}"

    with mock.patch(
        "model_engine_server.infra.gateways.s3_llm_artifact_gateway.boto3.Session",
        mock_boto3_session(fake_files),
    ):
        assert llm_artifact_gateway.download_files(uri, target) == [target]


def test_s3_llm_artifact_gateway_get_model_weights(llm_artifact_gateway):
    owner = "fakeuser"
    model_name = "fakemodel"
    fake_files = [f"{owner}/models--{model_name}/fake1", f"{owner}/models--{model_name}/fake2"]

    s3_prefix = hmi_config.hf_user_fine_tuned_weights_prefix
    weights_prefix = "/".join(s3_prefix.replace("s3://", "").split("/")[1:])
    fake_model_weights = [f"{weights_prefix}/{file}" for file in fake_files]
    expected_model_files = [f"{s3_prefix}/{file}" for file in fake_files]
    with mock.patch(
        "model_engine_server.infra.gateways.s3_llm_artifact_gateway.boto3.Session",
        mock_boto3_session(fake_model_weights),
    ):
        assert (
            llm_artifact_gateway.get_model_weights_urls(owner, model_name) == expected_model_files
        )
