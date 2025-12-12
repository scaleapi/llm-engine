import os
from unittest import mock

import pytest
from model_engine_server.infra.gateways.s3_utils import get_s3_client, get_s3_resource


@pytest.fixture
def mock_infra_config_aws():
    with mock.patch("model_engine_server.infra.gateways.s3_utils.infra_config") as mock_config:
        mock_config.return_value.cloud_provider = "aws"
        yield mock_config


@pytest.fixture
def mock_infra_config_onprem():
    with mock.patch("model_engine_server.infra.gateways.s3_utils.infra_config") as mock_config:
        config_instance = mock.Mock()
        config_instance.cloud_provider = "onprem"
        config_instance.s3_endpoint_url = "http://minio:9000"
        config_instance.s3_addressing_style = "path"
        mock_config.return_value = config_instance
        yield mock_config


@mock.patch("model_engine_server.infra.gateways.s3_utils.boto3.Session")
def test_get_s3_client_aws(mock_session, mock_infra_config_aws):
    mock_client = mock.Mock()
    mock_session.return_value.client.return_value = mock_client

    result = get_s3_client({"aws_profile": "test-profile"})

    assert result == mock_client
    mock_session.assert_called_with(profile_name="test-profile")
    mock_session.return_value.client.assert_called_with("s3")


@mock.patch("model_engine_server.infra.gateways.s3_utils.boto3.Session")
def test_get_s3_client_aws_no_profile(mock_session, mock_infra_config_aws):
    mock_client = mock.Mock()
    mock_session.return_value.client.return_value = mock_client

    result = get_s3_client()

    assert result == mock_client
    mock_session.assert_called_with()


@mock.patch("model_engine_server.infra.gateways.s3_utils.boto3.Session")
def test_get_s3_client_onprem(mock_session, mock_infra_config_onprem):
    mock_client = mock.Mock()
    mock_session.return_value.client.return_value = mock_client

    result = get_s3_client()

    assert result == mock_client
    mock_session.assert_called_with()
    call_kwargs = mock_session.return_value.client.call_args
    assert call_kwargs[0][0] == "s3"
    assert "endpoint_url" in call_kwargs[1]
    assert call_kwargs[1]["endpoint_url"] == "http://minio:9000"


@mock.patch("model_engine_server.infra.gateways.s3_utils.boto3.Session")
def test_get_s3_client_onprem_env_endpoint(mock_session):
    with mock.patch("model_engine_server.infra.gateways.s3_utils.infra_config") as mock_config:
        config_instance = mock.Mock()
        config_instance.cloud_provider = "onprem"
        config_instance.s3_endpoint_url = None
        config_instance.s3_addressing_style = "path"
        mock_config.return_value = config_instance

        with mock.patch.dict(os.environ, {"S3_ENDPOINT_URL": "http://env-minio:9000"}):
            mock_client = mock.Mock()
            mock_session.return_value.client.return_value = mock_client

            result = get_s3_client()

            assert result == mock_client
            call_kwargs = mock_session.return_value.client.call_args
            assert call_kwargs[1]["endpoint_url"] == "http://env-minio:9000"


@mock.patch("model_engine_server.infra.gateways.s3_utils.boto3.Session")
def test_get_s3_resource_aws(mock_session, mock_infra_config_aws):
    mock_resource = mock.Mock()
    mock_session.return_value.resource.return_value = mock_resource

    result = get_s3_resource({"aws_profile": "test-profile"})

    assert result == mock_resource
    mock_session.assert_called_with(profile_name="test-profile")
    mock_session.return_value.resource.assert_called_with("s3")


@mock.patch("model_engine_server.infra.gateways.s3_utils.boto3.Session")
def test_get_s3_resource_onprem(mock_session, mock_infra_config_onprem):
    mock_resource = mock.Mock()
    mock_session.return_value.resource.return_value = mock_resource

    result = get_s3_resource()

    assert result == mock_resource
    call_kwargs = mock_session.return_value.resource.call_args
    assert call_kwargs[0][0] == "s3"
    assert "endpoint_url" in call_kwargs[1]
    assert call_kwargs[1]["endpoint_url"] == "http://minio:9000"
