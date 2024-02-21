from unittest import mock

import pytest
from model_engine_server.domain.exceptions import StreamPutException
from model_engine_server.inference.infra.gateways.firehose_streaming_storage_gateway import (
    FirehoseStreamingStorageGateway,
)

stream_name = "fake-stream"

return_value = {
    "RecordId": "fake-record-id",
    "Encrypted": False,
    "ResponseMetadata": {"HTTPStatusCode": 200},
}


@pytest.fixture
def streaming_storage_gateway():
    gateway = FirehoseStreamingStorageGateway()
    return gateway


@pytest.fixture
def fake_record():
    return {"RESPONSE_BODY": {"task_id": "fake-task-id"}}


def mock_sts_client(*args, **kwargs):
    mock_client = mock.Mock()
    mock_client.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "fake-access-key-id",
            "SecretAccessKey": "fake-secret-access-key",
            "SessionToken": "fake-session-token",
        }
    }
    return mock_client


def mock_firehose_client(*args, **kwargs):
    mock_client = mock.Mock()
    mock_client.put_record.return_value = return_value
    return mock_client


def mock_session(*args, **kwargs):
    mock_session_obj = mock.Mock()
    mock_firehose = mock_firehose_client()
    mock_session_obj.client.return_value = mock_firehose
    return mock_session_obj


def mock_firehose_client_with_exception(*args, **kwargs):
    mock_client = mock.Mock()
    mock_client.put_record.return_value = {
        "RecordId": "fake-record-id",
        "Encrypted": False,
        "ResponseMetadata": {"HTTPStatusCode": 500},
    }
    return mock_client


def mock_session_with_exception(*args, **kwargs):
    mock_session_obj = mock.Mock()
    mock_firehose = mock_firehose_client_with_exception()

    mock_session_obj.client.return_value = mock_firehose

    return mock_session_obj


def test_firehose_streaming_storage_gateway_put_record(streaming_storage_gateway, fake_record):
    with mock.patch(
        "model_engine_server.inference.infra.gateways.firehose_streaming_storage_gateway.boto3.client",
        mock_sts_client,
    ), mock.patch(
        "model_engine_server.inference.infra.gateways.firehose_streaming_storage_gateway.boto3.Session",
        mock_session,
    ):
        assert streaming_storage_gateway.put_record(stream_name, fake_record) is return_value


def test_firehose_streaming_storage_gateway_put_record_with_exception(
    streaming_storage_gateway, fake_record
):
    with mock.patch(
        "model_engine_server.inference.infra.gateways.firehose_streaming_storage_gateway.boto3.client",
        mock_sts_client,
    ), mock.patch(
        "model_engine_server.inference.infra.gateways.firehose_streaming_storage_gateway.boto3.Session",
        mock_session_with_exception,
    ):
        with pytest.raises(StreamPutException):
            streaming_storage_gateway.put_record(stream_name, fake_record)
