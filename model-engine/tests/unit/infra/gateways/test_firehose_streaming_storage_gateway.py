from unittest import mock

import pytest
from model_engine_server.domain.exceptions import StreamPutException
from model_engine_server.inference.infra.gateways.firehose_streaming_storage_gateway import (
    FirehoseStreamingStorageGateway,
)

GW = "model_engine_server.inference.infra.gateways.firehose_streaming_storage_gateway"
stream_name = "fake-stream"
fake_record = {"RESPONSE_BODY": {"task_id": "fake-task-id"}}
ok_response = {"RecordId": "fake-record-id", "ResponseMetadata": {"HTTPStatusCode": 200}}
err_response = {"RecordId": "fake-record-id", "ResponseMetadata": {"HTTPStatusCode": 500}}


@pytest.fixture
def gateway():
    return FirehoseStreamingStorageGateway()


def _fake_client(put_return):
    client = mock.Mock()
    client.put_record.return_value = put_return
    return client


def test_put_record_success(gateway):
    with mock.patch.object(
        FirehoseStreamingStorageGateway,
        "_make_refreshable_client",
        return_value=_fake_client(ok_response),
    ):
        assert gateway.put_record(stream_name, fake_record) is ok_response


def test_put_record_raises_on_non_200(gateway):
    with mock.patch.object(
        FirehoseStreamingStorageGateway,
        "_make_refreshable_client",
        return_value=_fake_client(err_response),
    ):
        with pytest.raises(StreamPutException):
            gateway.put_record(stream_name, fake_record)


def test_client_built_once_across_calls(gateway):
    # MLI-7328 regression: client must be cached, not rebuilt per put_record. Per-call
    # construction leaked memory and OOM-killed the forwarder. Fails if that is reintroduced.
    client = _fake_client(ok_response)
    with mock.patch.object(
        FirehoseStreamingStorageGateway, "_make_refreshable_client", return_value=client
    ) as make_client:
        for _ in range(25):
            gateway.put_record(stream_name, fake_record)
    assert make_client.call_count == 1
    assert client.put_record.call_count == 25


def test_uses_refreshable_assumed_role_credentials(gateway):
    # Cached client must be backed by RefreshableCredentials so it survives the temporary STS
    # token expiry without a rebuild.
    sts = mock.Mock()
    sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "ak",
            "SecretAccessKey": "sk",
            "SessionToken": "tok",
            "Expiration": mock.Mock(isoformat=lambda: "2099-01-01T00:00:00Z"),
        }
    }
    firehose = _fake_client(ok_response)
    sts_session = mock.Mock()
    sts_session.client.return_value = sts
    firehose_session = mock.Mock()
    firehose_session.client.return_value = firehose
    with (
        mock.patch(f"{GW}.boto3.Session", side_effect=[sts_session, firehose_session]),
        mock.patch(f"{GW}.RefreshableCredentials.create_from_metadata") as create_creds,
        mock.patch(f"{GW}.get_session"),
    ):
        gateway.put_record(stream_name, fake_record)
        gateway.put_record(stream_name, fake_record)
    sts.assume_role.assert_called()
    assert create_creds.call_count == 1  # creds/client built once, then reused
    assert firehose.put_record.call_count == 2


def test_refresh_callback_reassumes_role(gateway):
    # Guards the refresh path deterministically (no 1h wait): botocore calls the refresh_using
    # callback near expiry; it must re-run assume_role and return fresh creds, so the cached
    # client keeps working past the temporary token's expiry.
    sts = mock.Mock()
    sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "ak",
            "SecretAccessKey": "sk",
            "SessionToken": "tok",
            "Expiration": mock.Mock(isoformat=lambda: "2099-01-01T00:00:00Z"),
        }
    }
    firehose = _fake_client(ok_response)
    session = mock.Mock()
    session.client.side_effect = lambda service, **kw: sts if service == "sts" else firehose
    captured = {}

    def capture_create(metadata, refresh_using, method):
        captured["refresh_using"] = refresh_using
        return mock.Mock()

    with (
        mock.patch(f"{GW}.boto3.Session", return_value=session),
        mock.patch(f"{GW}.RefreshableCredentials.create_from_metadata", side_effect=capture_create),
        mock.patch(f"{GW}.get_session"),
    ):
        gateway._get_firehose_client()
        assert sts.assume_role.call_count == 1  # initial assume on build

        refreshed = captured["refresh_using"]()  # what botocore calls near expiry
        assert sts.assume_role.call_count == 2  # re-assumed, no client rebuild
        assert refreshed["access_key"] == "ak" and "expiry_time" in refreshed


def test_failed_build_leaves_client_unset_and_retries(gateway):
    # If building the client fails, the cache must stay empty so the next call retries instead of
    # caching a broken client (the build is the load-bearing step of the MLI-7328 fix).
    good = _fake_client(ok_response)
    with mock.patch.object(
        FirehoseStreamingStorageGateway,
        "_make_refreshable_client",
        side_effect=[RuntimeError("assume_role failed"), good],
    ):
        with pytest.raises(RuntimeError):
            gateway.put_record(stream_name, fake_record)
        assert gateway._client is None  # not cached after a failed build
        assert gateway.put_record(stream_name, fake_record) is ok_response  # retried and succeeded
