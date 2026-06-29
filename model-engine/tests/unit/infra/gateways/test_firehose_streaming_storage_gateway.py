from datetime import datetime, timedelta, timezone
from unittest import mock

import pytest
from model_engine_server.domain.exceptions import StreamPutException
from model_engine_server.inference.infra.gateways.firehose_streaming_storage_gateway import (
    _FIREHOSE_MAX_POOL_CONNECTIONS,
    FirehoseStreamingStorageGateway,
)

GW = "model_engine_server.inference.infra.gateways.firehose_streaming_storage_gateway"
STREAM = "fake-stream"
RECORD = {"RESPONSE_BODY": {"task_id": "fake-task-id"}}
OK = {"RecordId": "rid", "ResponseMetadata": {"HTTPStatusCode": 200}}
ERR = {"RecordId": "rid", "ResponseMetadata": {"HTTPStatusCode": 500}}


@pytest.fixture
def gateway():
    return FirehoseStreamingStorageGateway()


def _firehose(put_return):
    client = mock.Mock()
    client.put_record.return_value = put_return
    return client


def _patch_boto(firehose, expiry):
    # Route boto3.client by service: "sts" returns a stub whose assume_role yields creds expiring
    # at `expiry`; anything else ("firehose") returns the given firehose stub. The real
    # _build_client runs, so the assume_role -> client wiring is exercised, not stubbed out.
    sts = mock.Mock()
    sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "ak",
            "SecretAccessKey": "sk",
            "SessionToken": "tok",
            "Expiration": expiry,
        }
    }
    return sts, mock.patch(
        f"{GW}.boto3.client", side_effect=lambda svc, **_: sts if svc == "sts" else firehose
    )


def _valid():
    return datetime.now(timezone.utc) + timedelta(hours=1)


def test_put_record_success(gateway):
    _, patch = _patch_boto(_firehose(OK), _valid())
    with patch:
        assert gateway.put_record(STREAM, RECORD) is OK


def test_put_record_raises_on_non_200(gateway):
    _, patch = _patch_boto(_firehose(ERR), _valid())
    with patch:
        with pytest.raises(StreamPutException):
            gateway.put_record(STREAM, RECORD)


def test_client_built_once_while_token_valid(gateway):
    # MLI-7328 regression: while the token is valid the client is built once, not per call.
    sts, patch = _patch_boto(_firehose(OK), _valid())
    with patch:
        for _ in range(25):
            gateway.put_record(STREAM, RECORD)
    assert sts.assume_role.call_count == 1


def test_client_built_with_assumed_role_creds_and_pool(gateway):
    # Wiring guard: the client uses the assumed-role session token and the sized connection pool
    # (a silent Config drop would revert to ~10 connections under gevent).
    _, patch = _patch_boto(_firehose(OK), _valid())
    with patch as boto_client:
        gateway.put_record(STREAM, RECORD)
    fh = next(c for c in boto_client.call_args_list if c.args and c.args[0] == "firehose")
    assert fh.kwargs["aws_session_token"] == "tok"
    assert fh.kwargs["config"].max_pool_connections == _FIREHOSE_MAX_POOL_CONNECTIONS


def test_rebuilds_when_token_near_expiry(gateway):
    # Token within the refresh leeway -> the next call re-assumes and rebuilds, rather than reusing
    # an about-to-expire client. (assume_role goes 1 -> 2.)
    sts, patch = _patch_boto(_firehose(OK), datetime.now(timezone.utc) + timedelta(minutes=1))
    with patch:
        gateway.put_record(STREAM, RECORD)
        assert sts.assume_role.call_count == 1
        gateway.put_record(STREAM, RECORD)
        assert sts.assume_role.call_count == 2


def test_failed_build_leaves_client_unset_and_retries(gateway):
    # A failed build must not cache a broken client; the next call retries (and can succeed).
    with mock.patch.object(
        FirehoseStreamingStorageGateway,
        "_build_client",
        side_effect=[RuntimeError("assume_role failed"), _firehose(OK)],
    ):
        with pytest.raises(RuntimeError):
            gateway.put_record(STREAM, RECORD)
        assert gateway._client is None  # not cached after a failed build
        assert gateway.put_record(STREAM, RECORD) is OK  # retried and succeeded
