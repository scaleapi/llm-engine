import json
from unittest.mock import MagicMock, patch

import pytest
from model_engine_server.common.dtos.tasks import EndpointPredictV1Request
from model_engine_server.domain.entities import CallbackAuth
from model_engine_server.domain.entities.model_endpoint_entity import ModelEndpointType
from model_engine_server.inference.post_inference_hooks import LoggingHook


@pytest.mark.parametrize(
    "callback_auth,secret",
    [
        pytest.param(
            {"kind": "basic", "username": "user", "password": "hunter2"}, "hunter2", id="basic"
        ),
        pytest.param(
            {"kind": "mtls", "cert": "cert-data", "key": "private-key"}, "private-key", id="mtls"
        ),
    ],
)
def test_callback_auth_secrets_not_logged(callback_auth, secret):
    request = EndpointPredictV1Request(
        args={"x": 1}, callback_auth=CallbackAuth.model_validate(callback_auth)
    )
    assert secret not in repr(request)
    assert secret not in str(request)

    streaming_storage_gateway = MagicMock()
    hook = LoggingHook(
        endpoint_name="endpoint",
        bundle_name="bundle",
        user_id="user",
        endpoint_id="end_1",
        endpoint_type=ModelEndpointType.SYNC,
        bundle_id="bun_1",
        labels={"team": "infra"},
        streaming_storage_gateway=streaming_storage_gateway,
    )
    with patch(
        "model_engine_server.inference.post_inference_hooks.infra_config"
    ) as mock_infra_config:
        mock_infra_config.return_value.firehose_stream_name = "stream"
        hook.handle(request_payload=request, response={"result": 1}, task_id="task_1")

    record = streaming_storage_gateway.put_record.call_args.kwargs["record"]
    request_body = json.loads(record["REQUEST_BODY"])
    assert "callback_auth" not in request_body
    assert request_body["args"] == {"x": 1}
