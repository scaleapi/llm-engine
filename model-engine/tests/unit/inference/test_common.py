from unittest.mock import MagicMock, patch

import pytest
from model_engine_server.inference.common import predict_on_cloudpickle

BUCKET = "results-bucket"
PAYLOAD = "s3://results-bucket/tmp/hosted-model-inference-intermediate-inputs/0b7c1f9e-3a2d-4e5f-8a9b-1c2d3e4f5a6b"


@pytest.mark.parametrize(
    "input_location",
    [
        pytest.param("http://attacker.example/p.pkl", id="http"),
        pytest.param("file:///tmp/p.pkl", id="file"),
        pytest.param(
            "s3://other-bucket/tmp/hosted-model-inference-intermediate-inputs/x",
            id="other-bucket",
        ),
        pytest.param("s3://results-bucket/other-prefix/p.pkl", id="other-prefix"),
        pytest.param(PAYLOAD + "@other-bucket/p.pkl", id="userinfo"),
    ],
)
def test_predict_on_cloudpickle_rejects_caller_chosen_location(monkeypatch, input_location):
    monkeypatch.setenv("RESULTS_S3_BUCKET", BUCKET)
    with patch("model_engine_server.inference.common.open_wrapper") as open_wrapper:
        with pytest.raises(ValueError):
            predict_on_cloudpickle(MagicMock(), input_location)
    open_wrapper.assert_not_called()


def test_predict_on_cloudpickle_rejects_without_results_bucket(monkeypatch):
    monkeypatch.delenv("RESULTS_S3_BUCKET", raising=False)
    with patch("model_engine_server.inference.common.open_wrapper") as open_wrapper:
        with pytest.raises(ValueError):
            predict_on_cloudpickle(MagicMock(), PAYLOAD)
    open_wrapper.assert_not_called()


def test_predict_on_cloudpickle_loads_intermediate_payload(monkeypatch):
    monkeypatch.setenv("RESULTS_S3_BUCKET", BUCKET)
    predict_fn = MagicMock(return_value="out")
    with (
        patch("model_engine_server.inference.common.open_wrapper") as open_wrapper,
        patch(
            "model_engine_server.inference.common.cloudpickle.load",
            return_value={"args": [1], "kwargs": {"k": 2}},
        ),
        patch(
            "model_engine_server.inference.common._write_to_s3",
            return_value={"result_url": "s3://r"},
        ) as write_to_s3,
    ):
        assert predict_on_cloudpickle(predict_fn, PAYLOAD) == {"result_url": "s3://r"}
    open_wrapper.assert_called_once_with(PAYLOAD, "rb")
    predict_fn.assert_called_once_with(1, k=2)
    write_to_s3.assert_called_once_with("out")
