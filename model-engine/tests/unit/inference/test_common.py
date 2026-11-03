from unittest.mock import MagicMock, patch

import pytest
from model_engine_server.common.dtos.tasks import EndpointPredictV1Request
from model_engine_server.inference.common import run_predict


@pytest.mark.parametrize(
    "request_params",
    [
        pytest.param({"cloudpickle": "http://attacker.example/p.pkl"}, id="cloudpickle-only"),
        pytest.param(
            {"cloudpickle": "s3://bucket/p.pkl", "args": {"x": 1}}, id="cloudpickle-with-args"
        ),
        pytest.param(
            {"cloudpickle": "file:///tmp/p.pkl", "url": "s3://bucket/in"}, id="cloudpickle-with-url"
        ),
    ],
)
def test_run_predict_rejects_cloudpickle(request_params):
    predict_fn = MagicMock()
    with patch("model_engine_server.inference.common.open_wrapper") as open_wrapper:
        with pytest.raises(ValueError, match="cloudpickle"):
            run_predict(predict_fn, EndpointPredictV1Request(**request_params))
    open_wrapper.assert_not_called()
    predict_fn.assert_not_called()


def test_run_predict_args_still_served():
    predict_fn = MagicMock(return_value={"y": 2})
    result = run_predict(predict_fn, EndpointPredictV1Request(args={"x": 1}))
    predict_fn.assert_called_once_with(x=1)
    assert result == {"result": '{"y": 2}'}
