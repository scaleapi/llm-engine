import pytest
from model_engine_server.common.dtos.tasks import EndpointPredictV1Request
from pydantic import ValidationError


@pytest.mark.parametrize(
    "destination_path",
    [
        pytest.param(None, id="none"),
        pytest.param("/predict", id="predict"),
        pytest.param("/v1/chat/completions", id="nested"),
        pytest.param("/", id="root"),
    ],
)
def test_destination_path_accepts_paths(destination_path):
    request = EndpointPredictV1Request(destination_path=destination_path)
    assert request.destination_path == destination_path


@pytest.mark.parametrize(
    "destination_path",
    [
        pytest.param("@evil.com/x", id="userinfo"),
        pytest.param(":9999/x", id="port"),
        pytest.param(".evil.com/x", id="host-suffix"),
        pytest.param("//host", id="empty-segment"),
        pytest.param("predict", id="no-leading-slash"),
        pytest.param("/predict?x=1", id="query"),
    ],
)
def test_destination_path_rejects_host_changes(destination_path):
    with pytest.raises(ValidationError):
        EndpointPredictV1Request(destination_path=destination_path)
