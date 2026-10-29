import pytest
from model_engine_server.common.dtos.llms.model_endpoints import (
    CreateVLLMModelEndpointRequest,
    UpdateVLLMModelEndpointRequest,
)
from model_engine_server.common.dtos.model_endpoints import CreateModelEndpointV1Request
from pydantic import ValidationError


def _name_has_error(model, name: str) -> bool:
    try:
        model.model_validate({"name": name})
    except ValidationError as e:
        return any(error["loc"] == ("name",) for error in e.errors())
    return False


@pytest.mark.parametrize("model", [CreateModelEndpointV1Request, CreateVLLMModelEndpointRequest])
@pytest.mark.parametrize(
    "name,rejected",
    [
        pytest.param("my-endpoint", False, id="dashes"),
        pytest.param("test_llm_endpoint.v2", False, id="underscore-dot"),
        pytest.param("a" * 63, False, id="max-length"),
        pytest.param("a" * 64, True, id="too-long"),
        pytest.param(
            "x\n          securityContext:\n            privileged: true", True, id="newline"
        ),
        pytest.param("my endpoint", True, id="space"),
        pytest.param("-endpoint", True, id="leading-dash"),
        pytest.param("endpoint:latest", True, id="colon"),
    ],
)
def test_create_endpoint_name_must_be_k8s_label_value(model, name, rejected):
    assert _name_has_error(model, name) is rejected


@pytest.mark.parametrize(
    "field", ["public_inference", "enable_startup_metrics", "force_bundle_recreation"]
)
def test_update_llm_endpoint_omitted_fields_are_unset(field):
    assert getattr(UpdateVLLMModelEndpointRequest(), field) is None
