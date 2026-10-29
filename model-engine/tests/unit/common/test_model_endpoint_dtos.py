import pytest
from model_engine_server.common.dtos.llms.batch_completion import BatchCompletionsModelConfig
from model_engine_server.common.dtos.llms.model_endpoints import (
    CreateVLLMModelEndpointRequest,
    UpdateVLLMModelEndpointRequest,
)
from model_engine_server.common.dtos.model_endpoints import CreateModelEndpointV1Request
from pydantic import ValidationError


def _field_has_error(model, field: str, value: str) -> bool:
    try:
        model.model_validate({field: value})
    except ValidationError as e:
        return any(error["loc"] == (field,) for error in e.errors())
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
    assert _field_has_error(model, "name", name) is rejected


@pytest.mark.parametrize(
    "model,field",
    [
        pytest.param(CreateVLLMModelEndpointRequest, "model_name", id="create"),
        pytest.param(UpdateVLLMModelEndpointRequest, "model_name", id="update"),
        pytest.param(BatchCompletionsModelConfig, "model", id="batch-completions"),
    ],
)
@pytest.mark.parametrize(
    "model_name,rejected",
    [
        pytest.param("meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", False, id="hf-name"),
        pytest.param("1" * 256, False, id="max-length"),
        pytest.param("1" * 1_000_000, True, id="1mb"),
    ],
)
def test_model_name_length_is_bounded(model, field, model_name, rejected):
    assert _field_has_error(model, field, model_name) is rejected


@pytest.mark.parametrize(
    "field", ["public_inference", "enable_startup_metrics", "force_bundle_recreation"]
)
def test_update_llm_endpoint_omitted_fields_are_unset(field):
    assert getattr(UpdateVLLMModelEndpointRequest(), field) is None
