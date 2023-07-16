import pytest
from llm_engine_server.domain.entities import (
    CallbackAuth,
    CallbackBasicAuth,
    ModelBundle,
    ModelEndpointConfig,
)


@pytest.mark.parametrize(
    "model_endpoint_config",
    [
        ModelEndpointConfig(
            endpoint_name="test_endpoint",
            bundle_name="test_bundle",
            post_inference_hooks=["test_hook"],
            user_id="test_user",
            default_callback_url="test_url",
        ),
        ModelEndpointConfig(
            endpoint_name="test_endpoint_2",
            bundle_name="test_bundle",
            post_inference_hooks=["test_hook"],
            user_id="test_user",
            default_callback_auth=CallbackAuth(
                __root__=CallbackBasicAuth(
                    kind="basic", username="test_user", password="test_password"
                )
            ),
        ),
    ],
)
def test_model_endpoint_config_serialization(
    model_endpoint_config: ModelEndpointConfig,
):
    serialized_config = model_endpoint_config.serialize()
    deserialized_config = ModelEndpointConfig.deserialize(serialized_config)
    assert model_endpoint_config == deserialized_config


def test_model_bundle_is_runnable(
    model_bundle_1: ModelBundle,
    model_bundle_2: ModelBundle,
    model_bundle_3: ModelBundle,
    model_bundle_4: ModelBundle,
):
    assert not model_bundle_1.is_runnable()
    assert not model_bundle_2.is_runnable()
    assert not model_bundle_3.is_runnable()
    assert model_bundle_4.is_runnable()
