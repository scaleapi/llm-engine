from typing import Any, Tuple

from model_engine_server.domain.entities import ModelBundle, ModelEndpoint


def test_model_endpoints_schema_success(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    model_endpoint_2: Tuple[ModelEndpoint, Any],
    test_api_key: str,
    get_test_client_wrapper,
):
    assert model_endpoint_1[0].infra_state is not None
    assert model_endpoint_2[0].infra_state is not None
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={
            model_bundle_1_v1[0].id: model_bundle_1_v1[0],
        },
        fake_model_endpoint_record_repository_contents={
            model_endpoint_1[0].record.id: model_endpoint_1[0].record,
            model_endpoint_2[0].record.id: model_endpoint_2[0].record,
        },
        fake_model_endpoint_infra_gateway_contents={
            model_endpoint_1[0].infra_state.deployment_name: model_endpoint_1[0].infra_state,
            model_endpoint_2[0].infra_state.deployment_name: model_endpoint_2[0].infra_state,
        },
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response = client.get("/v1/model-endpoints-schema.json", auth=(test_api_key, ""))

    assert response.status_code == 200
    assert response.json().get("components") is not None


def test_model_endpoints_schema_no_auth_returns_401(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    model_endpoint_2: Tuple[ModelEndpoint, Any],
    get_test_client_wrapper,
):
    assert model_endpoint_1[0].infra_state is not None
    assert model_endpoint_2[0].infra_state is not None
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={
            model_bundle_1_v1[0].id: model_bundle_1_v1[0],
        },
        fake_model_endpoint_record_repository_contents={
            model_endpoint_1[0].record.id: model_endpoint_1[0].record,
            model_endpoint_2[0].record.id: model_endpoint_2[0].record,
        },
        fake_model_endpoint_infra_gateway_contents={
            model_endpoint_1[0].infra_state.deployment_name: model_endpoint_1[0].infra_state,
            model_endpoint_2[0].infra_state.deployment_name: model_endpoint_2[0].infra_state,
        },
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response = client.get("/v1/model-endpoints-schema.json")

    assert response.status_code == 401
    assert response.json().get("components") is None


def test_model_endpoints_api_success(test_api_key: str, simple_client):
    response = simple_client.get("/v1/model-endpoints-api", auth=(test_api_key, ""))
    assert response.status_code == 200


def test_model_endpoints_api_no_auth_returns_401(simple_client):
    response = simple_client.get("/v1/model-endpoints-api")
    assert response.status_code == 401
