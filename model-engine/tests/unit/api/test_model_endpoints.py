import json
from typing import Any, Dict, Tuple

import pytest
from fastapi.testclient import TestClient
from model_engine_server.common.dtos.model_endpoints import GetModelEndpointV1Response
from model_engine_server.domain.entities import ModelBundle, ModelEndpoint, ModelEndpointStatus


def test_create_model_endpoint_success(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    create_model_endpoint_request_sync: Dict[str, Any],
    create_model_endpoint_request_async: Dict[str, Any],
    test_api_key: str,
    get_test_client_wrapper,
):
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={
            model_bundle_1_v1[0].id: model_bundle_1_v1[0],
        },
        fake_model_endpoint_record_repository_contents={},
        fake_model_endpoint_infra_gateway_contents={},
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response_1 = client.post(
        "/v1/model-endpoints",
        auth=(test_api_key, ""),
        json=create_model_endpoint_request_sync,
    )
    assert response_1.status_code == 200

    response_2 = client.post(
        "/v1/model-endpoints",
        auth=(test_api_key, ""),
        json=create_model_endpoint_request_async,
    )
    assert response_2.status_code == 200


def test_create_model_endpoint_invalid_team_returns_400(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    create_model_endpoint_request_sync: Dict[str, Any],
    create_model_endpoint_request_async: Dict[str, Any],
    test_api_key: str,
    get_test_client_wrapper,
):
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={
            model_bundle_1_v1[0].id: model_bundle_1_v1[0],
        },
        fake_model_endpoint_record_repository_contents={},
        fake_model_endpoint_infra_gateway_contents={},
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    invalid_team_name = "INVALID_TEAM"
    create_model_endpoint_request_sync["labels"]["team"] = invalid_team_name
    response_1 = client.post(
        "/v1/model-endpoints",
        auth=(test_api_key, ""),
        json=create_model_endpoint_request_sync,
    )
    assert response_1.status_code == 400

    create_model_endpoint_request_async["labels"]["team"] = invalid_team_name
    response_2 = client.post(
        "/v1/model-endpoints",
        auth=(test_api_key, ""),
        json=create_model_endpoint_request_async,
    )
    assert response_2.status_code == 400


def test_create_model_endpoint_invalid_streaming_bundle_returns_400(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    create_model_endpoint_request_streaming_invalid_bundle: Dict[str, Any],
    test_api_key: str,
    get_test_client_wrapper,
):
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={
            model_bundle_1_v1[0].id: model_bundle_1_v1[0],
        },
        fake_model_endpoint_record_repository_contents={},
        fake_model_endpoint_infra_gateway_contents={},
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response_1 = client.post(
        "/v1/model-endpoints",
        auth=(test_api_key, ""),
        json=create_model_endpoint_request_streaming_invalid_bundle,
    )
    assert response_1.status_code == 400


def test_create_model_endpoint_sync_invalid_streaming_bundle_returns_400(
    model_bundle_3_v1: Tuple[ModelBundle, Any],
    create_model_endpoint_request_sync_invalid_streaming_bundle: Dict[str, Any],
    test_api_key: str,
    get_test_client_wrapper,
):
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={
            model_bundle_3_v1[0].id: model_bundle_3_v1[0],
        },
        fake_model_endpoint_record_repository_contents={},
        fake_model_endpoint_infra_gateway_contents={},
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response_1 = client.post(
        "/v1/model-endpoints",
        auth=(test_api_key, ""),
        json=create_model_endpoint_request_sync_invalid_streaming_bundle,
    )
    assert response_1.status_code == 400


def test_create_model_endpoint_streaming_bundle_success(
    model_bundle_3_v1: Tuple[ModelBundle, Any],
    create_model_endpoint_request_streaming: Dict[str, Any],
    test_api_key: str,
    get_test_client_wrapper,
):
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={
            model_bundle_3_v1[0].id: model_bundle_3_v1[0],
        },
        fake_model_endpoint_record_repository_contents={},
        fake_model_endpoint_infra_gateway_contents={},
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response_1 = client.post(
        "/v1/model-endpoints",
        auth=(test_api_key, ""),
        json=create_model_endpoint_request_streaming,
    )
    assert response_1.status_code == 200


def test_create_model_endpoint_bundle_not_found_returns_404(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    create_model_endpoint_request_sync: Dict[str, Any],
    create_model_endpoint_request_async: Dict[str, Any],
    test_api_key: str,
    simple_client: TestClient,
):
    response_1 = simple_client.post(
        "/v1/model-endpoints",
        auth=(test_api_key, ""),
        json=create_model_endpoint_request_sync,
    )
    assert response_1.status_code == 404


def test_create_model_endpoint_bundle_not_authorized_returns_404(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    create_model_endpoint_request_sync: Dict[str, Any],
    create_model_endpoint_request_async: Dict[str, Any],
    test_api_key_2: str,
    get_test_client_wrapper,
):
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={
            model_bundle_1_v1[0].id: model_bundle_1_v1[0],
        },
        fake_model_endpoint_record_repository_contents={},
        fake_model_endpoint_infra_gateway_contents={},
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response_1 = client.post(
        "/v1/model-endpoints",
        auth=(test_api_key_2, ""),
        json=create_model_endpoint_request_sync,
    )
    assert response_1.status_code == 404


def test_create_model_endpoint_endpoint_already_exists_returns_400(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    create_model_endpoint_request_async: Dict[str, Any],
    test_api_key: str,
    get_test_client_wrapper,
):
    assert model_endpoint_1[0].infra_state is not None
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={
            model_bundle_1_v1[0].id: model_bundle_1_v1[0],
        },
        fake_model_endpoint_record_repository_contents={
            model_endpoint_1[0].record.id: model_endpoint_1[0].record,
        },
        fake_model_endpoint_infra_gateway_contents={
            model_endpoint_1[0].infra_state.deployment_name: model_endpoint_1[0].infra_state,
        },
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    create_model_endpoint_request_async["name"] = model_endpoint_1[0].record.name
    response_1 = client.post(
        "/v1/model-endpoints",
        auth=(test_api_key, ""),
        json=create_model_endpoint_request_async,
    )
    assert response_1.status_code == 400


def test_list_model_endpoints(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    model_endpoint_2: Tuple[ModelEndpoint, Any],
    test_api_key: str,
    test_api_key_2: str,
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
    response_1 = client.get(
        "/v1/model-endpoints?order_by=newest",
        auth=(test_api_key, ""),
    )
    expected_model_endpoint_1 = json.loads(
        GetModelEndpointV1Response.parse_obj(model_endpoint_1[1]).json()
    )
    expected_model_endpoint_2 = json.loads(
        GetModelEndpointV1Response.parse_obj(model_endpoint_2[1]).json()
    )
    assert response_1.status_code == 200
    assert response_1.json() == {
        "model_endpoints": [expected_model_endpoint_1, expected_model_endpoint_2]
    }

    response_2 = client.get(
        "/v1/model-endpoints?order_by=oldest",
        auth=(test_api_key_2, ""),
    )
    assert response_2.status_code == 200
    assert response_2.json() == {"model_endpoints": []}


def test_get_model_endpoint_by_id_success(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    test_api_key: str,
    get_test_client_wrapper,
):
    assert model_endpoint_1[0].infra_state is not None
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={
            model_bundle_1_v1[0].id: model_bundle_1_v1[0],
        },
        fake_model_endpoint_record_repository_contents={
            model_endpoint_1[0].record.id: model_endpoint_1[0].record,
        },
        fake_model_endpoint_infra_gateway_contents={
            model_endpoint_1[0].infra_state.deployment_name: model_endpoint_1[0].infra_state,
        },
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response = client.get(
        "/v1/model-endpoints/test_model_endpoint_id_1",
        auth=(test_api_key, ""),
    )
    assert response.status_code == 200
    assert response.json() == model_endpoint_1[1]


def test_get_model_endpoint_by_id_not_found_returns_404(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    test_api_key: str,
    get_test_client_wrapper,
):
    assert model_endpoint_1[0].infra_state is not None
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={
            model_bundle_1_v1[0].id: model_bundle_1_v1[0],
        },
        fake_model_endpoint_record_repository_contents={
            model_endpoint_1[0].record.id: model_endpoint_1[0].record,
        },
        fake_model_endpoint_infra_gateway_contents={
            model_endpoint_1[0].infra_state.deployment_name: model_endpoint_1[0].infra_state,
        },
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response = client.get(
        "/v1/model-endpoints/invalid_model_endpoint_id",
        auth=(test_api_key, ""),
    )
    assert response.status_code == 404


def test_get_model_endpoint_by_id_unauthorized_returns_404(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    test_api_key_2: str,
    get_test_client_wrapper,
):
    assert model_endpoint_1[0].infra_state is not None
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={
            model_bundle_1_v1[0].id: model_bundle_1_v1[0],
        },
        fake_model_endpoint_record_repository_contents={
            model_endpoint_1[0].record.id: model_endpoint_1[0].record,
        },
        fake_model_endpoint_infra_gateway_contents={
            model_endpoint_1[0].infra_state.deployment_name: model_endpoint_1[0].infra_state,
        },
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response = client.get(
        "/v1/model-endpoints/test_model_endpoint_id_1",
        auth=(test_api_key_2, ""),
    )
    assert response.status_code == 404


def test_update_model_endpoint_by_id_success(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    update_model_endpoint_request: Dict[str, Any],
    test_api_key: str,
    get_test_client_wrapper,
):
    assert model_endpoint_1[0].infra_state is not None
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={
            model_bundle_1_v1[0].id: model_bundle_1_v1[0],
        },
        fake_model_endpoint_record_repository_contents={
            model_endpoint_1[0].record.id: model_endpoint_1[0].record,
        },
        fake_model_endpoint_infra_gateway_contents={
            model_endpoint_1[0].infra_state.deployment_name: model_endpoint_1[0].infra_state,
        },
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response = client.put(
        "/v1/model-endpoints/test_model_endpoint_id_1",
        auth=(test_api_key, ""),
        json=update_model_endpoint_request,
    )
    assert response.status_code == 200
    assert response.json()["endpoint_creation_task_id"]


@pytest.mark.skip(reason="TODO: team validation is currently disabled")
def test_update_model_endpoint_by_id_invalid_team_returns_400(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    update_model_endpoint_request: Dict[str, Any],
    test_api_key: str,
    get_test_client_wrapper,
):
    assert model_endpoint_1[0].infra_state is not None
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={
            model_bundle_1_v1[0].id: model_bundle_1_v1[0],
        },
        fake_model_endpoint_record_repository_contents={
            model_endpoint_1[0].record.id: model_endpoint_1[0].record,
        },
        fake_model_endpoint_infra_gateway_contents={
            model_endpoint_1[0].infra_state.deployment_name: model_endpoint_1[0].infra_state,
        },
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    update_model_endpoint_request["labels"] = {
        "team": "some_invalid_team",
        "product": "my_product",
    }
    response = client.put(
        "/v1/model-endpoints/test_model_endpoint_id_1",
        auth=(test_api_key, ""),
        json=update_model_endpoint_request,
    )
    assert response.status_code == 400


def test_update_model_endpoint_by_id_endpoint_not_authorized_returns_404(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    update_model_endpoint_request: Dict[str, Any],
    test_api_key_2: str,
    get_test_client_wrapper,
):
    assert model_endpoint_1[0].infra_state is not None
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={
            model_bundle_1_v1[0].id: model_bundle_1_v1[0],
        },
        fake_model_endpoint_record_repository_contents={
            model_endpoint_1[0].record.id: model_endpoint_1[0].record,
        },
        fake_model_endpoint_infra_gateway_contents={
            model_endpoint_1[0].infra_state.deployment_name: model_endpoint_1[0].infra_state,
        },
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response = client.put(
        "/v1/model-endpoints/test_model_endpoint_id_1",
        auth=(test_api_key_2, ""),
        json=update_model_endpoint_request,
    )
    assert response.status_code == 404


def test_update_model_endpoint_by_id_bundle_not_authorized_returns_404(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_bundle_2_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    update_model_endpoint_request: Dict[str, Any],
    test_api_key: str,
    test_api_key_2: str,
    get_test_client_wrapper,
):
    model_bundle_2_v1[0].created_by = test_api_key_2
    model_bundle_2_v1[0].owner = test_api_key_2
    assert model_endpoint_1[0].infra_state is not None
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={
            model_bundle_1_v1[0].id: model_bundle_1_v1[0],
            model_bundle_2_v1[0].id: model_bundle_2_v1[0],
        },
        fake_model_endpoint_record_repository_contents={
            model_endpoint_1[0].record.id: model_endpoint_1[0].record,
        },
        fake_model_endpoint_infra_gateway_contents={
            model_endpoint_1[0].infra_state.deployment_name: model_endpoint_1[0].infra_state,
        },
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    update_model_endpoint_request["model_bundle_id"] = model_bundle_2_v1[0].id
    response = client.put(
        "/v1/model-endpoints/test_model_endpoint_id_1",
        auth=(test_api_key, ""),
        json=update_model_endpoint_request,
    )
    assert response.status_code == 404


def test_update_model_endpoint_by_id_endpoint_not_found_raises_404(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    update_model_endpoint_request: Dict[str, Any],
    test_api_key: str,
    get_test_client_wrapper,
):
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={
            model_bundle_1_v1[0].id: model_bundle_1_v1[0],
        },
        fake_model_endpoint_record_repository_contents={},
        fake_model_endpoint_infra_gateway_contents={},
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response = client.put(
        "/v1/model-endpoints/test_model_endpoint_id_1",
        auth=(test_api_key, ""),
        json=update_model_endpoint_request,
    )
    assert response.status_code == 404


def test_update_model_endpoint_by_id_bundle_not_found_returns_404(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    update_model_endpoint_request: Dict[str, Any],
    test_api_key: str,
    get_test_client_wrapper,
):
    assert model_endpoint_1[0].infra_state is not None
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={
            model_bundle_1_v1[0].id: model_bundle_1_v1[0],
        },
        fake_model_endpoint_record_repository_contents={
            model_endpoint_1[0].record.id: model_endpoint_1[0].record,
        },
        fake_model_endpoint_infra_gateway_contents={
            model_endpoint_1[0].infra_state.deployment_name: model_endpoint_1[0].infra_state,
        },
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    update_model_endpoint_request["model_bundle_id"] = "invalid_model_bundle_id"
    response = client.put(
        "/v1/model-endpoints/test_model_endpoint_id_1",
        auth=(test_api_key, ""),
        json=update_model_endpoint_request,
    )
    assert response.status_code == 404


@pytest.mark.skip(reason="Temporarily disabled returning 409s")
def test_update_model_endpoint_by_id_operation_in_progress_returns_409(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    update_model_endpoint_request: Dict[str, Any],
    test_api_key: str,
    get_test_client_wrapper,
):
    model_endpoint_1[0].record.status = ModelEndpointStatus.UPDATE_IN_PROGRESS
    assert model_endpoint_1[0].infra_state is not None
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={
            model_bundle_1_v1[0].id: model_bundle_1_v1[0],
        },
        fake_model_endpoint_record_repository_contents={
            model_endpoint_1[0].record.id: model_endpoint_1[0].record,
        },
        fake_model_endpoint_infra_gateway_contents={
            model_endpoint_1[0].infra_state.deployment_name: model_endpoint_1[0].infra_state,
        },
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response = client.put(
        "/v1/model-endpoints/test_model_endpoint_id_1",
        auth=(test_api_key, ""),
        json=update_model_endpoint_request,
    )
    assert response.status_code == 409


def test_delete_model_endpoint_by_id_success(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    test_api_key: str,
    get_test_client_wrapper,
):
    assert model_endpoint_1[0].infra_state is not None
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={
            model_bundle_1_v1[0].id: model_bundle_1_v1[0],
        },
        fake_model_endpoint_record_repository_contents={
            model_endpoint_1[0].record.id: model_endpoint_1[0].record,
        },
        fake_model_endpoint_infra_gateway_contents={
            model_endpoint_1[0].infra_state.deployment_name: model_endpoint_1[0].infra_state,
        },
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response = client.delete(
        "/v1/model-endpoints/test_model_endpoint_id_1",
        auth=(test_api_key, ""),
    )
    assert response.status_code == 200
    assert response.json() == {"deleted": True}


def test_delete_model_endpoint_by_id_not_found_returns_404(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    test_api_key: str,
    get_test_client_wrapper,
):
    assert model_endpoint_1[0].infra_state is not None
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={
            model_bundle_1_v1[0].id: model_bundle_1_v1[0],
        },
        fake_model_endpoint_record_repository_contents={
            model_endpoint_1[0].record.id: model_endpoint_1[0].record,
        },
        fake_model_endpoint_infra_gateway_contents={
            model_endpoint_1[0].infra_state.deployment_name: model_endpoint_1[0].infra_state,
        },
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response = client.delete(
        "/v1/model-endpoints/invalid_model_endpoint_id",
        auth=(test_api_key, ""),
    )
    assert response.status_code == 404


def test_delete_model_endpoint_by_id_unauthorized_returns_404(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    test_api_key_2: str,
    get_test_client_wrapper,
):
    assert model_endpoint_1[0].infra_state is not None
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={
            model_bundle_1_v1[0].id: model_bundle_1_v1[0],
        },
        fake_model_endpoint_record_repository_contents={
            model_endpoint_1[0].record.id: model_endpoint_1[0].record,
        },
        fake_model_endpoint_infra_gateway_contents={
            model_endpoint_1[0].infra_state.deployment_name: model_endpoint_1[0].infra_state,
        },
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response = client.delete(
        "/v1/model-endpoints/test_model_endpoint_id_1",
        auth=(test_api_key_2, ""),
    )
    assert response.status_code == 404
