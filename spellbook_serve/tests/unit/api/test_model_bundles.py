from typing import Any, Dict, Tuple

import pytest
from fastapi.testclient import TestClient

from spellbook_serve.domain.entities import ModelBundle


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_create_model_bundle_success(
    create_model_bundle_request_pytorch: Dict[str, Any],
    create_model_bundle_request_custom: Dict[str, Any],
    test_api_key: str,
    simple_client: TestClient,
    version: str,
):
    response_1 = simple_client.post(
        f"/{version}/model-bundles",
        auth=(test_api_key, ""),
        json=create_model_bundle_request_pytorch[version],
    )
    assert response_1.status_code == 200
    assert "model_bundle_id" in response_1.json()

    response_2 = simple_client.post(
        f"/{version}/model-bundles",
        auth=(test_api_key, ""),
        json=create_model_bundle_request_pytorch[version],
    )
    assert response_2.status_code == 200
    assert "model_bundle_id" in response_2.json()

    response_3 = simple_client.post(
        f"/{version}/model-bundles",
        auth=(test_api_key, ""),
        json=create_model_bundle_request_custom[version],
    )
    assert response_3.status_code == 200
    assert "model_bundle_id" in response_3.json()


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_create_model_bundle_docker_not_found_raises_400(
    create_model_bundle_request_custom: Dict[str, Any],
    test_api_key: str,
    get_test_client_wrapper,
    version: str,
):
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=False,
        fake_model_bundle_repository_contents={},
        fake_model_endpoint_record_repository_contents={},
        fake_model_endpoint_infra_gateway_contents={},
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response = client.post(
        f"/{version}/model-bundles",
        auth=(test_api_key, ""),
        json=create_model_bundle_request_custom[version],
    )
    assert response.status_code == 400


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_clone_model_bundle_success(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    create_model_bundle_request_pytorch: Dict[str, Any],
    test_api_key: str,
    get_test_client_wrapper,
    version: str,
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
    response = client.post(
        f"/{version}/model-bundles/clone-with-changes",
        auth=(test_api_key, ""),
        json={"original_model_bundle_id": model_bundle_1_v1[0].id, "app_config": {"foo": "bar"}},
    )
    assert response.status_code == 200
    response_json = response.json()
    assert response_json["model_bundle_id"] != model_bundle_1_v1[0].id


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_clone_model_bundle_unauthorized_returns_404(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    create_model_bundle_request_pytorch: Dict[str, Any],
    test_api_key_2: str,
    get_test_client_wrapper,
    version: str,
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
    response = client.post(
        f"/{version}/model-bundles/clone-with-changes",
        auth=(test_api_key_2, ""),  # Not the owner, should be unauthorized
        json={"original_model_bundle_id": model_bundle_1_v1[0].id, "app_config": {"foo": "bar"}},
    )
    assert response.status_code == 404


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_clone_model_bundle_not_found_returns_404(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    create_model_bundle_request_pytorch: Dict[str, Any],
    test_api_key: str,
    get_test_client_wrapper,
    version: str,
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
    response = client.post(
        f"/{version}/model-bundles/clone-with-changes",
        auth=(test_api_key, ""),
        json={"original_model_bundle_id": "unknown model bundle id", "app_config": {"foo": "bar"}},
    )
    assert response.status_code == 404


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_list_model_bundles(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_bundle_1_v2: Tuple[ModelBundle, Any],
    model_bundle_2_v1: Tuple[ModelBundle, Any],
    test_api_key: str,
    get_test_client_wrapper,
    version: str,
):
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={
            model_bundle_1_v1[0].id: model_bundle_1_v1[0],
            model_bundle_1_v2[0].id: model_bundle_1_v2[0],
            model_bundle_2_v1[0].id: model_bundle_2_v1[0],
        },
        fake_model_endpoint_record_repository_contents={},
        fake_model_endpoint_infra_gateway_contents={},
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response_1 = client.get(
        f"/{version}/model-bundles?model_name=test_model_bundle_name_1&order_by=newest",
        auth=(test_api_key, ""),
    )
    model_bundle_1_v1_json = model_bundle_1_v1[1][version]
    model_bundle_1_v2_json = model_bundle_1_v2[1][version]
    assert response_1.status_code == 200
    assert response_1.json() == {"model_bundles": [model_bundle_1_v2_json, model_bundle_1_v1_json]}

    response_2 = client.get(
        f"/{version}/model-bundles?model_name=test_model_bundle_name_1&order_by=oldest",
        auth=(test_api_key, ""),
    )
    assert response_2.status_code == 200
    assert response_2.json() == {"model_bundles": [model_bundle_1_v1_json, model_bundle_1_v2_json]}


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_get_model_bundle_by_id_success(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    test_api_key: str,
    get_test_client_wrapper,
    version: str,
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
    response = client.get(
        f"/{version}/model-bundles/test_model_bundle_id_1",
        auth=(test_api_key, ""),
    )
    assert response.status_code == 200
    assert response.json() == model_bundle_1_v1[1][version]


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_get_model_bundle_by_id_not_found_returns_404(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    test_api_key: str,
    get_test_client_wrapper,
    version: str,
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
    response = client.get(
        f"/{version}/model-bundles/invalid_model_bundle_id",
        auth=(test_api_key, ""),
    )
    assert response.status_code == 404


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_get_model_bundle_by_id_unauthorized_returns_404(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    test_api_key_2: str,
    get_test_client_wrapper,
    version: str,
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
    response = client.get(
        f"/{version}/model-bundles/test_model_bundle_id_1",
        auth=(test_api_key_2, ""),
    )
    assert response.status_code == 404


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_get_latest_model_bundle_success(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    test_api_key: str,
    get_test_client_wrapper,
    version: str,
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
    response = client.get(
        f"/{version}/model-bundles/latest?model_name=test_model_bundle_name_1",
        auth=(test_api_key, ""),
    )
    assert response.status_code == 200
    assert response.json() == model_bundle_1_v1[1][version]


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_get_latest_model_bundle_not_found_returns_404(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    test_api_key: str,
    get_test_client_wrapper,
    version: str,
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
    response = client.get(
        f"/{version}/model-bundles/latest?model_name=invalid_model_bundle_name",
        auth=(test_api_key, ""),
    )
    assert response.status_code == 404
