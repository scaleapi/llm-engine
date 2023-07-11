from typing import Any, Dict, Tuple

from fastapi.testclient import TestClient

from spellbook_serve.domain.entities.docker_image_batch_job_bundle_entity import (
    DockerImageBatchJobBundle,
)


def test_create_docker_image_batch_bundle_success(
    create_docker_image_batch_job_bundle_request: Dict[str, Any],
    test_api_key: str,
    simple_client: TestClient,
):
    response_1 = simple_client.post(
        "/v1/docker-image-batch-job-bundles",
        auth=(test_api_key, ""),
        json=create_docker_image_batch_job_bundle_request,
    )
    assert response_1.status_code == 200
    assert "docker_image_batch_job_bundle_id" in response_1.json()


def test_create_docker_image_batch_bundle_bad_gpu_request(
    create_docker_image_batch_job_bundle_request: Dict[str, Any],
    test_api_key: str,
    simple_client: TestClient,
):
    create_docker_image_batch_job_bundle_request["resource_requests"][
        "gpu_type"
    ] = "nvidia-hopper-h9001"
    response_1 = simple_client.post(
        "/v1/docker-image-batch-job-bundles",
        auth=(test_api_key, ""),
        json=create_docker_image_batch_job_bundle_request,
    )
    assert response_1.status_code == 422


def test_list_docker_image_batch_job_model_bundles_success(
    test_api_key: str,
    get_test_client_wrapper,
    docker_image_batch_job_bundle_1_v1: Tuple[DockerImageBatchJobBundle, Any],
    docker_image_batch_job_bundle_1_v2: Tuple[DockerImageBatchJobBundle, Any],
    docker_image_batch_job_bundle_2_v1: Tuple[DockerImageBatchJobBundle, Any],
):
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={},
        fake_model_endpoint_record_repository_contents={},
        fake_model_endpoint_infra_gateway_contents={},
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={
            docker_image_batch_job_bundle_1_v1[0].id: docker_image_batch_job_bundle_1_v1[0],
            docker_image_batch_job_bundle_1_v2[0].id: docker_image_batch_job_bundle_1_v2[0],
            docker_image_batch_job_bundle_2_v1[0].id: docker_image_batch_job_bundle_2_v1[0],
        },
    )
    bun_name = docker_image_batch_job_bundle_1_v1[0].name
    response_1 = client.get(
        f"/v1/docker-image-batch-job-bundles?bundle_name={bun_name}&order_by=newest",
        auth=(test_api_key, ""),
    )
    docker_image_batch_job_1_v1_json = docker_image_batch_job_bundle_1_v1[1]
    docker_image_batch_job_1_v2_json = docker_image_batch_job_bundle_1_v2[1]
    assert response_1.status_code == 200
    assert response_1.json() == {
        "docker_image_batch_job_bundles": [
            docker_image_batch_job_1_v2_json,
            docker_image_batch_job_1_v1_json,
        ]
    }

    response_2 = client.get(
        f"/v1/docker-image-batch-job-bundles?bundle_name={bun_name}&order_by=oldest",
        auth=(test_api_key, ""),
    )
    assert response_2.status_code == 200
    assert response_2.json() == {
        "docker_image_batch_job_bundles": [
            docker_image_batch_job_1_v1_json,
            docker_image_batch_job_1_v2_json,
        ]
    }


def test_get_latest_docker_image_batch_job_bundle_success(
    test_api_key: str,
    get_test_client_wrapper,
    docker_image_batch_job_bundle_1_v1: Tuple[DockerImageBatchJobBundle, Any],
    docker_image_batch_job_bundle_1_v2: Tuple[DockerImageBatchJobBundle, Any],
    docker_image_batch_job_bundle_2_v1: Tuple[DockerImageBatchJobBundle, Any],
):
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={},
        fake_model_endpoint_record_repository_contents={},
        fake_model_endpoint_infra_gateway_contents={},
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={
            docker_image_batch_job_bundle_1_v1[0].id: docker_image_batch_job_bundle_1_v1[0],
            docker_image_batch_job_bundle_1_v2[0].id: docker_image_batch_job_bundle_1_v2[0],
            docker_image_batch_job_bundle_2_v1[0].id: docker_image_batch_job_bundle_2_v1[0],
        },
    )
    bundle_name = docker_image_batch_job_bundle_1_v1[0].name
    response_1 = client.get(
        f"/v1/docker-image-batch-job-bundles/latest?bundle_name={bundle_name}",
        auth=(test_api_key, ""),
    )
    assert response_1.status_code == 200
    assert response_1.json() == docker_image_batch_job_bundle_1_v2[1]


def test_get_latest_docker_image_batch_job_bundle_not_found_returns_404(
    test_api_key: str,
    get_test_client_wrapper,
    docker_image_batch_job_bundle_1_v1: Tuple[DockerImageBatchJobBundle, Any],
    docker_image_batch_job_bundle_1_v2: Tuple[DockerImageBatchJobBundle, Any],
    docker_image_batch_job_bundle_2_v1: Tuple[DockerImageBatchJobBundle, Any],
):
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={},
        fake_model_endpoint_record_repository_contents={},
        fake_model_endpoint_infra_gateway_contents={},
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={
            docker_image_batch_job_bundle_1_v1[0].id: docker_image_batch_job_bundle_1_v1[0],
            docker_image_batch_job_bundle_1_v2[0].id: docker_image_batch_job_bundle_1_v2[0],
            docker_image_batch_job_bundle_2_v1[0].id: docker_image_batch_job_bundle_2_v1[0],
        },
    )
    response_1 = client.get(
        "/v1/docker-image-batch-job-bundles/latest?bundle_name=invalid_name",
        auth=(test_api_key, ""),
    )
    assert response_1.status_code == 404


def test_get_docker_image_batch_job_model_bundle_success(
    test_api_key: str,
    get_test_client_wrapper,
    docker_image_batch_job_bundle_1_v1: Tuple[DockerImageBatchJobBundle, Any],
    docker_image_batch_job_bundle_1_v2: Tuple[DockerImageBatchJobBundle, Any],
    docker_image_batch_job_bundle_2_v1: Tuple[DockerImageBatchJobBundle, Any],
):
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={},
        fake_model_endpoint_record_repository_contents={},
        fake_model_endpoint_infra_gateway_contents={},
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={
            docker_image_batch_job_bundle_1_v1[0].id: docker_image_batch_job_bundle_1_v1[0],
            docker_image_batch_job_bundle_1_v2[0].id: docker_image_batch_job_bundle_1_v2[0],
            docker_image_batch_job_bundle_2_v1[0].id: docker_image_batch_job_bundle_2_v1[0],
        },
    )
    bun_id = docker_image_batch_job_bundle_1_v1[0].id
    response_1 = client.get(
        f"/v1/docker-image-batch-job-bundles/{bun_id}",
        auth=(test_api_key, ""),
    )
    assert response_1.status_code == 200
    assert response_1.json() == docker_image_batch_job_bundle_1_v1[1]


def test_get_docker_image_batch_job_model_bundle_not_found_returns_404(
    test_api_key: str,
    get_test_client_wrapper,
    docker_image_batch_job_bundle_1_v1: Tuple[DockerImageBatchJobBundle, Any],
    docker_image_batch_job_bundle_1_v2: Tuple[DockerImageBatchJobBundle, Any],
    docker_image_batch_job_bundle_2_v1: Tuple[DockerImageBatchJobBundle, Any],
):
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={},
        fake_model_endpoint_record_repository_contents={},
        fake_model_endpoint_infra_gateway_contents={},
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={
            docker_image_batch_job_bundle_1_v1[0].id: docker_image_batch_job_bundle_1_v1[0],
            docker_image_batch_job_bundle_1_v2[0].id: docker_image_batch_job_bundle_1_v2[0],
            docker_image_batch_job_bundle_2_v1[0].id: docker_image_batch_job_bundle_2_v1[0],
        },
    )
    response_1 = client.get(
        "/v1/docker-image-batch-job-bundles/invalid_id",
        auth=(test_api_key, ""),
    )
    assert response_1.status_code == 404


def test_get_docker_image_batch_job_model_bundle_unauthorized_returns_404(
    test_api_key_2: str,
    get_test_client_wrapper,
    docker_image_batch_job_bundle_1_v1: Tuple[DockerImageBatchJobBundle, Any],
    docker_image_batch_job_bundle_1_v2: Tuple[DockerImageBatchJobBundle, Any],
    docker_image_batch_job_bundle_2_v1: Tuple[DockerImageBatchJobBundle, Any],
):
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={},
        fake_model_endpoint_record_repository_contents={},
        fake_model_endpoint_infra_gateway_contents={},
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={
            docker_image_batch_job_bundle_1_v1[0].id: docker_image_batch_job_bundle_1_v1[0],
            docker_image_batch_job_bundle_1_v2[0].id: docker_image_batch_job_bundle_1_v2[0],
            docker_image_batch_job_bundle_2_v1[0].id: docker_image_batch_job_bundle_2_v1[0],
        },
    )
    bun_id = docker_image_batch_job_bundle_1_v1[0].id
    response_1 = client.get(
        f"/v1/docker-image-batch-job-bundles/{bun_id}",
        auth=(test_api_key_2, ""),
    )
    assert response_1.status_code == 404
