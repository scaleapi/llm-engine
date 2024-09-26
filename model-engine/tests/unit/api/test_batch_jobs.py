from typing import Any, Dict, Tuple

import pytest
from fastapi.testclient import TestClient
from model_engine_server.domain.entities import (
    BatchJob,
    DockerImageBatchJob,
    GpuType,
    ModelBundle,
    ModelEndpoint,
    Trigger,
)
from model_engine_server.domain.entities.docker_image_batch_job_bundle_entity import (
    DockerImageBatchJobBundle,
)


def test_create_batch_job_success(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    create_batch_job_request: Dict[str, Any],
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
    response = client.post(
        "/v1/batch-jobs",
        auth=(test_api_key, ""),
        json=create_batch_job_request,
    )
    assert response.status_code == 200
    assert "job_id" in response.json()


@pytest.mark.skip(reason="TODO: team validation is currently disabled")
def test_create_batch_job_invalid_team_returns_400(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    create_batch_job_request: Dict[str, Any],
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
    create_batch_job_request["labels"]["team"] = "invalid_team"
    response = client.post(
        "/v1/batch-jobs",
        auth=(test_api_key, ""),
        json=create_batch_job_request,
    )
    assert response.status_code == 400


def test_create_batch_job_bundle_not_found_returns_404(
    create_batch_job_request: Dict[str, Any],
    test_api_key: str,
    simple_client: TestClient,
):
    response = simple_client.post(
        "/v1/batch-jobs",
        auth=(test_api_key, ""),
        json=create_batch_job_request,
    )
    assert response.status_code == 404


def test_get_batch_job_success(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    batch_job_1: Tuple[BatchJob, Any],
    test_api_key: str,
    get_test_client_wrapper,
):
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={},
        fake_model_endpoint_record_repository_contents={},
        fake_model_endpoint_infra_gateway_contents={},
        fake_batch_job_record_repository_contents={
            batch_job_1[0].record.id: batch_job_1[0].record,
        },
        fake_batch_job_progress_gateway_contents=batch_job_1[0].progress.json(),
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response = client.get(
        f"/v1/batch-jobs/{batch_job_1[0].record.id}",
        auth=(test_api_key, ""),
    )
    assert response.status_code == 200
    batch_job_1[1]["duration"] = response.json()["duration"]
    assert response.json() == batch_job_1[1]


def test_get_batch_job_unauthorized_returns_404(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    batch_job_1: Tuple[BatchJob, Any],
    get_test_client_wrapper,
):
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={},
        fake_model_endpoint_record_repository_contents={},
        fake_model_endpoint_infra_gateway_contents={},
        fake_batch_job_record_repository_contents={
            batch_job_1[0].record.id: batch_job_1[0].record,
        },
        fake_batch_job_progress_gateway_contents=batch_job_1[0].progress.json(),
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response = client.get(
        f"/v1/batch-jobs/{batch_job_1[0].record.id}",
        auth=("invalid_api_key", ""),
    )
    assert response.status_code == 404


def test_get_batch_job_not_found_returns_404(
    test_api_key: str,
    simple_client: TestClient,
):
    response = simple_client.get(
        "/v1/batch-jobs/some_batch_job_id",
        auth=(test_api_key, ""),
    )
    assert response.status_code == 404


def test_update_batch_job_success(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    batch_job_1: Tuple[BatchJob, Any],
    test_api_key: str,
    get_test_client_wrapper,
):
    assert model_endpoint_1[0].infra_state is not None
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={},
        fake_model_endpoint_record_repository_contents={
            model_endpoint_1[0].record.id: model_endpoint_1[0].record,
        },
        fake_model_endpoint_infra_gateway_contents={
            model_endpoint_1[0].infra_state.deployment_name: model_endpoint_1[0].infra_state,
        },
        fake_batch_job_record_repository_contents={
            batch_job_1[0].record.id: batch_job_1[0].record,
        },
        fake_batch_job_progress_gateway_contents=batch_job_1[0].progress.json(),
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response = client.put(
        f"/v1/batch-jobs/{batch_job_1[0].record.id}",
        auth=(test_api_key, ""),
        json={"cancel": True},
    )
    assert response.status_code == 200
    assert response.json() == {"success": True}

    # Canceling is idempotent.
    response = client.put(
        f"/v1/batch-jobs/{batch_job_1[0].record.id}",
        auth=(test_api_key, ""),
        json={"cancel": True},
    )
    assert response.status_code == 200
    assert response.json() == {"success": True}


def test_update_batch_job_unauthorized_returns_404(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    batch_job_1: Tuple[BatchJob, Any],
    get_test_client_wrapper,
):
    assert model_endpoint_1[0].infra_state is not None
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={},
        fake_model_endpoint_record_repository_contents={
            model_endpoint_1[0].record.id: model_endpoint_1[0].record,
        },
        fake_model_endpoint_infra_gateway_contents={
            model_endpoint_1[0].infra_state.deployment_name: model_endpoint_1[0].infra_state,
        },
        fake_batch_job_record_repository_contents={
            batch_job_1[0].record.id: batch_job_1[0].record,
        },
        fake_batch_job_progress_gateway_contents=batch_job_1[0].progress.json(),
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response = client.put(
        f"/v1/batch-jobs/{batch_job_1[0].record.id}",
        auth=("invalid_api_key", ""),
        json={"cancel": True},
    )
    assert response.status_code == 404


def test_update_batch_job_not_found_returns_404(
    test_api_key: str,
    simple_client: TestClient,
):
    response = simple_client.put(
        "/v1/batch-jobs/some_batch_job_id",
        auth=(test_api_key, ""),
        json={"cancel": True},
    )
    assert response.status_code == 404


# these test both the api + use cases for docker image batch jobs


def test_create_docker_image_batch_job_success(
    test_api_key: str,
    get_test_client_wrapper,
    create_docker_image_batch_job_request: Dict[str, Any],
    docker_image_batch_job_bundle_1_v1: Tuple[DockerImageBatchJobBundle, Any],
):
    client = get_test_client_wrapper(
        fake_docker_image_batch_job_bundle_repository_contents={
            docker_image_batch_job_bundle_1_v1[0].id: docker_image_batch_job_bundle_1_v1[0]
        }
    )
    response = client.post(
        "/v1/docker-image-batch-jobs",
        auth=(test_api_key, ""),
        json=create_docker_image_batch_job_request,
    )
    assert response.status_code == 200


@pytest.mark.parametrize("gpu_type", [e for e in GpuType])
def test_create_docker_image_batch_job_success_existing_gpu_specification(
    test_api_key: str,
    get_test_client_wrapper,
    create_docker_image_batch_job_request: Dict[str, Any],
    docker_image_batch_job_bundle_1_v1: Tuple[DockerImageBatchJobBundle, Any],
    gpu_type: str,
):
    di_batch_bundle = docker_image_batch_job_bundle_1_v1[0]
    di_batch_bundle.gpu_type = gpu_type
    client = get_test_client_wrapper(
        fake_docker_image_batch_job_bundle_repository_contents={di_batch_bundle.id: di_batch_bundle}
    )
    response = client.post(
        "/v1/docker-image-batch-jobs",
        auth=(test_api_key, ""),
        json=create_docker_image_batch_job_request,
    )
    assert response.status_code == 200, f"Failed on {gpu_type=}"


def test_create_docker_image_batch_job_not_present(
    test_api_key: str,
    get_test_client_wrapper,
    create_docker_image_batch_job_request: Dict[str, Any],
):
    client = get_test_client_wrapper(fake_docker_image_batch_job_bundle_repository_contents={})
    response = client.post(
        "/v1/docker-image-batch-jobs",
        auth=(test_api_key, ""),
        json=create_docker_image_batch_job_request,
    )
    assert response.status_code == 404


def test_create_docker_image_batch_job_unauthorized(
    test_api_key_2: str,
    get_test_client_wrapper,
    create_docker_image_batch_job_request: Dict[str, Any],
    docker_image_batch_job_bundle_1_v1: Tuple[DockerImageBatchJobBundle, Any],
):
    client = get_test_client_wrapper(
        fake_docker_image_batch_job_bundle_repository_contents={
            docker_image_batch_job_bundle_1_v1[0].id: docker_image_batch_job_bundle_1_v1[0]
        }
    )
    del create_docker_image_batch_job_request["docker_image_batch_job_bundle_name"]
    create_docker_image_batch_job_request["docker_image_batch_job_bundle_id"] = (
        docker_image_batch_job_bundle_1_v1[0].id
    )
    response = client.post(
        "/v1/docker-image-batch-jobs",
        auth=(test_api_key_2, ""),
        json=create_docker_image_batch_job_request,
    )
    assert response.status_code == 404


def test_create_docker_image_batch_job_no_bundle_id_or_name(
    test_api_key: str,
    get_test_client_wrapper,
    create_docker_image_batch_job_request: Dict[str, Any],
    docker_image_batch_job_bundle_1_v1: Tuple[DockerImageBatchJobBundle, Any],
):
    client = get_test_client_wrapper(
        fake_docker_image_batch_job_bundle_repository_contents={
            docker_image_batch_job_bundle_1_v1[0].id: docker_image_batch_job_bundle_1_v1[0]
        }
    )
    del create_docker_image_batch_job_request["docker_image_batch_job_bundle_name"]
    response = client.post(
        "/v1/docker-image-batch-jobs",
        auth=(test_api_key, ""),
        json=create_docker_image_batch_job_request,
    )
    assert response.status_code == 422


def test_create_docker_image_batch_job_bundle_id_and_name(
    test_api_key: str,
    get_test_client_wrapper,
    create_docker_image_batch_job_request: Dict[str, Any],
    docker_image_batch_job_bundle_1_v1: Tuple[DockerImageBatchJobBundle, Any],
):
    client = get_test_client_wrapper(
        fake_docker_image_batch_job_bundle_repository_contents={
            docker_image_batch_job_bundle_1_v1[0].id: docker_image_batch_job_bundle_1_v1[0]
        }
    )
    create_docker_image_batch_job_request["docker_image_batch_job_bundle_id"] = (
        docker_image_batch_job_bundle_1_v1[0].id
    )
    response = client.post(
        "/v1/docker-image-batch-jobs",
        auth=(test_api_key, ""),
        json=create_docker_image_batch_job_request,
    )
    assert response.status_code == 422


def test_create_docker_image_batch_job_bad_resources(
    test_api_key: str,
    get_test_client_wrapper,
    create_docker_image_batch_job_request: Dict[str, Any],
    docker_image_batch_job_bundle_1_v1: Tuple[DockerImageBatchJobBundle, Any],
):
    client = get_test_client_wrapper(
        fake_docker_image_batch_job_bundle_repository_contents={
            docker_image_batch_job_bundle_1_v1[0].id: docker_image_batch_job_bundle_1_v1[0]
        }
    )
    create_docker_image_batch_job_request["resource_requests"]["cpus"] = -0.1
    response = client.post(
        "/v1/docker-image-batch-jobs",
        auth=(test_api_key, ""),
        json=create_docker_image_batch_job_request,
    )
    assert response.status_code == 400


def test_create_docker_image_batch_job_bad_resources_gpu_type(
    test_api_key: str,
    get_test_client_wrapper,
    create_docker_image_batch_job_request: Dict[str, Any],
    docker_image_batch_job_bundle_1_v1: Tuple[DockerImageBatchJobBundle, Any],
):
    client = get_test_client_wrapper(
        fake_docker_image_batch_job_bundle_repository_contents={
            docker_image_batch_job_bundle_1_v1[0].id: docker_image_batch_job_bundle_1_v1[0]
        }
    )
    create_docker_image_batch_job_request["resource_requests"]["gpu_type"] = "nvidia-hopper-h9001"
    response = client.post(
        "/v1/docker-image-batch-jobs",
        auth=(test_api_key, ""),
        json=create_docker_image_batch_job_request,
    )
    assert response.status_code == 422


def test_create_docker_image_batch_job_empty_resources(
    test_api_key: str,
    get_test_client_wrapper,
    create_docker_image_batch_job_request: Dict[str, Any],
    docker_image_batch_job_bundle_1_v1: Tuple[DockerImageBatchJobBundle, Any],
):
    client = get_test_client_wrapper(
        fake_docker_image_batch_job_bundle_repository_contents={
            docker_image_batch_job_bundle_1_v1[0].id: docker_image_batch_job_bundle_1_v1[0]
        }
    )
    create_docker_image_batch_job_request["resource_requests"] = {}
    response = client.post(
        "/v1/docker-image-batch-jobs",
        auth=(test_api_key, ""),
        json=create_docker_image_batch_job_request,
    )
    assert response.status_code == 400


def test_create_docker_image_batch_job_no_image(
    test_api_key: str,
    get_test_client_wrapper,
    create_docker_image_batch_job_request: Dict[str, Any],
    docker_image_batch_job_bundle_1_v1: Tuple[DockerImageBatchJobBundle, Any],
):
    client = get_test_client_wrapper(
        fake_docker_image_batch_job_bundle_repository_contents={
            docker_image_batch_job_bundle_1_v1[0].id: docker_image_batch_job_bundle_1_v1[0]
        },
        fake_docker_repository_image_always_exists=False,
    )
    response = client.post(
        "/v1/docker-image-batch-jobs",
        auth=(test_api_key, ""),
        json=create_docker_image_batch_job_request,
    )
    assert response.status_code == 404


def test_create_docker_image_batch_job_invalid_time_limit(
    test_api_key: str,
    get_test_client_wrapper,
    create_docker_image_batch_job_request: Dict[str, Any],
    docker_image_batch_job_bundle_1_v1: Tuple[DockerImageBatchJobBundle, Any],
):
    client = get_test_client_wrapper(
        fake_docker_image_batch_job_bundle_repository_contents={
            docker_image_batch_job_bundle_1_v1[0].id: docker_image_batch_job_bundle_1_v1[0]
        }
    )
    create_docker_image_batch_job_request["override_job_max_runtime_s"] = -1
    response = client.post(
        "/v1/docker-image-batch-jobs",
        auth=(test_api_key, ""),
        json=create_docker_image_batch_job_request,
    )
    assert response.status_code == 400


def test_get_docker_image_batch_job_success(
    test_api_key: str,
    get_test_client_wrapper,
    docker_image_batch_job_1: Tuple[DockerImageBatchJob, Any],
):
    batch_job_id = docker_image_batch_job_1[0].id
    client = get_test_client_wrapper(
        fake_docker_image_batch_job_gateway_contents={batch_job_id: docker_image_batch_job_1[0]}
    )
    response = client.get(
        f"/v1/docker-image-batch-jobs/{batch_job_id}",
        auth=(test_api_key, ""),
    )
    assert response.status_code == 200
    assert response.json() == docker_image_batch_job_1[1]


def test_get_docker_image_batch_job_unauthorized(
    test_api_key_2: str,
    get_test_client_wrapper,
    docker_image_batch_job_1: Tuple[DockerImageBatchJob, Any],
):
    batch_job_id = docker_image_batch_job_1[0].id
    client = get_test_client_wrapper(
        fake_docker_image_batch_job_gateway_contents={batch_job_id: docker_image_batch_job_1[0]}
    )
    response = client.get(
        f"/v1/docker-image-batch-jobs/{batch_job_id}",
        auth=(test_api_key_2, ""),
    )
    assert response.status_code == 404


def test_get_docker_image_batch_job_not_exist(
    test_api_key: str,
    get_test_client_wrapper,
    docker_image_batch_job_1: Tuple[DockerImageBatchJob, Any],
):
    batch_job_id = docker_image_batch_job_1[0].id
    client = get_test_client_wrapper(fake_docker_image_batch_job_gateway_contents={})
    response = client.get(
        f"/v1/docker-image-batch-jobs/{batch_job_id}",
        auth=(test_api_key, ""),
    )
    assert response.status_code == 404


def test_list_jobs_success(
    test_api_key: str,
    get_test_client_wrapper,
    trigger_1: Tuple[Trigger, Any],
    trigger_2: Tuple[Trigger, Any],
):
    client = get_test_client_wrapper(
        fake_trigger_repository_contents={
            trigger_1[0].id: trigger_1[0],
            trigger_2[0].id: trigger_2[0],
        },
    )
    response = client.get(
        "/v1/docker-image-batch-jobs",
        auth=(test_api_key, ""),
    )
    assert response.status_code == 200
    assert "jobs" in response.json()


def test_list_jobs_by_trigger_success(
    test_api_key: str,
    get_test_client_wrapper,
    trigger_1: Tuple[Trigger, Any],
    trigger_2: Tuple[Trigger, Any],
):
    client = get_test_client_wrapper(
        fake_trigger_repository_contents={
            trigger_1[0].id: trigger_1[0],
            trigger_2[0].id: trigger_2[0],
        },
    )
    response = client.get(
        f"/v1/docker-image-batch-jobs?trigger_id={trigger_1[0].id}",
        auth=(test_api_key, ""),
    )
    assert response.status_code == 200
    assert "jobs" in response.json()


def test_list_jobs_by_trigger_not_found_returns_404(
    test_api_key: str,
    get_test_client_wrapper,
    trigger_1: Tuple[Trigger, Any],
    trigger_2: Tuple[Trigger, Any],
):
    client = get_test_client_wrapper(
        fake_trigger_repository_contents={
            trigger_1[0].id: trigger_1[0],
            trigger_2[0].id: trigger_2[0],
        },
    )
    response = client.get(
        "/v1/docker-image-batch-jobs?trigger_id=some_trigger_id",
        auth=(test_api_key, ""),
    )
    assert response.status_code == 404


def test_list_jobs_by_trigger_unauthorized_returns_404(
    get_test_client_wrapper,
    trigger_1: Tuple[Trigger, Any],
    trigger_2: Tuple[Trigger, Any],
):
    client = get_test_client_wrapper(
        fake_trigger_repository_contents={
            trigger_1[0].id: trigger_1[0],
            trigger_2[0].id: trigger_2[0],
        },
    )
    response = client.get(
        f"/v1/docker-image-batch-jobs?trigger_id={trigger_1[0].id}",
        auth=("some_invalid_id", ""),
    )
    assert response.status_code == 404


def test_update_docker_image_batch_job_noop(
    test_api_key: str,
    get_test_client_wrapper,
    docker_image_batch_job_1: Tuple[DockerImageBatchJob, Any],
):
    batch_job_id = docker_image_batch_job_1[0].id
    client = get_test_client_wrapper(
        fake_docker_image_batch_job_gateway_contents={batch_job_id: docker_image_batch_job_1[0]}
    )
    response = client.put(
        f"/v1/docker-image-batch-jobs/{batch_job_id}",
        auth=(test_api_key, ""),
        json={"cancel": False},
    )
    assert response.status_code == 200
    assert response.json() == {"success": False}


def test_update_docker_image_batch_job_success(
    test_api_key: str,
    get_test_client_wrapper,
    docker_image_batch_job_1: Tuple[DockerImageBatchJob, Any],
):
    batch_job_id = docker_image_batch_job_1[0].id
    client = get_test_client_wrapper(
        fake_docker_image_batch_job_gateway_contents={batch_job_id: docker_image_batch_job_1[0]}
    )
    response = client.put(
        f"/v1/docker-image-batch-jobs/{batch_job_id}",
        auth=(test_api_key, ""),
        json={"cancel": True},
    )
    assert response.status_code == 200
    assert response.json() == {"success": True}


@pytest.mark.parametrize("cancel", [True, False])
def test_update_docker_image_batch_job_not_found(
    test_api_key: str,
    get_test_client_wrapper,
    docker_image_batch_job_1: Tuple[DockerImageBatchJob, Any],
    cancel: bool,
):
    batch_job_id = docker_image_batch_job_1[0].id
    client = get_test_client_wrapper(fake_docker_image_batch_job_gateway_contents={})
    response = client.put(
        f"/v1/docker-image-batch-jobs/{batch_job_id}",
        auth=(test_api_key, ""),
        json={"cancel": cancel},
    )
    assert response.status_code == 404


@pytest.mark.parametrize("cancel", [True, False])
def test_update_docker_image_batch_job_not_authorized(
    test_api_key_2: str,
    get_test_client_wrapper,
    docker_image_batch_job_1: Tuple[DockerImageBatchJob, Any],
    cancel: bool,
):
    batch_job_id = docker_image_batch_job_1[0].id
    client = get_test_client_wrapper(
        fake_docker_image_batch_job_gateway_contents={batch_job_id: docker_image_batch_job_1[0]}
    )
    response = client.put(
        f"/v1/docker-image-batch-jobs/{batch_job_id}",
        auth=(test_api_key_2, ""),
        json={"cancel": cancel},
    )
    assert response.status_code == 404
