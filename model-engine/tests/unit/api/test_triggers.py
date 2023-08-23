from typing import Any, Dict, Tuple

from fastapi.testclient import TestClient
from model_engine_server.domain.entities import Trigger
from model_engine_server.domain.entities.docker_image_batch_job_bundle_entity import (
    DockerImageBatchJobBundle,
)


def test_create_trigger_success(
    create_trigger_request: Dict[str, Any],
    test_api_key: str,
    get_test_client_wrapper,
    docker_image_batch_job_bundle_3_v1: Tuple[DockerImageBatchJobBundle, Any],
):
    # populate docker image batch bundle repo
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={},
        fake_model_endpoint_record_repository_contents={},
        fake_model_endpoint_infra_gateway_contents={},
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={
            docker_image_batch_job_bundle_3_v1[0].id: docker_image_batch_job_bundle_3_v1[0],
        },
    )

    response_1 = client.post(
        "/v1/triggers",
        auth=(test_api_key, ""),
        json=create_trigger_request,
    )
    assert response_1.status_code == 200
    assert "trigger_id" in response_1.json()


def test_create_trigger_batch_bundle_not_found_returns_404(
    create_trigger_request: Dict[str, Any],
    test_api_key: str,
    simple_client: TestClient,
):
    response_1 = simple_client.post(
        "/v1/triggers",
        auth=(test_api_key, ""),
        json=create_trigger_request,
    )
    assert response_1.status_code == 404


def test_create_trigger_batch_bundle_unauthorized_returns_400(
    create_trigger_request: Dict[str, Any],
    test_api_key: str,
    get_test_client_wrapper,
    docker_image_batch_job_bundle_3_v1: Tuple[DockerImageBatchJobBundle, Any],
):
    # populate docker image batch bundle repo
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={},
        fake_model_endpoint_record_repository_contents={},
        fake_model_endpoint_infra_gateway_contents={},
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={
            docker_image_batch_job_bundle_3_v1[0].id: docker_image_batch_job_bundle_3_v1[0],
        },
    )

    response_1 = client.post(
        "/v1/triggers",
        auth=("some_invalid_id", ""),
        json=create_trigger_request,
    )
    assert response_1.status_code == 404


def test_create_trigger_bad_cron_returns_400(
    create_trigger_request: Dict[str, Any],
    test_api_key: str,
    get_test_client_wrapper,
    docker_image_batch_job_bundle_3_v1: Tuple[DockerImageBatchJobBundle, Any],
):
    # populate docker image batch bundle repo
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={},
        fake_model_endpoint_record_repository_contents={},
        fake_model_endpoint_infra_gateway_contents={},
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={
            docker_image_batch_job_bundle_3_v1[0].id: docker_image_batch_job_bundle_3_v1[0],
        },
    )

    create_trigger_request["cron_schedule"] = "field is wrong"
    response_1 = client.post(
        "/v1/triggers",
        auth=(test_api_key, ""),
        json=create_trigger_request,
    )
    assert response_1.status_code == 400


def test_list_triggers_success(
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
        "/v1/triggers",
        auth=(test_api_key, ""),
    )
    assert response.status_code == 200
    assert response.json() == {
        "triggers": [trigger_1[1], trigger_2[1]],
    }


def test_get_trigger_success(
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
        f"/v1/triggers/{trigger_1[0].id}",
        auth=(test_api_key, ""),
    )
    assert response.status_code == 200
    assert response.json() == trigger_1[1]


def test_get_trigger_not_found_returns_404(
    test_api_key: str,
    simple_client: TestClient,
):
    response = simple_client.get(
        "/v1/triggers/some_trigger_id",
        auth=(test_api_key, ""),
    )
    assert response.status_code == 404


def test_get_trigger_unauthorized_returns_404(
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
        f"/v1/triggers/{trigger_1[0].id}",
        auth=("some_invalid_id", ""),
    )
    assert response.status_code == 404


def test_update_trigger_success(
    update_trigger_request: Dict[str, Any],
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
    response = client.put(
        f"/v1/triggers/{trigger_1[0].id}",
        auth=(test_api_key, ""),
        json=update_trigger_request,
    )
    assert response.json().get("success")

    response = client.get(
        f"/v1/triggers/{trigger_1[0].id}",
        auth=(test_api_key, ""),
    )
    assert response.status_code == 200
    assert response.json().get("cron_schedule") == "0 * * * *"


def test_update_trigger_not_found_returns_404(
    update_trigger_request: Dict[str, Any],
    test_api_key: str,
    simple_client: TestClient,
):
    response = simple_client.put(
        "/v1/triggers/some_trigger_id",
        auth=(test_api_key, ""),
        json=update_trigger_request,
    )
    assert response.status_code == 404


def test_update_trigger_unauthorized_returns_404(
    update_trigger_request: Dict[str, Any],
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
    response = client.put(
        f"/v1/triggers/{trigger_1[0].id}",
        auth=("some_invalid_id", ""),
        json=update_trigger_request,
    )
    assert response.status_code == 404


def test_update_trigger_bad_cron_returns_400(
    update_trigger_request: Dict[str, Any],
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

    update_trigger_request["cron_schedule"] = "field is wrong"
    response = client.put(
        f"/v1/triggers/{trigger_1[0].id}",
        auth=(test_api_key, ""),
        json=update_trigger_request,
    )
    assert response.status_code == 400


def test_delete_trigger_success(
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
    response = client.delete(
        f"/v1/triggers/{trigger_1[0].id}",
        auth=(test_api_key, ""),
    )
    assert response.json().get("success")

    response = client.get(
        f"/v1/triggers/{trigger_1[0].id}",
        auth=(test_api_key, ""),
    )
    assert response.status_code == 404


def test_delete_trigger_not_found_returns_404(
    test_api_key: str,
    simple_client: TestClient,
):
    response = simple_client.delete(
        "/v1/triggers/some_trigger_id",
        auth=(test_api_key, ""),
    )
    assert response.status_code == 404


def test_delete_trigger_unauthorized_returns_404(
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
    response = client.delete(
        f"/v1/triggers/{trigger_1[0].id}",
        auth=("some_invalid_id", ""),
    )
    assert response.status_code == 404
