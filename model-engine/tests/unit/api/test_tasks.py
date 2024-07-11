from typing import Any, Dict, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from model_engine_server.common.dtos.tasks import EndpointPredictV1Request
from model_engine_server.domain.entities import ModelBundle, ModelEndpoint
from model_engine_server.domain.exceptions import (
    InvalidRequestException,
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
    UpstreamServiceError,
)


def test_create_async_task_success(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    endpoint_predict_request_1: Tuple[EndpointPredictV1Request, Dict[str, Any]],
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
    response = client.post(
        f"/v1/async-tasks?model_endpoint_id={model_endpoint_1[0].record.id}",
        auth=(test_api_key, ""),
        json=endpoint_predict_request_1[1],
    )
    assert response.status_code == 200
    assert response.json() == {"task_id": "test_task_id"}


def test_create_async_task_raises_404_not_authorized(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    endpoint_predict_request_1: Tuple[EndpointPredictV1Request, Dict[str, Any]],
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
    response = client.post(
        f"/v1/async-tasks?model_endpoint_id={model_endpoint_1[0].record.id}",
        auth=("invalid_user_id", ""),
        json=endpoint_predict_request_1[1],
    )
    assert response.status_code == 404


def test_create_async_task_raises_404_not_found(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    endpoint_predict_request_1: Tuple[EndpointPredictV1Request, Dict[str, Any]],
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
    response = client.post(
        "/v1/async-tasks?model_endpoint_id=invalid_model_endpoint_id",
        auth=(test_api_key, ""),
        json=endpoint_predict_request_1[1],
    )
    assert response.status_code == 404


def test_create_async_task_raises_400_invalid_requests(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    endpoint_predict_request_1: Tuple[EndpointPredictV1Request, Dict[str, Any]],
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
    mock_use_case = MagicMock()
    mock_use_case.return_value.execute = MagicMock(side_effect=InvalidRequestException)
    with patch(
        "model_engine_server.api.tasks_v1.CreateAsyncInferenceTaskV1UseCase",
        mock_use_case,
    ):
        response = client.post(
            "/v1/async-tasks?model_endpoint_id=invalid_model_endpoint_id",
            auth=(test_api_key, ""),
            json=endpoint_predict_request_1[1],
        )
        assert response.status_code == 400


def test_get_async_task_success(
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
        "/v1/async-tasks/test_task_id",
        auth=(test_api_key, ""),
    )
    assert response.status_code == 200


def test_get_async_task_raises_404_object_not_found(
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
    mock_use_case = MagicMock()
    mock_use_case.return_value.execute = MagicMock(side_effect=ObjectNotFoundException)
    with patch(
        "model_engine_server.api.tasks_v1.GetAsyncInferenceTaskV1UseCase",
        mock_use_case,
    ):
        response = client.get(
            "/v1/async-tasks/test_task_id",
            auth=(test_api_key, ""),
        )
        assert response.status_code == 404


def test_get_async_task_raises_404_object_not_authorized(
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
    mock_use_case = MagicMock()
    mock_use_case.return_value.execute = MagicMock(side_effect=ObjectNotAuthorizedException)
    with patch(
        "model_engine_server.api.tasks_v1.GetAsyncInferenceTaskV1UseCase",
        mock_use_case,
    ):
        response = client.get(
            "/v1/async-tasks/test_task_id",
            auth=(test_api_key, ""),
        )
        assert response.status_code == 404


def test_create_sync_task_success(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_2: Tuple[ModelEndpoint, Any],
    endpoint_predict_request_1: Tuple[EndpointPredictV1Request, Dict[str, Any]],
    test_api_key: str,
    get_test_client_wrapper,
):
    assert model_endpoint_2[0].infra_state is not None
    client = get_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={
            model_bundle_1_v1[0].id: model_bundle_1_v1[0],
        },
        fake_model_endpoint_record_repository_contents={
            model_endpoint_2[0].record.id: model_endpoint_2[0].record,
        },
        fake_model_endpoint_infra_gateway_contents={
            model_endpoint_2[0].infra_state.deployment_name: model_endpoint_2[0].infra_state,
        },
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    )
    response = client.post(
        f"/v1/sync-tasks?model_endpoint_id={model_endpoint_2[0].record.id}",
        auth=(test_api_key, ""),
        json=endpoint_predict_request_1[1],
    )
    assert response.status_code == 200
    assert response.json()


def test_create_sync_task_raises_404_not_authorized(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    endpoint_predict_request_1: Tuple[EndpointPredictV1Request, Dict[str, Any]],
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
    response = client.post(
        f"/v1/sync-tasks?model_endpoint_id={model_endpoint_1[0].record.id}",
        auth=("invalid_user_id", ""),
        json=endpoint_predict_request_1[1],
    )
    assert response.status_code == 404


def test_create_sync_task_raises_404_not_found(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    endpoint_predict_request_1: Tuple[EndpointPredictV1Request, Dict[str, Any]],
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
    response = client.post(
        "/v1/sync-tasks?model_endpoint_id=invalid_model_endpoint_id",
        auth=(test_api_key, ""),
        json=endpoint_predict_request_1[1],
    )
    assert response.status_code == 404


def test_create_sync_task_returns_failure(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    endpoint_predict_request_1: Tuple[EndpointPredictV1Request, Dict[str, Any]],
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
    mock_use_case = MagicMock()
    mock_use_case.return_value.execute = AsyncMock(
        side_effect=UpstreamServiceError(400, b"test_content")
    )
    with patch(
        "model_engine_server.api.tasks_v1.CreateSyncInferenceTaskV1UseCase",
        mock_use_case,
    ):
        response = client.post(
            f"/v1/sync-tasks?model_endpoint_id={model_endpoint_1[0].record.id}",
            auth=(test_api_key, ""),
            json=endpoint_predict_request_1[1],
        )
        assert response.status_code == 200
        assert response.json()["status"] == "FAILURE"


@pytest.mark.asyncio
async def test_create_streaming_task_success(
    model_bundle_5: ModelBundle,
    model_endpoint_streaming: ModelEndpoint,
    endpoint_predict_request_1: Tuple[EndpointPredictV1Request, Dict[str, Any]],
    test_api_key: str,
    get_async_test_client_wrapper,
):
    assert model_endpoint_streaming.infra_state is not None
    async with get_async_test_client_wrapper(
        fake_docker_repository_image_always_exists=True,
        fake_model_bundle_repository_contents={
            model_bundle_5.id: model_bundle_5,
        },
        fake_model_endpoint_record_repository_contents={
            model_endpoint_streaming.record.id: model_endpoint_streaming.record,
        },
        fake_model_endpoint_infra_gateway_contents={
            model_endpoint_streaming.infra_state.deployment_name: model_endpoint_streaming.infra_state,
        },
        fake_batch_job_record_repository_contents={},
        fake_batch_job_progress_gateway_contents={},
        fake_docker_image_batch_job_bundle_repository_contents={},
    ) as client:
        async with client.stream(
            method="POST",
            url=f"/v1/streaming-tasks?model_endpoint_id={model_endpoint_streaming.record.id}",
            auth=(test_api_key, ""),
            json=endpoint_predict_request_1[1],
        ) as response:
            assert response.status_code == 200
            count = 0
            async for message in response.aiter_bytes():
                assert (
                    message
                    == b'data: {"status":"SUCCESS","result":null,"traceback":null}\r\n\r\n'
                )
                count += 1
            assert count == 1
