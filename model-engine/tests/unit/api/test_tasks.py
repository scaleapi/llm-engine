import asyncio
import threading
from typing import Any, Dict, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest
from model_engine_server.api import tasks_v1
from model_engine_server.common.dtos.tasks import (
    EndpointPredictV1Request,
    GetAsyncTaskV1Response,
    TaskStatus,
)
from model_engine_server.domain.entities import ModelBundle, ModelEndpoint
from model_engine_server.domain.exceptions import (
    InvalidRequestException,
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
    UpstreamServiceError,
)


@pytest.fixture
def fresh_task_limiter_cache():
    # The memoized limiter binds to the event loop that first creates it; clear it so
    # this test (and later tests on other loops) each get a limiter on their own loop.
    tasks_v1._get_task_limiter.cache_clear()
    yield
    tasks_v1._get_task_limiter.cache_clear()


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


@pytest.mark.asyncio
async def test_task_limiter_lazy_and_memoized(fresh_task_limiter_cache):
    limiter = tasks_v1._get_task_limiter()
    assert tasks_v1._get_task_limiter() is limiter
    assert limiter.total_tokens == 40


@pytest.mark.asyncio
async def test_task_polls_saturate_only_dedicated_limiter(
    fresh_task_limiter_cache,
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    test_api_key: str,
    get_async_test_client_wrapper,
):
    assert model_endpoint_1[0].infra_state is not None
    release = threading.Event()

    def blocking_execute(user, task_id):
        release.wait(timeout=10)
        return GetAsyncTaskV1Response(task_id=task_id, status=TaskStatus.PENDING)

    mock_use_case = MagicMock()
    mock_use_case.return_value.execute = blocking_execute
    client = get_async_test_client_wrapper(
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
    try:
        with patch(
            "model_engine_server.api.tasks_v1.GetAsyncInferenceTaskV1UseCase",
            mock_use_case,
        ):
            polls = [
                asyncio.create_task(
                    client.get("/v1/async-tasks/test_task_id", auth=(test_api_key, ""))
                )
                for _ in range(45)
            ]
            limiter = tasks_v1._get_task_limiter()
            deadline = asyncio.get_event_loop().time() + 10
            while limiter.borrowed_tokens < 40 and asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(0.01)
            # The dedicated limiter is saturated at exactly its capacity; the 5 extra polls queue.
            assert limiter.borrowed_tokens == 40
            # The default anyio threadpool must remain serviceable while polls are blocked.
            assert await asyncio.wait_for(anyio.to_thread.run_sync(lambda: "ok"), 2) == "ok"
            release.set()
            responses = await asyncio.gather(*polls)
        assert [response.status_code for response in responses] == [200] * 45
    finally:
        release.set()
        await client.aclose()


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
                    == b'data: {"status":"SUCCESS","result":null,"traceback":null,"status_code":200}\r\n\r\n'
                )
                count += 1
            assert count == 1


def test_get_async_task_response_body_and_size_metric(
    model_bundle_1_v1: Tuple[ModelBundle, Any],
    model_endpoint_1: Tuple[ModelEndpoint, Any],
    test_api_key: str,
    get_test_client_wrapper,
    monkeypatch,
):
    """The off-loop serialized Response must stay contract-identical JSON and emit
    the per-tenant result-size distribution."""
    from model_engine_server.api import tasks_v1

    emitted = []
    monkeypatch.setattr(
        tasks_v1.statsd,
        "distribution",
        lambda name, value, tags: emitted.append((name, value, tags)),
    )
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
    body = response.json()
    assert body["task_id"] == "test_task_id"
    assert set(body) == {"task_id", "status", "result", "traceback", "status_code"}
    assert len(emitted) == 1
    name, value, tags = emitted[0]
    assert name == "model_engine.async_task.result_bytes"
    assert value == len(response.content)
    assert any(tag.startswith("user_id:") for tag in tags)
    assert any(tag.startswith("status:") for tag in tags)


def test_emit_result_size_counts_bytes_not_characters(monkeypatch):
    from model_engine_server.api import tasks_v1
    from model_engine_server.common.dtos.tasks import GetAsyncTaskV1Response, TaskStatus

    emitted = []
    monkeypatch.setattr(
        tasks_v1.statsd, "distribution", lambda name, value, tags: emitted.append(value)
    )
    task = GetAsyncTaskV1Response(task_id="t", status=TaskStatus.SUCCESS)
    body = "ü" * 10  # 10 characters, 20 UTF-8 bytes
    tasks_v1._emit_result_size(task, body.encode("utf-8"), "user")
    assert emitted == [20]
