import json
from datetime import datetime, timedelta
from typing import Any

import pytest
from model_engine_server.common.dtos.tasks import GetAsyncTaskV1Response, TaskStatus
from model_engine_server.infra.gateways import LiveAsyncModelEndpointInferenceGateway


@pytest.fixture
def fake_live_async_model_inference_gateway(fake_task_queue_gateway):
    return LiveAsyncModelEndpointInferenceGateway(task_queue_gateway=fake_task_queue_gateway)


@pytest.mark.asyncio
def test_task_create_get_url(
    fake_live_async_model_inference_gateway: LiveAsyncModelEndpointInferenceGateway,
    endpoint_predict_request_1,
):
    create_response = fake_live_async_model_inference_gateway.create_task(
        "test_topic", endpoint_predict_request_1[0], 60
    )
    task_id = create_response.task_id
    task_queue_gateway: Any = fake_live_async_model_inference_gateway.task_queue_gateway
    assert len(task_queue_gateway.queue) == 1
    assert task_queue_gateway.queue[task_id]["args"][0] == endpoint_predict_request_1[0].dict()
    assert (datetime.now() - task_queue_gateway.queue[task_id]["args"][1]) < timedelta(seconds=1)
    assert (
        task_queue_gateway.queue[task_id]["args"][2] == endpoint_predict_request_1[0].return_pickled
    )

    get_response_1 = fake_live_async_model_inference_gateway.get_task(task_id)
    assert get_response_1 == GetAsyncTaskV1Response(task_id=task_id, status=TaskStatus.PENDING)

    task_queue_gateway.do_task(result=42, status="success")

    get_response_2 = fake_live_async_model_inference_gateway.get_task(task_id)
    assert get_response_2 == GetAsyncTaskV1Response(
        task_id=task_id, status=TaskStatus.SUCCESS, result=42, status_code=200
    )


@pytest.mark.asyncio
def test_task_create_get_args_callback(
    fake_live_async_model_inference_gateway: LiveAsyncModelEndpointInferenceGateway,
    endpoint_predict_request_2,
):
    create_response = fake_live_async_model_inference_gateway.create_task(
        "test_topic", endpoint_predict_request_2[0], 60
    )
    task_id = create_response.task_id
    task_queue_gateway: Any = fake_live_async_model_inference_gateway.task_queue_gateway
    assert len(task_queue_gateway.queue) == 1
    assert task_queue_gateway.queue[task_id]["args"][0] == {
        "args": endpoint_predict_request_2[0].args.root,
        "url": None,
        "cloudpickle": None,
        "callback_auth": json.loads(endpoint_predict_request_2[0].callback_auth.json()),
        "callback_url": endpoint_predict_request_2[0].callback_url,
        "return_pickled": endpoint_predict_request_2[0].return_pickled,
        "destination_path": None,
    }
    assert (datetime.now() - task_queue_gateway.queue[task_id]["args"][1]) < timedelta(seconds=1)
    assert (
        task_queue_gateway.queue[task_id]["args"][2] == endpoint_predict_request_2[0].return_pickled
    )

    get_response_1 = fake_live_async_model_inference_gateway.get_task(task_id)
    assert get_response_1 == GetAsyncTaskV1Response(task_id=task_id, status=TaskStatus.PENDING)

    task_queue_gateway.do_task(result=42, status="success")

    get_response_2 = fake_live_async_model_inference_gateway.get_task(task_id)
    assert get_response_2 == GetAsyncTaskV1Response(
        task_id=task_id, status=TaskStatus.SUCCESS, result=42, status_code=200
    )


def test_task_expires_seconds_passed_to_queue(
    fake_live_async_model_inference_gateway: LiveAsyncModelEndpointInferenceGateway,
    endpoint_predict_request_1,
):
    """Test that task_expires_seconds is correctly passed to the task queue gateway."""
    task_expires_seconds = 3600  # 1 hour

    create_response = fake_live_async_model_inference_gateway.create_task(
        "test_topic", endpoint_predict_request_1[0], task_expires_seconds
    )
    task_id = create_response.task_id
    task_queue_gateway: Any = fake_live_async_model_inference_gateway.task_queue_gateway

    # Verify the expires parameter was passed to the task queue
    assert task_queue_gateway.queue[task_id]["expires"] == task_expires_seconds


def test_task_expires_seconds_with_different_values(
    fake_live_async_model_inference_gateway: LiveAsyncModelEndpointInferenceGateway,
    endpoint_predict_request_1,
):
    """Test task_expires_seconds with various values."""
    test_values = [60, 300, 3600, 86400, 604800]  # 1min, 5min, 1hr, 1day, 1week
    task_queue_gateway: Any = fake_live_async_model_inference_gateway.task_queue_gateway

    for expires_value in test_values:
        create_response = fake_live_async_model_inference_gateway.create_task(
            "test_topic", endpoint_predict_request_1[0], expires_value
        )
        task_id = create_response.task_id
        assert task_queue_gateway.queue[task_id]["expires"] == expires_value
