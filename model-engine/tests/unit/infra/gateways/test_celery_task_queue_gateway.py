from unittest.mock import MagicMock, patch

import pytest
from model_engine_server.common.dtos.model_endpoints import BrokerType
from model_engine_server.common.dtos.tasks import TaskStatus
from model_engine_server.infra.gateways.celery_task_queue_gateway import CeleryTaskQueueGateway


@pytest.fixture
def gateway():
    return CeleryTaskQueueGateway(broker_type=BrokerType.REDIS, tracing_gateway=MagicMock())


def _make_async_result(state, result=None, traceback=None):
    mock_result = MagicMock()
    mock_result.state = state
    mock_result.result = result
    mock_result.traceback = traceback
    return mock_result


def test_get_task_failure_with_traceback(gateway):
    """FAILURE responses surface the traceback string from res.traceback."""
    async_result = _make_async_result(
        state="FAILURE",
        result=RuntimeError("Out of memory"),
        traceback="Traceback (most recent call last): ...\nRuntimeError: Out of memory",
    )

    with patch.object(gateway, "_get_celery_dest") as mock_dest:
        mock_dest.return_value.AsyncResult.return_value = async_result
        response = gateway.get_task("task-123")

    assert response.status == TaskStatus.FAILURE
    assert response.result.root == "Out of memory"
    assert (
        response.traceback == "Traceback (most recent call last): ...\nRuntimeError: Out of memory"
    )
    assert response.status_code is None


def test_get_task_failure_with_no_traceback(gateway):
    """FAILURE responses with no traceback (e.g. hard pod kill) still surface the exception message."""
    async_result = _make_async_result(
        state="FAILURE",
        result=RuntimeError("crash"),
        traceback=None,
    )

    with patch.object(gateway, "_get_celery_dest") as mock_dest:
        mock_dest.return_value.AsyncResult.return_value = async_result
        response = gateway.get_task("task-456")

    assert response.status == TaskStatus.FAILURE
    assert response.result.root == "crash"
    assert response.traceback is None


def test_get_task_failure_with_none_result(gateway):
    """FAILURE responses where res.result is None (no exception recorded) should have result=None."""
    async_result = _make_async_result(state="FAILURE", result=None, traceback=None)

    with patch.object(gateway, "_get_celery_dest") as mock_dest:
        mock_dest.return_value.AsyncResult.return_value = async_result
        response = gateway.get_task("task-789")

    assert response.status == TaskStatus.FAILURE
    assert response.result is None
    assert response.traceback is None
