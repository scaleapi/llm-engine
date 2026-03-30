from unittest.mock import MagicMock, patch

import pytest
from model_engine_server.common.dtos.model_endpoints import BrokerType
from model_engine_server.common.dtos.tasks import TaskStatus
from model_engine_server.domain.exceptions import BrokerUnavailableException
from model_engine_server.infra.gateways.celery_task_queue_gateway import (
    CeleryTaskQueueGateway,
    _is_broker_connection_error,
)


# Stand-in for azure.servicebus.exceptions.ServiceBusError so tests run
# without the azure-servicebus package installed.
class _MockServiceBusError(Exception):
    pass


@pytest.fixture
def gateway():
    return CeleryTaskQueueGateway(broker_type=BrokerType.REDIS, tracing_gateway=MagicMock())


@pytest.fixture
def servicebus_gateway(monkeypatch):
    """Gateway configured for ServiceBus with a mock ServiceBusError class."""
    monkeypatch.setattr(
        "model_engine_server.infra.gateways.celery_task_queue_gateway.ServiceBusError",
        _MockServiceBusError,
    )
    return CeleryTaskQueueGateway(broker_type=BrokerType.SERVICEBUS, tracing_gateway=MagicMock())


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


# ---------------------------------------------------------------------------
# Tests for broker connection retry logic
# ---------------------------------------------------------------------------


class TestIsBrokerConnectionError:
    def test_servicebus_error_detected(self, monkeypatch):
        monkeypatch.setattr(
            "model_engine_server.infra.gateways.celery_task_queue_gateway.ServiceBusError",
            _MockServiceBusError,
        )
        assert _is_broker_connection_error(_MockServiceBusError("oops"), BrokerType.SERVICEBUS)

    def test_chained_servicebus_error_detected(self, monkeypatch):
        monkeypatch.setattr(
            "model_engine_server.infra.gateways.celery_task_queue_gateway.ServiceBusError",
            _MockServiceBusError,
        )
        cause = _MockServiceBusError("inner")
        wrapper = RuntimeError("wrapper")
        wrapper.__cause__ = cause
        assert _is_broker_connection_error(wrapper, BrokerType.SERVICEBUS)

    def test_implicit_context_not_matched(self, monkeypatch):
        """Implicit chaining (__context__) should NOT trigger a retry."""
        monkeypatch.setattr(
            "model_engine_server.infra.gateways.celery_task_queue_gateway.ServiceBusError",
            _MockServiceBusError,
        )
        wrapper = RuntimeError("cleanup failed")
        wrapper.__context__ = _MockServiceBusError("original")
        assert not _is_broker_connection_error(wrapper, BrokerType.SERVICEBUS)

    def test_non_servicebus_error_not_detected(self, monkeypatch):
        monkeypatch.setattr(
            "model_engine_server.infra.gateways.celery_task_queue_gateway.ServiceBusError",
            _MockServiceBusError,
        )
        assert not _is_broker_connection_error(ValueError("bad"), BrokerType.SERVICEBUS)

    def test_redis_errors_never_retried(self, monkeypatch):
        monkeypatch.setattr(
            "model_engine_server.infra.gateways.celery_task_queue_gateway.ServiceBusError",
            _MockServiceBusError,
        )
        assert not _is_broker_connection_error(_MockServiceBusError("oops"), BrokerType.REDIS)

    def test_servicebus_error_none_import(self, monkeypatch):
        """When azure-servicebus isn't installed, ServiceBusError is None."""
        monkeypatch.setattr(
            "model_engine_server.infra.gateways.celery_task_queue_gateway.ServiceBusError",
            None,
        )
        assert not _is_broker_connection_error(Exception("x"), BrokerType.SERVICEBUS)


class TestSendTaskWithRetry:
    """Tests for _send_task_with_retry via the public send_task() method."""

    def test_retries_once_on_connection_error_then_succeeds(self, servicebus_gateway):
        mock_result = MagicMock()
        mock_result.id = "task-ok"
        mock_result.state = "PENDING"

        mock_dest = MagicMock()
        mock_dest.send_task.side_effect = [
            _MockServiceBusError("stale AMQP client"),
            mock_result,
        ]

        with patch.object(servicebus_gateway, "_get_celery_dest", return_value=mock_dest):
            response = servicebus_gateway.send_task(task_name="test.task", queue_name="test-queue")

        assert response.task_id == "task-ok"
        assert mock_dest.send_task.call_count == 2
        mock_dest.pool.force_close_all.assert_called_once()

    def test_retry_fails_raises_broker_unavailable(self, servicebus_gateway):
        mock_dest = MagicMock()
        mock_dest.send_task.side_effect = _MockServiceBusError("dead")

        with patch.object(servicebus_gateway, "_get_celery_dest", return_value=mock_dest):
            with pytest.raises(BrokerUnavailableException, match="after retry"):
                servicebus_gateway.send_task(task_name="test.task", queue_name="test-queue")

        # First attempt + one retry = 2 calls.
        assert mock_dest.send_task.call_count == 2
        mock_dest.pool.force_close_all.assert_called_once()

    def test_non_connection_error_propagates_immediately(self, servicebus_gateway):
        mock_dest = MagicMock()
        mock_dest.send_task.side_effect = ValueError("bad payload")

        with patch.object(servicebus_gateway, "_get_celery_dest", return_value=mock_dest):
            with pytest.raises(ValueError, match="bad payload"):
                servicebus_gateway.send_task(task_name="test.task", queue_name="test-queue")

        # Should NOT have retried.
        assert mock_dest.send_task.call_count == 1
        mock_dest.pool.force_close_all.assert_not_called()

    def test_redis_gateway_does_not_retry(self, monkeypatch):
        """Redis broker errors are never retried (no connection error match)."""
        monkeypatch.setattr(
            "model_engine_server.infra.gateways.celery_task_queue_gateway.ServiceBusError",
            _MockServiceBusError,
        )
        gw = CeleryTaskQueueGateway(broker_type=BrokerType.REDIS, tracing_gateway=MagicMock())

        mock_dest = MagicMock()
        mock_dest.send_task.side_effect = ConnectionError("redis down")

        with patch.object(gw, "_get_celery_dest", return_value=mock_dest):
            with pytest.raises(ConnectionError, match="redis down"):
                gw.send_task(task_name="test.task", queue_name="test-queue")

        assert mock_dest.send_task.call_count == 1
