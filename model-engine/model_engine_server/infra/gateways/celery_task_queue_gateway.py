import json
import threading
import time
from typing import Any, Dict, List, Optional

import botocore
from model_engine_server.common.dtos.model_endpoints import BrokerType
from model_engine_server.common.dtos.tasks import (
    CreateAsyncTaskV1Response,
    GetAsyncTaskV1Response,
    TaskStatus,
)
from model_engine_server.core.celery import TaskVisibility, celery_app
from model_engine_server.core.celery.app import get_redis_endpoint, get_redis_instance
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.core.tracing.tracing_gateway import TracingGateway
from model_engine_server.domain.exceptions import BrokerUnavailableException, InvalidRequestException
from model_engine_server.domain.gateways.task_queue_gateway import TaskQueueGateway

try:
    from azure.servicebus.exceptions import ServiceBusError
except ImportError:
    ServiceBusError = None  # type: ignore[assignment,misc]

logger = make_logger(logger_name())
_cloud_provider = infra_config().cloud_provider
backend_protocol = (
    "abs" if _cloud_provider == "azure" else ("redis" if _cloud_provider == "gcp" else "s3")
)

celery_redis = celery_app(
    None,
    s3_bucket=infra_config().s3_bucket,
    broker_type=str(BrokerType.REDIS.value),
    backend_protocol=backend_protocol,
)
celery_redis_24h = celery_app(
    None,
    s3_bucket=infra_config().s3_bucket,
    broker_type=str(BrokerType.REDIS.value),
    task_visibility=TaskVisibility.VISIBILITY_24H,
    backend_protocol=backend_protocol,
)
celery_sqs = celery_app(
    None,
    s3_bucket=infra_config().s3_bucket,
    broker_type=str(BrokerType.SQS.value),
    backend_protocol=backend_protocol,
)
celery_servicebus = celery_app(
    None, broker_type=str(BrokerType.SERVICEBUS.value), backend_protocol=backend_protocol
)

# ---------------------------------------------------------------------------
# Broker health tracking & connection-error detection
# ---------------------------------------------------------------------------

_BROKER_RETRY_ATTEMPTS = 3
_BROKER_RETRY_WAIT_MULTIPLIER = 1  # seconds
_BROKER_RETRY_WAIT_MAX = 5  # seconds


class _BrokerHealthTracker:
    """Thread-safe tracker for consecutive broker send failures.

    Used by the readiness probe to determine whether the pod should be
    removed from the K8s service load balancer.
    """

    def __init__(self, failure_threshold: int = 3):
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._failure_threshold = failure_threshold

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1

    @property
    def is_healthy(self) -> bool:
        with self._lock:
            return self._consecutive_failures < self._failure_threshold


broker_health_tracker = _BrokerHealthTracker(failure_threshold=3)


def _is_broker_connection_error(exc: Exception, broker_type: BrokerType) -> bool:
    """Return True if *exc* is a transient broker connection error that
    warrants a retry with a fresh connection pool.

    Dispatches per broker type so each transport can define its own set of
    retryable errors.  Today only ServiceBus is handled; Redis / SQS errors
    propagate immediately.  Extending to other brokers is a one-line change
    in this function.
    """
    if broker_type == BrokerType.SERVICEBUS:
        if ServiceBusError is None:
            return False
        if isinstance(exc, ServiceBusError):
            return True
        # Also match when kombu wraps the real error as a chained cause.
        cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
        return isinstance(cause, ServiceBusError)
    # Redis / SQS: no retry today.
    return False


class CeleryTaskQueueGateway(TaskQueueGateway):
    def __init__(self, broker_type: BrokerType, tracing_gateway: TracingGateway):
        self.broker_type = broker_type
        assert self.broker_type in [
            BrokerType.SQS,
            BrokerType.REDIS,
            BrokerType.REDIS_24H,
            BrokerType.SERVICEBUS,
        ]
        self.tracing_gateway = tracing_gateway

        # Log initialization
        if not infra_config().debug_mode:  # pragma: no cover
            logger.info(
                f"Initializing CeleryTaskQueueGateway with broker: {self.broker_type.value}"
            )
        else:  # pragma: no cover
            logger.info(
                "Initializing CeleryTaskQueueGateway",
                extra={
                    "broker_type": self.broker_type.value,
                    "backend_protocol": backend_protocol,
                    "infra_config_cloud_provider": infra_config().cloud_provider,
                },
            )

    def _get_celery_dest(self):
        if self.broker_type == BrokerType.SQS:
            return celery_sqs
        elif self.broker_type == BrokerType.REDIS_24H:
            return celery_redis_24h
        elif self.broker_type == BrokerType.REDIS:
            return celery_redis
        else:
            return celery_servicebus

    # ------------------------------------------------------------------
    # Retry helpers for transient broker connection errors
    # ------------------------------------------------------------------

    @staticmethod
    def _reset_connection_pool(celery_dest) -> None:
        """Close all pooled connections so the next send creates a fresh one."""
        try:
            celery_dest.pool.force_close_all()
            logger.info("Reset Celery broker connection pool")
        except Exception as e:
            logger.warning(f"Failed to reset Celery connection pool: {e}")

    def _send_task_with_retry(
        self,
        celery_dest,
        task_name: str,
        args,
        kwargs,
        queue_name: str,
        expires,
    ):
        """Publish a task, retrying on transient broker connection errors.

        On each retry the connection pool is reset so stale AMQP senders are
        evicted. Non-connection errors are re-raised immediately.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, _BROKER_RETRY_ATTEMPTS + 1):
            try:
                result = celery_dest.send_task(
                    name=task_name,
                    args=args,
                    kwargs=kwargs,
                    queue=queue_name,
                    expires=expires,
                )
                broker_health_tracker.record_success()
                return result
            except Exception as exc:
                if not _is_broker_connection_error(exc, self.broker_type):
                    raise  # Not a connection error; let the outer handler deal with it.

                last_exc = exc
                logger.warning(
                    "Broker send_task failed (attempt %d/%d): %s",
                    attempt,
                    _BROKER_RETRY_ATTEMPTS,
                    exc,
                    extra={
                        "queue_name": queue_name,
                        "task_name": task_name,
                        "broker_type": self.broker_type.value,
                        "error_type": type(exc).__name__,
                    },
                )
                self._reset_connection_pool(celery_dest)

                if attempt < _BROKER_RETRY_ATTEMPTS:
                    backoff = min(
                        _BROKER_RETRY_WAIT_MULTIPLIER * (2 ** (attempt - 1)),
                        _BROKER_RETRY_WAIT_MAX,
                    )
                    time.sleep(backoff)

        # All retries exhausted.
        broker_health_tracker.record_failure()
        logger.error(
            "Broker send_task failed after %d attempts",
            _BROKER_RETRY_ATTEMPTS,
            extra={
                "queue_name": queue_name,
                "task_name": task_name,
                "broker_type": self.broker_type.value,
                "error_type": type(last_exc).__name__,
                "error_message": str(last_exc),
            },
        )
        raise BrokerUnavailableException(
            f"Failed to send task after {_BROKER_RETRY_ATTEMPTS} attempts: {last_exc}"
        ) from last_exc

    def _log_broker_details(self, celery_dest, queue_name: str):
        """Log detailed broker connection information for debugging"""
        if not infra_config().debug_mode:  # pragma: no cover
            return
        try:
            broker_url = celery_dest.conf.broker_url
            backend_url = celery_dest.conf.result_backend

            logger.info(  # pragma: no cover
                "Celery broker details",
                extra={
                    "broker_type": self.broker_type.value,
                    "broker_url": broker_url,
                    "backend_url": backend_url,
                    "queue_name": queue_name,
                    "celery_app_name": celery_dest.main,
                },
            )

            # For Redis, also check the actual connection
            if self.broker_type in [BrokerType.REDIS, BrokerType.REDIS_24H]:
                try:
                    redis_endpoint = get_redis_endpoint(0)  # Default db
                    logger.info(  # pragma: no cover
                        "Redis connection details",
                        extra={
                            "redis_endpoint": redis_endpoint,
                            "queue_name": queue_name,
                        },
                    )

                    # Test Redis connection and queue inspection
                    redis_client = get_redis_instance(0)
                    queue_length_before = redis_client.llen(queue_name)
                    redis_client.close()

                    logger.info(  # pragma: no cover
                        "Pre-send queue state",
                        extra={
                            "queue_name": queue_name,
                            "queue_length_before_send": queue_length_before,
                        },
                    )

                except Exception as e:
                    logger.warning(  # pragma: no cover
                        "Failed to inspect Redis queue state",
                        extra={
                            "queue_name": queue_name,
                            "error": str(e),
                        },
                    )
        except Exception as e:
            logger.warning(  # pragma: no cover
                "Failed to log broker details",
                extra={
                    "broker_type": self.broker_type.value,
                    "error": str(e),
                },
            )

    def send_task(
        self,
        task_name: str,
        queue_name: str,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        expires: Optional[int] = None,
    ) -> CreateAsyncTaskV1Response:
        send_start_time = time.time()

        # Log detailed send attempt
        if infra_config().debug_mode:  # pragma: no cover
            logger.info(
                "Starting task send operation",
                extra={
                    "task_name": task_name,
                    "queue_name": queue_name,
                    "broker_type": self.broker_type.value,
                    "args_provided": args is not None,
                    "kwargs_provided": kwargs is not None,
                    "args_count": len(args) if args else 0,
                    "kwargs_keys": list(kwargs.keys()) if kwargs else [],
                    "expires": expires,
                },
            )

        # Used for both endpoint infra creation and async tasks
        celery_dest = self._get_celery_dest()
        kwargs = kwargs or {}

        # Log broker details for debugging
        if infra_config().debug_mode:  # pragma: no cover
            self._log_broker_details(celery_dest, queue_name)

        with self.tracing_gateway.create_span("send_task_to_queue") as span:
            kwargs.update(self.tracing_gateway.encode_trace_kwargs())

            try:
                if infra_config().debug_mode:  # pragma: no cover
                    logger.info(
                        "Calling celery send_task",
                        extra={
                            "task_name": task_name,
                            "queue_name": queue_name,
                            "final_kwargs_keys": list(kwargs.keys()),
                        },
                    )

                res = self._send_task_with_retry(
                    celery_dest, task_name, args, kwargs, queue_name, expires
                )

                if infra_config().debug_mode:  # pragma: no cover
                    send_duration = time.time() - send_start_time

                span.input = {
                    "queue_name": queue_name,
                    "args": json.loads(json.dumps(args, indent=4, sort_keys=True, default=str)),
                    "task_id": res.id,
                    "task_name": task_name,
                }
                span.output = {"task_id": res.id}
                if infra_config().debug_mode:  # pragma: no cover
                    logger.info(
                        "Task sent successfully",
                        extra={
                            "task_id": res.id,
                            "task_name": task_name,
                            "queue_name": queue_name,
                            "send_duration_seconds": send_duration,
                            "broker_type": self.broker_type.value,
                            "task_state": res.state,
                        },
                    )

            except BrokerUnavailableException:
                raise

            except botocore.exceptions.ClientError as e:
                send_duration = time.time() - send_start_time

                if infra_config().debug_mode:  # pragma: no cover
                    # Debug mode - detailed error logging
                    logger.error(
                        "ClientError sending task to queue",
                        extra={
                            "queue_name": queue_name,
                            "task_name": task_name,
                            "broker_type": self.broker_type.value,
                            "error_code": getattr(e, "response", {}).get("Error", {}).get("Code"),
                            "error_message": str(e),
                            "send_duration_seconds": send_duration,
                        },
                    )
                else:
                    # Production mode - simple error with stack trace
                    logger.exception(f"Error sending task to queue {queue_name}: {e}")

                raise InvalidRequestException(f"Error sending celery task: {e}")

            except Exception as e:
                send_duration = time.time() - send_start_time

                if infra_config().debug_mode:  # pragma: no cover
                    # Debug mode - detailed error logging
                    logger.error(
                        "Unexpected error sending task to queue",
                        extra={
                            "queue_name": queue_name,
                            "task_name": task_name,
                            "broker_type": self.broker_type.value,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "send_duration_seconds": send_duration,
                        },
                    )
                else:
                    # Production mode - simple error with stack trace
                    logger.exception(f"Error sending task to queue {queue_name}: {e}")

                raise

            logger.info(f"Task {res.id} sent to queue {queue_name} from gateway")

            return CreateAsyncTaskV1Response(task_id=res.id)

    def get_task(self, task_id: str) -> GetAsyncTaskV1Response:
        # Only used for async tasks
        celery_dest = self._get_celery_dest()
        res = celery_dest.AsyncResult(task_id)
        response_state = res.state
        if response_state == "SUCCESS":
            # No longer wrapping things in the result itself, since the DTO already has a 'result' key:
            # result_dict = (
            #    response_result if type(response_result) is dict else {"result": response_result}
            # )
            status_code = None
            result = res.result
            if isinstance(result, dict) and "status_code" in result:
                # Filter out status code from result if it was added by the forwarder
                # This is admittedly kinda hacky and would technically introduce an edge case
                # if we ever decide not to have async tasks wrap response.
                status_code = result["status_code"]
                del result["status_code"]
            return GetAsyncTaskV1Response(
                task_id=task_id,
                status=TaskStatus.SUCCESS,
                result=result,
                status_code=status_code,
            )

        elif response_state == "FAILURE":
            return GetAsyncTaskV1Response(
                task_id=task_id,
                status=TaskStatus.FAILURE,
                result=str(res.result) if res.result is not None else None,
                traceback=res.traceback,
                status_code=None,  # probably
            )
        elif response_state == "RETRY":
            # Backwards compatibility, otherwise we'd need to add "RETRY" to the clients
            response_state = "PENDING"

        try:
            task_status = TaskStatus(response_state)
            return GetAsyncTaskV1Response(task_id=task_id, status=task_status)
        except ValueError:
            logger.info(f"Task {task_id} has an unknown state: <{response_state}> ")
            return GetAsyncTaskV1Response(task_id=task_id, status=TaskStatus.UNDEFINED)
