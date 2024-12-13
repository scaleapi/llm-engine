import argparse
import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, TypedDict, Union

from celery import Celery, Task, states
from model_engine_server.common.constants import DEFAULT_CELERY_TASK_NAME, LIRA_CELERY_TASK_NAME
from model_engine_server.common.dtos.model_endpoints import BrokerType
from model_engine_server.common.dtos.tasks import EndpointPredictV1Request
from model_engine_server.core.celery import (
    DEFAULT_TASK_VISIBILITY_SECONDS,
    TaskVisibility,
    celery_app,
)
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.core.utils.format import format_stacktrace
from model_engine_server.inference.forwarding.forwarding import (
    Forwarder,
    LoadForwarder,
    load_named_config,
)
from model_engine_server.inference.infra.gateways.datadog_inference_monitoring_metrics_gateway import (
    DatadogInferenceMonitoringMetricsGateway,
)
from requests import ConnectionError

logger = make_logger(logger_name())


class ErrorResponse(TypedDict):
    """The response payload for any inference request that encountered an error."""

    error: str
    error_metadata: str


def raw_celery_response(backend, task_id: str) -> Dict[str, Any]:
    key_info: str = backend.get_key_for_task(task_id)
    info_as_str: str = backend.get(key_info)
    info: dict = json.loads(info_as_str)
    return info


def error_response(msg: str, e_unhandled: Exception) -> ErrorResponse:
    stacktrace = format_stacktrace(e_unhandled)
    return {
        "error": str(e_unhandled),
        "error_metadata": f"{msg}\n{stacktrace}",
    }


def create_celery_service(
    forwarder: Forwarder,
    task_visibility: TaskVisibility,
    broker_type: str,
    backend_protocol: str,
    queue_name: Optional[str] = None,
    sqs_url: Optional[str] = None,
) -> Celery:
    """
    Creates a celery application.
    Returns:
        app (celery.app.base.Celery): Celery app.
        exec_func (celery.local.PromiseProxy): Callable task function.
    """

    app: Celery = celery_app(
        name=None,
        s3_bucket=infra_config().s3_bucket,
        aws_role=infra_config().profile_ml_inference_worker,
        task_visibility=task_visibility,
        broker_type=broker_type,
        broker_transport_options=(
            {"predefined_queues": {queue_name: {"url": sqs_url}}}
            if broker_type == str(BrokerType.SQS.value)
            else None
        ),
        backend_protocol=backend_protocol,
    )

    monitoring_metrics_gateway = DatadogInferenceMonitoringMetricsGateway()

    class ErrorHandlingTask(Task):
        """Sets a 'custom' field with error in the Task response for FAILURE.

        Used when services are ran via the Celery backend.
        """

        def after_return(
            self, status: str, retval: Union[dict, Exception], task_id: str, args, kwargs, einfo
        ) -> None:
            """Handler that ensures custom error response information is available whenever a Task fails.

            Specifically, whenever the task's :param:`status` is `"FAILURE"` and the return value
            :param:`retval` is an `Exception`, this handler extracts information from the `Exception`
            and constructs a custom error response JSON value (see :func:`error_response` for details).

            This handler then re-propagates the Celery-required exception information (`"exc_type"` and
            `"exc_message"`) while adding this new error response information under the `"custom"` key.
            """
            if status == states.FAILURE and isinstance(retval, Exception):
                logger.warning(f"Setting custom error response for failed task {task_id}")

                info: dict = raw_celery_response(self.backend, task_id)
                result: dict = info["result"]
                err: Exception = retval

                error_payload = error_response("Internal failure", err)

                # Inspired by pattern from:
                # https://www.distributedpython.com/2018/09/28/celery-task-states/
                self.update_state(
                    state=states.FAILURE,
                    meta={
                        "exc_type": result["exc_type"],
                        "exc_message": result["exc_message"],
                        "custom": json.dumps(error_payload, indent=False),
                    },
                )
            request_params = args[0]
            request_params_pydantic = EndpointPredictV1Request.parse_obj(request_params)
            if forwarder.post_inference_hooks_handler:
                forwarder.post_inference_hooks_handler.handle(request_params_pydantic, retval, task_id)  # type: ignore

    # See documentation for options:
    # https://docs.celeryproject.org/en/stable/userguide/tasks.html#list-of-options
    # We autoretry on requests.ConnectionError to handle the case where the main container
    # shuts down because the pod scales down. This kicks the task back to the queue and
    # allows a new worker to pick it up.
    @app.task(
        base=ErrorHandlingTask,
        name=LIRA_CELERY_TASK_NAME,
        track_started=True,
        autoretry_for=(ConnectionError,),
    )
    def exec_func(payload, arrival_timestamp, *ignored_args, **ignored_kwargs):
        if len(ignored_args) > 0:
            logger.warning(f"Ignoring {len(ignored_args)} positional arguments: {ignored_args=}")
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignoring {len(ignored_kwargs)} keyword arguments: {ignored_kwargs=}")
        try:
            monitoring_metrics_gateway.emit_async_task_received_metric(queue_name)
            result = forwarder(payload)
            request_duration = datetime.now() - arrival_timestamp
            if request_duration > timedelta(seconds=DEFAULT_TASK_VISIBILITY_SECONDS):
                monitoring_metrics_gateway.emit_async_task_stuck_metric(queue_name)
            return result
        except Exception:
            logger.exception("Celery service failed to respond to request.")
            raise

    # Have celery service also accept pre-LIRA celery task name to ensure no downtime
    # when transitioning from pre-LIRA single container architecture to LIRA
    # multi-container-architecture.
    @app.task(
        base=ErrorHandlingTask,
        name=DEFAULT_CELERY_TASK_NAME,
        track_started=True,
    )
    def exec_func_pre_lira(payload, arrival_timestamp, *ignored_args, **ignored_kwargs):
        return exec_func(payload, arrival_timestamp, *ignored_args, **ignored_kwargs)

    return app


def start_celery_service(
    app: Celery,
    queue: str,
    concurrency: int,
) -> None:
    worker = app.Worker(
        queues=[queue],
        concurrency=concurrency,
        loglevel="INFO",
        optimization="fair",
        # Don't use pool="solo" so we can send multiple concurrent requests over
        # Historically, pool="solo" argument fixes the known issues of celery and some of the libraries.
        # Particularly asyncio and torchvision transformers. This isn't relevant since celery-forwarder
        # is quite lightweight
        # TODO: we should probably use eventlet or gevent for the pool, since
        # the forwarder is nearly the most extreme example of IO bound.
    )
    worker.start()


def entrypoint():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--set", type=str, action="append")
    parser.add_argument("--task-visibility", type=str, required=True)
    parser.add_argument(
        "--num-workers",
        type=int,
        required=True,
        help="Defines number of concurrent requests to work on",
    )
    parser.add_argument("--broker-type", type=str, default=None)
    parser.add_argument("--backend-protocol", type=str, default="s3")
    parser.add_argument("--queue", type=str, required=True)
    parser.add_argument("--sqs-url", type=str, default=None)

    args = parser.parse_args()

    if args.broker_type is None:
        args.broker_type = str(BrokerType.SQS.value if args.sqs_url else BrokerType.REDIS.value)

    forwarder_config = load_named_config(args.config, args.set)
    forwarder_loader = LoadForwarder(**forwarder_config["async"])
    forwader = forwarder_loader.load(None, None)

    app = create_celery_service(
        forwader,
        TaskVisibility.VISIBILITY_24H,
        args.broker_type,
        args.backend_protocol,
        args.queue,
        args.sqs_url,
    )
    start_celery_service(app, args.queue, args.num_workers)


if __name__ == "__main__":
    entrypoint()
