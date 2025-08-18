import logging
import os
from enum import IntEnum, unique
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import aioredis
import celery
from celery import Celery
from celery.app import backends
from celery.app.control import Inspect
from celery.result import AsyncResult
from model_engine_server.core.aws.roles import session
from model_engine_server.core.aws.secrets import get_key_file
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import (
    CustomJSONFormatter,
    logger_name,
    make_logger,
    silence_chatty_logger,
)
from redis import Redis, StrictRedis

logger = make_logger(logger_name())

# Make sure to override the 's3' backend alias with our own S3Backend class.
# This is because the Celery code does not actually work when you try and
# override the backend with a class instead of a URL, despite the fact
# that the `backend` constructor arg type is a Union[str, Type[celery.backends.base.Backend]]
backends.BACKEND_ALIASES["s3"] = "model_engine_server.core.celery.s3:S3Backend"
backends.BACKEND_ALIASES["azureblockblob"] = (
    "model_engine_server.core.celery.abs:AzureBlockBlobBackend"
)


DEFAULT_TASK_VISIBILITY_SECONDS = 86400


@unique
class TaskVisibility(IntEnum):
    """
    This defines whether the visibility timeout will be 1 minute, 10 minutes or 1 hour. Visibility timeout is the
    number of seconds to wait for the worker to acknowledge the task before the message is redelivered to another
    worker. This configuration is specific to Redis. Since Celery keeps all unacknowledged tasks in a single collection
    regardless of queue or app deployment, the visibility timeout is global for the all apps using the same Redis.
    For example, if you had a deployment X that runs for 10s on average and deployment Y that runs for 20min on average,
    you can't specify visibility timeout for X and Y separately. As a consequence if you set the visibility timeout
    to 30min, requests to X that are stuck in the unacknowledged state will have to wait for 30min to be put back to
    queue, even though you know within 20s whether or not they succeeded. If you set the visibility timeout to 20s,
    requests to Y will be constantly retried every 20s even though they're still being processed the first time.
    The workaround for this is to use different db indexes for Redis, there are 15 by default,
    we discretized to 3 types of visibility timeouts - 1 min, 10 minutes and the default 1 hour.
    If you use the default TaskVisibility.VISIBILITY_1H, you don't have to change anything else.

    Let's say you want to use some other value for task_visibility, e.g. TaskVisibility.VISIBILITY_1M.
    You'll have to keep a few things in mind:

    1. When defining your Kubernetes deployment, you'll want to specify this index in the annotations, e.g.:

        metadata:
          name: "${DEPLOYMENT_NAME}"
          annotations:
            celery.scaleml.autoscaler/queue: "${QUEUE}"
            celery.scaleml.autoscaler/taskVisibility: "VISIBILITY_1M"  # <- THIS!
            celery.scaleml.autoscaler/perWorker: "${PER_WORKER}"
            celery.scaleml.autoscaler/minWorkers: "${MIN_WORKERS}"
            celery.scaleml.autoscaler/maxWorkers: "${MAX_WORKERS}"

    2. When making requests to such deployment, you'll have to do:
        ```python
        from model_engine_server.core.celery.app import TaskVisibility, celery_app
        app = celery_app(None, task_visibility=TaskVisibility.VISIBILITY_1M)
        future_result = app.send_task("some.task.name", args=["some", "args"], queue="some-queue")
        ```

    Read more about Redis indexes: https://www.mikeperham.com/2015/09/24/storing-data-with-redis/#databases
    Read more here: https://docs.celeryproject.org/en/stable/getting-started/brokers/redis.html#visibility-timeout
    """

    VISIBILITY_1H = 0
    VISIBILITY_10M = 1
    VISIBILITY_1M = 2
    VISIBILITY_24H = 3

    # Note: 15 is reserved for storing data on k8s deployments;
    # we should not use it to store celery tasks
    # although the chances of a key collision aren't very high

    @staticmethod
    def get_visibility_timeout_in_seconds(value: "TaskVisibility") -> int:
        if value == TaskVisibility.VISIBILITY_1M:
            return 60
        if value == TaskVisibility.VISIBILITY_10M:
            return 600
        if value == TaskVisibility.VISIBILITY_1H:
            return 3600
        if value == TaskVisibility.VISIBILITY_24H:
            return 86400
        raise ValueError(f"Unknown value {value}")

    @staticmethod
    def seconds_to_visibility(timeout: int) -> "TaskVisibility":
        """The equivalent TaskVisibility for the given timeout, in seconds.

        Raises ValueError if no such visibility exists.
        """
        if timeout == 60:
            return TaskVisibility.VISIBILITY_1M
        if timeout == 600:
            return TaskVisibility.VISIBILITY_10M
        if timeout == 3600:
            return TaskVisibility.VISIBILITY_1H
        if timeout == 86400:
            return TaskVisibility.VISIBILITY_24H
        raise ValueError(f"Unsupported timeout for TaskVisibility: {timeout}s")

    @staticmethod
    def from_name(name: str) -> "TaskVisibility":
        # pylint: disable=no-member,protected-access
        lookup = {
            x.name: x.value for x in TaskVisibility._value2member_map_.values()
        }  # type: ignore
        return TaskVisibility(lookup[name.upper()])


def silence_chatty_celery_loggers() -> None:
    """Drastically reduces the log activity of the Celery gossip protocol and event state loggers.

    Specifically, sets the `celery.worker.consumer.gossip`, `celery.worker.control` and `celery.events.state`
    loggers to the FATAL level.
    """
    silence_chatty_logger(
        "celery.worker.consumer.gossip",
        "celery.worker.control",
        "celery.events.state",
        quieter=logging.FATAL,
    )


def create_celery_logger_handler(celery_logger, propagate):
    celery_logger.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(CustomJSONFormatter())
    celery_logger.addHandler(stream_handler)
    celery_logger.propagate = propagate
    silence_chatty_celery_loggers()


# pylint: disable=redefined-outer-name,unused-argument
@celery.signals.after_setup_task_logger.connect
def after_setup_celery_task_logger(logger, **kwargs):
    """This function sets the 'celery.task' logger handler and formatter"""
    create_celery_logger_handler(logger, True)


# pylint: disable=redefined-outer-name,unused-argument
@celery.signals.after_setup_logger.connect
def after_setup_celery_logger(logger, **kwargs):
    """This function sets the 'celery' logger handler and formatter"""
    create_celery_logger_handler(logger, False)


def get_all_db_indexes() -> Iterable[int]:
    """
    All the different values in the TaskVisibility enum class define different Redis db numbers / indexes. This fn
    will return an iterable of all different values so that e.g. autoscaler knows what dbs to check for.
    :return: a tuple of db indexes, e.g. (0, 1, 2)
    corresponding to values for (VISIBILITY_1H, VISIBILITY_10M, VISIBILITY_1M, VISIBILITY_24H)
    """
    # pylint: disable=no-member,protected-access
    return tuple(TaskVisibility._value2member_map_.keys())  # type: ignore


def get_redis_host_port():
    host, port = None, None
    if os.getenv("REDIS_HOST") and os.getenv("REDIS_PORT"):
        host = os.getenv("REDIS_HOST")
        port = os.getenv("REDIS_PORT")
    # In the case of k8s, pick the right endpoint based on the config
    elif os.getenv("KUBERNETES_SERVICE_HOST"):
        host = infra_config().redis_host
        port = 6379
    # For debugging purposes
    elif os.getenv("USE_REDIS_LOCALHOST") == "1":
        logger.info("Using localhost as Redis host")
        host = "127.0.0.1"
        port = 6379
    # In the case of local testing, pick the right endpoint based on the config
    elif os.getenv("KUBECONFIG"):
        logger.info(f"Inferring redis host from config env: {infra_config().env}")
        host = f"redis-elasticache-message-broker.{infra_config().dns_host_domain}"
        port = 6379

    logger.info(f"Using Redis host and port: {host}:{port}")
    return host, port


def get_redis_endpoint(db_index: int = 0) -> str:
    if infra_config().redis_aws_secret_name is not None:
        logger.info("Using infra_config().redis_aws_secret_name for Redis endpoint")
        creds = get_key_file(infra_config().redis_aws_secret_name)  # Use default role
        scheme = creds.get("scheme", "redis://")
        host = creds["host"]
        port = creds["port"]
        query_params = creds.get("query_params", "")
        auth_token = creds.get("auth_token", None)
        if auth_token is not None:
            return f"{scheme}:{auth_token}@{host}:{port}/{db_index}{query_params}"
        return f"{scheme}{host}:{port}/{db_index}{query_params}"
    host, port = get_redis_host_port()
    auth_token = os.getenv("REDIS_AUTH_TOKEN")
    if auth_token:
        return f"rediss://:{auth_token}@{host}:{port}/{db_index}?ssl_cert_reqs=none"
    return f"redis://{host}:{port}/{db_index}"


def get_redis_instance(db_index: int = 0) -> Union[Redis, StrictRedis]:
    host, port = get_redis_host_port()
    auth_token = os.getenv("REDIS_AUTH_TOKEN")

    if auth_token:
        return StrictRedis(
            host=host,
            port=port,
            db=db_index,
            password=auth_token,
            ssl=True,
            ssl_cert_reqs="none",
        )
    return Redis(host=host, port=port, db=db_index)


async def get_async_redis_instance(db_index: int = 0) -> aioredis.client.Redis:
    host, port = get_redis_host_port()
    return await aioredis.client.Redis.from_url(f"redis://{host}:{port}/{db_index}")


def celery_app(
    name: Optional[str],
    modules: List[str] = None,
    task_acks_late: bool = True,
    task_reject_on_worker_lost: bool = True,
    task_track_started: bool = False,
    task_visibility: Union[TaskVisibility, int] = TaskVisibility.VISIBILITY_1H,
    task_time_limit: Optional[float] = None,
    task_soft_time_limit: Optional[float] = None,
    task_remote_tracebacks: bool = True,
    worker_prefetch_multiplier: int = 1,
    result_serializer: str = "json",
    result_compression: Optional[str] = None,
    accept_content: Iterable[str] = ("json",),
    s3_bucket: Optional[str] = os.environ.get("S3_BUCKET"),
    s3_base_path: str = "tmp/celery/",
    backend_protocol: str = "s3",
    broker_type: str = "redis",
    aws_role: Optional[str] = None,
    broker_transport_options: Optional[Dict[str, Any]] = None,
    **extra_changes,
) -> Celery:
    """
    :param name: Name of the Celery app. This can be None if you only want a generic app to send a task on an
    existing deployment.

    :param modules: List of modules to import in order to properly register all tasks of a module. If set to None,
    this assumes you're decorating all tasks in the same file that declares the Celery app.

    :param task_acks_late: [optional] flag whether to acknowledge tasks AFTER they were executed. Do not change the
    default unless you're confident you know what you're doing. This is similar to setting to at least once delivery
    (when set to True) versus at most once delivery (when set to False), however has more nuance to it.
    Defaults to True.
    Read more here: https://docs.celeryproject.org/en/stable/userguide/configuration.html#std-setting-task_acks_late

    :param task_reject_on_worker_lost: [optional] flag whether to allow a message to be re-queued when the worker
    process executing it abruptly exits or is signaled (e.g., KILL/INT, etc). Defaults to True.
    Warning: if task_acks_late and task_reject_on_worker_lost are both True, we may end up in a state where tasks
    that cause the worker to crash get reprocessed infinitely.
    Read more here: https://docs.celeryproject.org/en/stable/userguide/configuration.html#task-reject-on-worker-lost

    :param task_track_started: [optional] flag whether to report its status as ‘started’ when the task is executed
    by a worker. Having a ‘started’ state can be useful for when there are long running tasks and there’s a need to
    report what task is currently running. Defaults to False.
    Read more here: https://docs.celeryproject.org/en/stable/userguide/configuration.html#task-track-started

    :param task_visibility: [optional] Enum type TaskVisibility that defines whether the visibility timeout will be
    1 minute, 10 minutes or 1 hour. Visibility timeout is the number of seconds to wait for the worker to
    acknowledge the task before the message is redelivered to another worker. This configuration is specific to Redis.
    Since Celery keeps all unacknowledged tasks in a single collection regardless of queue or app deployment,
    the visibility timeout is global for the all apps using the same Redis. For example, if you had a deployment X that
    runs for 10s on average and deployment Y that runs for 20min on average, you can't specify visibility timeout for
    X and Y separately. As a consequence if you set the visibility timeout to 30min, requests to X that are stuck in
    the unacknowledged state will have to wait for 30min to be put back to queue, even though you know within 20s
    whether or not they succeeded. If you set the visibility timeout to 20s, requests to Y will be constantly retried
    every 20s even though they're still being processed the first time. The workaround for this is to use different
    db indexes for Redis, there are 15 by default, we discretized to 3 types of visibility timeouts - 1 min, 10 minutes
    and the default 1 hour. If you use the default TaskVisibility.VISIBILITY_1H, you don't have to change anything else.

    Let's say you want to use some other value for task_visibility, e.g. TaskVisibility.VISIBILITY_1M.
    You'll have to keep a few things in mind:

    1. When defining your Kubernetes deployment, you'll want to specify this index in the annotations, e.g.:

        metadata:
          name: "${DEPLOYMENT_NAME}"
          annotations:
            celery.scaleml.autoscaler/queue: "${QUEUE}"
            celery.scaleml.autoscaler/taskVisibility: "VISIBILITY_1M"  # <- THIS!
            celery.scaleml.autoscaler/perWorker: "${PER_WORKER}"
            celery.scaleml.autoscaler/minWorkers: "${MIN_WORKERS}"
            celery.scaleml.autoscaler/maxWorkers: "${MAX_WORKERS}"

    2. When making requests to such deployment, you'll have to do:
        ```python
        from model_engine_server.core.celery import TaskVisibility, celery_app
        app = celery_app(None, task_visibility=TaskVisibility.VISIBILITY_1M)
        future_result = app.send_task("some.task.name", args=["some", "args"], queue="some-queue")
        ```

    Defaults to 3600s (1 hour).
    Read more about Redis indexes: https://www.mikeperham.com/2015/09/24/storing-data-with-redis/#databases
    Read more here: https://docs.celeryproject.org/en/stable/getting-started/brokers/redis.html#visibility-timeout

    :param task_time_limit: [optional] app-level settings, task hard time limit in seconds.
    The worker processing the task will be killed and replaced with a new one when this is exceeded.
    Defaults to None. This can be set on the task-level as:
    * @app.task(time_limit=20)
    * mytask.apply_async(args=[], kwargs={}, time_limit=30)
    Read more here: https://docs.celeryproject.org/en/stable/userguide/configuration.html#task-time-limit

    :param task_soft_time_limit: [optional] app-level settings, task soft time limit in seconds.
    Limit to after which celery.exceptions.SoftTimeLimitExceeded is thrown to log, clean up etc.
    Defaults to None. This can be set on the task-level as:
    * @app.task(soft_time_limit=20)
    * mytask.apply_async(args=[], kwargs={}, soft_time_limit=30)
    Read more here: https://docs.celeryproject.org/en/stable/userguide/configuration.html#task-soft-time-limit

    :param task_remote_tracebacks: [optional] flag whether to return worker's stack trace in case of a task exception
    (i.e. what threw the error) as opposed to client stack trace (i.e. where the task was initiated).
    Defaults to True.
    Read more here: https://docs.celeryproject.org/en/stable/userguide/configuration.html#task-remote-tracebacks

    :param worker_prefetch_multiplier: [optional] the number of tasks (messages) a worker can reserve for itself.
    Reserving tasks means they will be popped from the Redis queue. Setting this to a higher number potentially
    increases a risk of tasks being lost in case of a worker's hard crash. The benefit is that by prefetching more
    tasks, the worker is able to execute the tasks slightly faster. This number is multiplied by the concurrency
    argument of the worker, e.g. if concurrency is set to 4 and worker_prefetch_multiplier is set to 2, the worker will
    prefetch 8 tasks. Behaves un-intuitively when set to 0. Defaults to 1.
    Read more here: https://docs.celeryproject.org/en/stable/userguide/optimizing.html#prefetch-limits

    :param result_serializer: Task result serialization format. Should be one of: {"json", "pickle", "yaml", "msgpack"}.
    Defaults to "json".
    Visit https://docs.celeryproject.org/en/stable/userguide/calling.html#calling-serializers for more info.

    :param result_compression: [optional] Compression method used for task result. Defaults to None.
    Visit https://docs.celeryproject.org/en/stable/userguide/calling.html#compression to explore the options.

    :param accept_content: [optional] A white-list of content-types/serializers to allow.
    Defaults to ("json",). Any content type can be added, including pickle, yaml or msgpack.
    Read more here: https://docs.celeryproject.org/en/stable/userguide/configuration.html#accept-content
    # FIXME: Celery doesn't like when you run workers as root (which we are since Docker) and use pickle as a
    # FIXME: serializer. Until we figure out how to run as a non-root user, it might be better to avoid pickle.

    :param s3_bucket: [optional] Bucket name to store task results when using S3 as backend. The results uri will be
    "s3://<s3_bucket>/<s3_base_path>/...".

    :param s3_base_path: [optional] Base path for task results when using S3 as backend. The results uri will be
    "s3://<s3_bucket>/<s3_base_path>/...".

    :param backend_protocol: [optional] Backend protocol to use, currently supports "s3", "redis", and "abs".
    Defaults to "s3". Redis might be faster than S3 but is not persistent, so using "redis" is discouraged.
    If you do end up using this, make sure you set up `result_expires`
    (https://docs.celeryproject.org/en/stable/userguide/configuration.html#result-expires) to something reasonable
    (1 day by default) and run `celery beat` periodically to clear expired results from Redis. Visit
    https://docs.celeryproject.org/en/stable/userguide/periodic-tasks.html to learn more about celery beat

    :param broker_type: [defaults to "redis"] The broker type. We currently support "redis", "sqs", and "servicebus".

    :param aws_role: [optional] AWS role to use.

    :param extra_changes: Extra keyword arguments to Celery app.
    Visit https://docs.celeryproject.org/en/stable/userguide/configuration.html to see options.

    :return: Celery app
    """

    assert result_serializer in [
        "json",
        "pickle",
        "yaml",
        "msgpack",
    ], 'Serializer must be one of "{"json", "pickle", "yaml", "msgpack"}"'
    assert (
        result_serializer in accept_content
    ), f'Serializer {result_serializer} must be in "accept_content" {accept_content}'
    assert worker_prefetch_multiplier >= 0, '"worker_prefetch_multiplier" must be non-negative.'

    if isinstance(task_visibility, int):
        task_visibility = TaskVisibility(task_visibility)

    visibility_timeout = TaskVisibility.get_visibility_timeout_in_seconds(task_visibility)
    assert task_time_limit is None or task_time_limit <= visibility_timeout, (
        f'"task_time_limit" {task_time_limit} must be less than or equal to visibility_timeout {visibility_timeout}.'
        f'To change visibility timeout, change the "task_visibility" argument (1 hour by default).'
    )

    if not task_acks_late:
        logger.warning(
            'Setting "task_acks_late" to False is discouraged. Workers will acknowledge task messages '
            "before executing them, risking that messages will be lost in case of worker crash."
        )

    if worker_prefetch_multiplier == 0:
        logger.warning(
            'Setting "worker_prefetch_multiplier" to 0 is discouraged. The worker will keep consuming '
            "messages, not respecting that there may be other available worker nodes that may be able to "
            "process them sooner, or that the messages may not even fit in memory."
        )

    if task_time_limit is None:
        task_time_limit = visibility_timeout

    if task_soft_time_limit is not None and task_soft_time_limit >= task_time_limit:
        logger.warning(
            f'Setting "task_soft_time_limit" ({task_soft_time_limit}s) to a value greater than or equal '
            f'than "task_time_limit" ({task_time_limit}s) will have no effect. If you haven\'t specified '
            f'"task_time_limit" yourself, the value is set based on task_visibility (1 hour by default).'
        )

    logger.info(f"Using broker type {broker_type}")
    broker_transport_options = broker_transport_options or {}
    broker, broker_transport_options = _get_broker_endpoint_and_transport_options(
        broker_type, task_visibility.value, visibility_timeout, broker_transport_options
    )

    # See https://docs.celeryproject.org/en/stable/userguide/configuration.html
    conf_changes = {
        "task_acks_late": task_acks_late,
        "task_reject_on_worker_lost": task_reject_on_worker_lost,
        "task_track_started": task_track_started,
        "task_time_limit": task_time_limit,
        "task_soft_time_limit": task_soft_time_limit,
        "task_remote_tracebacks": task_remote_tracebacks,
        "worker_prefetch_multiplier": worker_prefetch_multiplier,
        "result_serializer": result_serializer,
        "result_compression": result_compression,
        "accept_content": accept_content,
        "broker_transport_options": broker_transport_options,
        **extra_changes,
    }

    if s3_bucket is None:
        s3_bucket = infra_config().s3_bucket

    backend_url, extra_conf_changes = _get_backend_url_and_conf(
        backend_protocol,
        s3_bucket=s3_bucket,
        s3_base_path=s3_base_path,
        aws_role=aws_role,
    )

    conf_changes.update(extra_conf_changes)

    return Celery(
        name,
        include=modules,
        broker=broker,
        backend=backend_url,
        changes=conf_changes,
    )


def _get_broker_endpoint_and_transport_options(
    broker_type: str,
    task_visibility: int,
    visibility_timeout: int,
    broker_transport_options: Dict[str, Any],
) -> Tuple[str, Dict[str, str]]:
    """
    Helper function for getting a broker endpoint and a broker_transport_options dict.
    """
    out_broker_transport_options = broker_transport_options.copy()
    out_broker_transport_options["visibility_timeout"] = visibility_timeout

    if broker_type == "redis":
        # https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/redis.html
        return get_redis_endpoint(task_visibility), out_broker_transport_options
    if broker_type == "sqs":
        # https://docs.celeryq.dev/en/stable/getting-started/backends-and-brokers/sqs.html

        # If we need more config values passed in (e.g. polling_interval, wait_time_seconds, queue_prefix,
        # backoff_policy, etc., then we can expose broker_transport_options in the top-level celery() wrapper function.
        # Going to try this with defaults first.
        out_broker_transport_options["region"] = os.environ.get("AWS_REGION", "us-west-2")

        # changing wait_time_seconds from the default of 10 based on https://github.com/celery/celery/discussions/7283
        # goal is to prevent async requests from being stuck in pending when workers die; the hypothesis is that this is caused by SQS long polling
        out_broker_transport_options["wait_time_seconds"] = 0
        out_broker_transport_options["polling_interval"] = 5

        # NOTE: The endpoints should ideally use predefined queues. However, the sender probably needs the flexibility
        # of not requiring predefined queues.
        # assert (
        #    "predefined_queues" in out_broker_transport_options
        # ), "Need to pass in predefined_queues for SQS"

        # Plain "sqs://" signifies to use instance metadata.
        return "sqs://", out_broker_transport_options
    if broker_type == "servicebus":
        return (
            f"azureservicebus://DefaultAzureCredential@{os.getenv('SERVICEBUS_NAMESPACE')}.servicebus.windows.net",
            out_broker_transport_options,
        )

    raise ValueError(
        f"Only 'redis', 'sqs', and 'servicebus' are supported values for broker_type, got value {broker_type}"
    )


def _get_backend_url_and_conf(
    backend_protocol: str,
    s3_bucket: str,
    s3_base_path: str,
    aws_role: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Helper function for getting a backend URL and an updated conf dictionary. The returned conf dictionary is a copy
    of the in_conf_changes.
    """
    out_conf_changes: Dict[str, Any] = {}
    if backend_protocol == "redis":
        # use db_num=1 for backend to differentiate from broker
        backend_url = get_redis_endpoint(1)
    elif backend_protocol == "s3":
        # Check if AWS is disabled - if so, fall back to Redis backend
        if os.environ.get('DISABLE_AWS') == 'true':
            logger.warning("AWS disabled - falling back to Redis backend instead of S3")
            backend_url = get_redis_endpoint(1)
        else:
            backend_url = "s3://"
            if aws_role is None:
                aws_session = session(infra_config().profile_ml_worker)
            else:
                aws_session = session(aws_role)
            
            # If AWS is disabled, session will be None - fall back to Redis
            if aws_session is None:
                logger.warning("AWS session is None - falling back to Redis backend")
                backend_url = get_redis_endpoint(1)
            else:
                out_conf_changes.update(
                    {
                        "s3_boto3_session": aws_session,
                        "s3_bucket": s3_bucket,
                        "s3_base_path": s3_base_path,
                    }
                )
    elif backend_protocol == "abs":
        backend_url = f"azureblockblob://{os.getenv('ABS_ACCOUNT_NAME')}"
    else:
        raise ValueError(
            f'Unknown backend protocol "{backend_protocol}". Should be one of ["s3", "redis", "abs].'
        )

    return backend_url, out_conf_changes


def inspect_app(app: Optional[Celery] = None, **kwargs) -> Inspect:
    """
    Helper function to obtain Celery's Inspect object to monitor:
    * active tasks currently being executed
    * reserved tasks currently being reserved (claimed) by workers but waiting for execution
    * revoked tasks history
    * a few others

    Read https://docs.celeryproject.org/en/stable/userguide/monitoring.html#introduction for more info.
    """
    if not app:
        # noinspection PyTypeChecker
        app = celery_app(name=kwargs.pop("name", None), **kwargs)  # type: ignore
    return app.control.inspect()


def get_async_result(task_id: str, app: Optional[Celery] = None, **kwargs) -> AsyncResult:
    """
    Helper function to obtain Celery's AsyncResult based on a task id. You can submit tasks in one process,
    store the task ids and later check the status or collect the result in a different process
    """
    if not app:
        # noinspection PyTypeChecker
        app = celery_app(name=kwargs.pop("name", None), **kwargs)  # type: ignore
    return AsyncResult(task_id, app=app)


def get_num_unclaimed_tasks(queue_name: str, redis_instance: Optional[Redis] = None) -> int:
    _redis_instance = redis_instance if redis_instance is not None else get_redis_instance()
    num_unclaimed = _redis_instance.llen(queue_name)
    if redis_instance is None:
        _redis_instance.close()  # type: ignore
    return num_unclaimed


async def get_num_unclaimed_tasks_async(
    queue_name: str, redis_instance: Optional[aioredis.client.Redis] = None
) -> int:
    _redis_instance = redis_instance if redis_instance is not None else get_async_redis_instance()
    num_unclaimed = await _redis_instance.llen(queue_name)  # type: ignore
    if redis_instance is None:
        await _redis_instance.close()  # type: ignore
    return num_unclaimed
