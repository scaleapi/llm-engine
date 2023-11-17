import asyncio as aio
import dataclasses
import hashlib
import logging
import os
import time
from abc import ABC, abstractmethod
from bisect import bisect
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from math import ceil
from typing import Any, DefaultDict, Dict, List, Set, Tuple

import aioredis
import stringcase
from celery.app.control import Inspect
from datadog import statsd
from kubernetes_asyncio import client
from kubernetes_asyncio import config as kube_config
from kubernetes_asyncio.client.rest import ApiException
from kubernetes_asyncio.config.config_exception import ConfigException
from model_engine_server.core.aws.roles import session
from model_engine_server.core.celery import (
    TaskVisibility,
    celery_app,
    get_all_db_indexes,
    get_redis_host_port,
    inspect_app,
)
from model_engine_server.core.loggers import logger_name, make_logger


def excluded_namespaces():
    try:
        from plugins.celery_autoscaler_dependencies import CELERY_AUTOSCALER_EXCLUDED_NAMESPACES

        return CELERY_AUTOSCALER_EXCLUDED_NAMESPACES
    except ModuleNotFoundError:
        return []


ELASTICACHE_REDIS_BROKER = "redis-elasticache-message-broker-master"
SQS_BROKER = "sqs-message-broker-master"

UPDATE_DEPLOYMENT_MAX_RETRIES = 10

SQS_SAMPLE_COUNT = 10

logger = make_logger(logger_name())

autoscaler_broker = os.environ.get("BROKER_NAME", SQS_BROKER)
aws_profile = os.environ.get("AWS_PROFILE")


@dataclasses.dataclass
class CeleryAutoscalerParams:
    queue: str
    broker: str = SQS_BROKER
    task_visibility: TaskVisibility = TaskVisibility.VISIBILITY_1H
    per_worker: int = 1
    min_workers: int = 0
    max_workers: int = 1


def _hash_any_to_int(data: Any):
    return int(hashlib.md5(str(data).encode()).hexdigest(), 16)


async def list_deployments(core_api, apps_api) -> Dict[Tuple[str, str], CeleryAutoscalerParams]:
    namespaces = await core_api.list_namespace()
    celery_deployments_params = {}
    for namespace in namespaces.items:
        namespace_name = namespace.metadata.name
        if namespace_name in excluded_namespaces():
            continue
        namespace_start_time = time.time()
        deployments = await apps_api.list_namespaced_deployment(namespace=namespace_name)
        logger.info(
            f"list_namespaced_deployment with {namespace_name} took {time.time() - namespace_start_time} seconds"
        )
        for deployment in deployments.items:
            deployment_name = deployment.metadata.name
            annotations = deployment.metadata.annotations

            if not annotations:
                continue

            # Parse parameters
            params = {}

            if "celery.scaleml.autoscaler/broker" in annotations:
                deployment_broker = annotations["celery.scaleml.autoscaler/broker"]
            else:
                deployment_broker = ELASTICACHE_REDIS_BROKER

            if deployment_broker != autoscaler_broker:
                logger.debug(
                    f"Skipping deployment {deployment_name}; deployment's broker {deployment_broker} is not {autoscaler_broker}"
                )
                continue

            for f in dataclasses.fields(CeleryAutoscalerParams):
                k = f.name
                v = annotations.get(f"celery.scaleml.autoscaler/{stringcase.camelcase(k)}")
                if not v:
                    continue

                try:
                    if k == "task_visibility":
                        v = TaskVisibility.from_name(v)
                    v = f.type(v)
                except (ValueError, KeyError):
                    logger.exception(f"Unable to convert {f.name}: {v} to {f.type}")

                params[k] = v

            try:
                celery_autoscaler_params = CeleryAutoscalerParams(**params)
            except TypeError:
                logger.debug(
                    f"Missing params, skipping deployment : {deployment_name} in {namespace_name}"
                )
                continue

            celery_deployments_params[(deployment_name, namespace_name)] = celery_autoscaler_params

    return celery_deployments_params


class InstanceLogger(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return "%s %s" % (self.extra["name"], msg), kwargs


class Instance:
    def __init__(self, api, name, namespace, params: CeleryAutoscalerParams, env):
        self.api = api
        self.name = name
        self.namespace = namespace
        self.params = params
        self.history: List[Tuple[float, float]] = []
        self.logger = InstanceLogger(logger, {"name": name})
        self.env = env

    async def check_queue_size_and_update_deployment(self, queue_size: int) -> None:
        workers_wanted = ceil(queue_size / self.params.per_worker)

        time_now = time.monotonic()
        self.history.append((workers_wanted, time_now))

        # Take last 10 minutes
        times = [t for _, t in self.history]
        evict = bisect(times, time_now - 600)
        self.history = self.history[evict:]

        workers_wanted = max(self.history)[0]  # type: ignore
        workers_wanted = min(self.params.max_workers, workers_wanted)
        workers_wanted = max(self.params.min_workers, workers_wanted)

        await self.update_deployment(workers_wanted)

    async def update_deployment(self, workers_wanted) -> None:
        for _ in range(UPDATE_DEPLOYMENT_MAX_RETRIES):
            try:
                dep = await self.api.read_namespaced_deployment(
                    name=self.name, namespace=self.namespace
                )

                if dep.spec.replicas == workers_wanted:
                    self.logger.debug("Deployment not updated.")
                    break

                dep.spec.replicas = workers_wanted

                await self.api.patch_namespaced_deployment(
                    name=self.name,
                    namespace=self.namespace,
                    body=dep,
                )

                self.logger.info(f"Deployment updated. replicas={dep.spec.replicas}")
                emit_health_metric("scaling_succeeded", self.env)
                return
            except ApiException as exc:
                if exc.status == 409:
                    self.logger.info("409 retry")
                    continue
                elif exc.status == 404:
                    self.logger.warning("404 not found")
                    return
                emit_health_metric("scaling_failed", self.env)
                raise
        else:
            emit_health_metric("scaling_failed", self.env)
            raise Exception("Ran out of retries updating deployment")


@dataclasses.dataclass
class QueueSizes:
    """Obtained from Inspect.active()"""

    active: int = 0

    """Obtained from Inspect.active()
    """
    reserved: int = 0

    """Computed by summing Redis queue lengths across all db_indexes.
    """
    enqueued: int = 0

    """The sum of all of other fields.
    """
    total: int = 0

    # Ignoring these other Inspect categories for now, since they have a different structure
    # from 'active' and 'reserved'. We can add them later if we want - it'd just require some
    # more complexity to parse them out.
    #
    # scheduled: int = 0
    # revoked: int = 0
    # registered: int = 0


@dataclasses.dataclass
class WorkerMetrics:
    """
    Key: db_index
    Value: number of workers
    """

    worker_counts: DefaultDict[int, int]


@dataclasses.dataclass
class BrokerMetrics:
    """
    Key: (queue_name, db_index)
    Value: QueueSizes
    """

    queue_sizes: DefaultDict[Tuple[str, int], QueueSizes]

    """"
    Represents the number of active redis client connections
    """
    connection_count: int

    """
    Represents the max number of redis client connections allowed
    """
    max_connections: int


@dataclasses.dataclass
class Metrics:
    worker_metrics: WorkerMetrics
    broker_metrics: BrokerMetrics


def emit_metrics(
    metrics: Metrics,
    env: str,
) -> None:
    """
    Emits a given mapping of queue sizes to Datadog.
    """
    queue_sizes = metrics.broker_metrics.queue_sizes
    for q, queue_size in queue_sizes.items():
        queue_name, _ = q
        tags = [
            f"env:{env}",
            f"queue:{queue_name}",
        ]

        for metric_name, metric_value in queue_size.__dict__.items():
            statsd.gauge(f"celery.queue_size.{metric_name}", metric_value, tags=tags)

    # Redis-specific, can be ignored for sqs (worker_counts should be empty anyways)
    for db_index, worker_count in metrics.worker_metrics.worker_counts.items():
        task_visibility = TaskVisibility(db_index).name.lower()
        tags = [
            f"env:{env}",
            f"task_visibility:{task_visibility}",
        ]
        statsd.gauge("celery.worker_count", worker_count, tags=tags)

    if metrics.broker_metrics.connection_count is not None:
        tags = [
            f"env:{env}",
        ]
        statsd.gauge(
            "celery.connection_count",
            metrics.broker_metrics.connection_count,
            tags=tags,
        )

    if metrics.broker_metrics.max_connections is not None:
        tags = [
            f"env:{env}",
        ]
        statsd.gauge("celery.max_connections", metrics.broker_metrics.max_connections, tags=tags)


def emit_health_metric(metric_name: str, env: str):
    tags = [f"env:{env}"]
    statsd.increment(f"celery_autoscaler.{metric_name}", tags=tags)


class AutoscalerBroker(ABC):
    """
    Base class for autoscaler brokers.
    """

    @abstractmethod
    async def get_broker_metrics(
        self,
        queues: Set[Tuple[str, int]],
        queue_sizes: DefaultDict[Tuple[str, int], QueueSizes],
    ) -> BrokerMetrics:
        """
        Calculates broker related metrics.

        Args:
            queues: a set of (queue_name, db_index)
            queue_sizes: number of active and reserved tasks for each queue

        Returns: broker metrics
        """


class RedisBroker(AutoscalerBroker):
    def __init__(self, use_elasticache: bool, initialized: bool = False):
        self.use_elasticache = use_elasticache
        self.initialized = initialized

    async def _init_client(self):
        (
            host,
            port,
        ) = (
            get_redis_host_port()
        )  # Switches the redis instance based on CELERY_ELASTICACHE_ENABLED's value
        self.redis = {
            db_index: aioredis.client.Redis.from_url(f"redis://{host}:{port}/{db_index}")
            for db_index in get_all_db_indexes()
        }
        self.initialized = True

    async def _get_queue_sizes(
        self,
        queues: Set[Tuple[str, int]],
        queue_sizes: DefaultDict[Tuple[str, int], QueueSizes],
    ):
        if not self.initialized:
            await self._init_client()

        for queue_name, db_index in queues:
            q = (queue_name, db_index)
            enqueued = await self.redis[db_index].llen(queue_name)
            queue_sizes[q].enqueued += enqueued
            queue_sizes[q].total += enqueued
        return queue_sizes

    async def _get_connection_count(self):
        redis_client = next(iter(self.redis.values()), None)  # get any redis client

        if redis_client is not None:
            if (
                self.use_elasticache
            ):  # We are using elasticache which doesn't allow us to do `CONFIG GET`
                info = await redis_client.info()
                connection_count = info.get("connected_clients")
                max_connections = info.get("maxclients")
            else:
                (info, config) = await aio.gather(
                    redis_client.info(),
                    redis_client.config_get("maxclients"),
                )
                max_connections = config.get("maxclients")
                connection_count = info.get("connected_clients")

        return connection_count, max_connections

    async def get_broker_metrics(
        self,
        queues: Set[Tuple[str, int]],
        queue_sizes: DefaultDict[Tuple[str, int], QueueSizes],
    ) -> BrokerMetrics:
        queue_sizes = await self._get_queue_sizes(queues, queue_sizes)
        connection_count, max_connections = await self._get_connection_count()
        return BrokerMetrics(
            queue_sizes=queue_sizes,
            connection_count=connection_count,
            max_connections=max_connections,
        )


class SQSBroker(AutoscalerBroker):
    @staticmethod
    def _get_sqs_queue_size(queue_name: str):
        sqs_client = session(aws_profile).client("sqs", region_name="us-west-2")
        try:
            total_start_time = time.time()
            queue_size_hist = []
            reserved_size_hist = []
            # We intentionally launch several requests to the same queue.
            # We have found multiple samples results in more accurate length estimates compared to a single request.
            # Performance-wise: The first request takes ~0.5s, subsequent requests take ~0.005s
            for _ in range(SQS_SAMPLE_COUNT):
                response = sqs_client.get_queue_attributes(
                    QueueUrl=queue_name,
                    AttributeNames=[
                        "ApproximateNumberOfMessages",
                        "ApproximateNumberOfMessagesNotVisible",
                    ],
                )
                queue_size_hist.append(int(response["Attributes"]["ApproximateNumberOfMessages"]))
                reserved_size_hist.append(
                    int(response["Attributes"]["ApproximateNumberOfMessagesNotVisible"])
                )
            total_end_time = time.time()
            queue_size = max(queue_size_hist)
            # SQS's ApproximateNumberOfMessagesNotVisible should correspond to celery's
            #  number of active + number of reserved tasks
            reserved_size = max(reserved_size_hist)
            logger.info(
                f"SQS {queue_name} total: {total_end_time - total_start_time} seconds, queue size {queue_size}, reserved size {reserved_size}"
            )

        except sqs_client.exceptions.QueueDoesNotExist as e:
            logger.info(f"Queue does not exist {queue_name}: {e}")
            queue_size = 0
            reserved_size = 0
        except Exception as e:
            logger.error(f"Failed to get queue attributes {queue_name}: {e}")
            queue_size = 0
            reserved_size = 0
        return queue_size, reserved_size

    def _get_queue_sizes(
        self,
        queues: Set[Tuple[str, int]],
        queue_sizes: DefaultDict[Tuple[str, int], QueueSizes],
    ):
        queue_names = [queue_name for queue_name, _ in queues]
        with ThreadPoolExecutor() as executor:
            results = executor.map(SQSBroker._get_sqs_queue_size, queue_names)

        for q, (enqueued, reserved) in zip(queues, results):
            queue_sizes[q].enqueued += enqueued
            queue_sizes[q].reserved += reserved
            queue_sizes[q].total += enqueued + reserved
        return queue_sizes

    async def get_broker_metrics(
        self,
        queues: Set[Tuple[str, int]],
        queue_sizes: DefaultDict[Tuple[str, int], QueueSizes],
    ) -> BrokerMetrics:
        queue_sizes = self._get_queue_sizes(queues, queue_sizes)
        return BrokerMetrics(
            queue_sizes=queue_sizes,
            connection_count=None,
            max_connections=None,
        )  # connection_count and max_connections are redis-specific metrics


def get_worker_metrics(
    inspect: Dict[int, Inspect],
    queues: Set[Tuple[str, int]],
) -> Tuple[WorkerMetrics, DefaultDict[Tuple[str, int], QueueSizes]]:
    """
    Given a set of Celery Inspect results for each db connection,
    computes the number of workers for each db connection, and number of active and reserved tasks.

    In the case of SQS this will return no data for queue_sizes/worker counts, as inspect is empty
    """
    queue_sizes: DefaultDict[Tuple[str, int], QueueSizes] = defaultdict(QueueSizes)
    worker_counts: DefaultDict[int, int] = defaultdict(int)
    for db_index, insp in inspect.items():
        insp_categories = {
            "active": insp.active(),
            "reserved": insp.reserved(),
        }

        worker_ping = insp.ping()
        if worker_ping:
            worker_counts[db_index] = len(worker_ping.values())

        for insp_key, worker_group in filter(lambda x: x[1], insp_categories.items()):
            for task_list in worker_group.values():
                for task in task_list:
                    queue_name = task["delivery_info"]["routing_key"]
                    q = (queue_name, db_index)

                    if q in queues:
                        queue_sizes[q].__dict__[insp_key] += 1
                        queue_sizes[q].total += 1
    return WorkerMetrics(worker_counts=worker_counts), queue_sizes


async def get_metrics(
    broker: AutoscalerBroker,
    inspect: Dict[int, Inspect],
    queues: Set[Tuple[str, int]],
) -> Metrics:
    """
    Given a set of Redis db connections and Celery Inspect results for each db connection,
    computes worker and broker metrics.
    """

    worker_metrics, active_reserved_queue_sizes = get_worker_metrics(inspect, queues)
    broker_metrics = await broker.get_broker_metrics(queues, active_reserved_queue_sizes)

    return Metrics(
        worker_metrics=worker_metrics,
        broker_metrics=broker_metrics,
    )


async def main():
    instances: Dict[Tuple[str, str], Instance] = {}
    try:
        kube_config.load_incluster_config()
    except ConfigException:
        logger.info("No incluster kubernetes config, falling back to local")
        await kube_config.load_kube_config()

    core_api = client.CoreV1Api()
    apps_api = client.AppsV1Api()

    BROKER_NAME_TO_CLASS = {
        ELASTICACHE_REDIS_BROKER: RedisBroker(use_elasticache=True),
        SQS_BROKER: SQSBroker(),
    }

    broker = BROKER_NAME_TO_CLASS[autoscaler_broker]
    broker_type = "redis" if isinstance(broker, RedisBroker) else "sqs"

    if broker_type == "redis":
        inspect = {
            db_index: inspect_app(
                app=celery_app(
                    None, broker_type=broker_type, task_visibility=db_index, aws_role=aws_profile
                )
            )
            for db_index in get_all_db_indexes()
        }
    elif broker_type == "sqs":
        # for sqs we will get active/reserved counts directly from sqs as opposed to using
        #   an inspect object
        inspect = {}
    else:
        raise ValueError("broker_type not redis or sqs, how did we get here?")

    env = os.getenv("DD_ENV")
    instance_count = int(os.getenv("POD_NAME", "pod-0").split("-")[-1])
    num_shards = int(os.getenv("NUM_SHARDS", 1))

    env = f"{env}-{autoscaler_broker}"

    while True:
        try:
            loop_start = time.time()
            deployments = await list_deployments(core_api=core_api, apps_api=apps_api)
            logger.info(f"list_deployments took {time.time() - loop_start} seconds")
            celery_queues = set()
            celery_queues_params = []
            for deployment_and_namespace, params in sorted(
                deployments.items()
            ):  # sort for a bit more determinism
                # Hash the deployment / namespace to deterministically partition the deployments.
                # Skip all deployments not in this partition.
                if _hash_any_to_int(deployment_and_namespace) % num_shards != instance_count:
                    continue

                deployment_name, namespace = deployment_and_namespace
                instance = instances.get(deployment_and_namespace)
                if instance is None or instance.params != params:
                    instances[deployment_and_namespace] = Instance(
                        apps_api, deployment_name, namespace, params, env
                    )

                # We're treating a queue as a pair consisting of a (queue_name, db_index).
                # This means that two queues that happen to have the same name are treated
                # as semantically distinct if they have different db_indexes.
                celery_queues.add((params.queue, params.task_visibility.value))
                celery_queues_params.append(params.__dict__)

            # Clean up instances not in set
            for deployment_and_namespace in set(instances) - set(deployments):
                del instances[deployment_and_namespace]

            # Get queue sizes
            # (queue_name, db_index) -> QueueSizes
            start_get_metrics = time.time()
            metrics = await get_metrics(broker, inspect=inspect, queues=celery_queues)
            logger.info(f"get_metrics took {time.time() - start_get_metrics} seconds")

            queue_sizes = metrics.broker_metrics.queue_sizes
            for k, v in sorted(queue_sizes.items()):
                queue_name, _ = k
                logger.info(f"Inflight : {queue_name} : {v.total}")

            emit_metrics(metrics=metrics, env=env)

            # Update scaling
            for instance in instances.values():
                queue_size = queue_sizes[
                    (instance.params.queue, int(instance.params.task_visibility))
                ]
                try:
                    await instance.check_queue_size_and_update_deployment(queue_size.total)
                except Exception as e:
                    logger.exception(f"Failed to update {instance.name}: {e}")

            # Wait before next iteration
            iteration_len = time.time() - loop_start
            logger.info(f"Iteration length: {iteration_len} seconds.")
            if iteration_len < 3:
                await aio.sleep(3 - iteration_len)

            emit_health_metric("heartbeat", env)
        except Exception as e:
            logger.exception(f"Error in deployment loop: {e}")
            continue


if __name__ == "__main__":
    aio.run(main())
