import asyncio
import os
import time
import traceback
from typing import Any, Dict

import aioredis
from celery.signals import worker_process_init
from celery.utils.log import get_task_logger
from model_engine_server.api.dependencies import get_monitoring_metrics_gateway
from model_engine_server.common.config import hmi_config
from model_engine_server.common.constants import READYZ_FPATH
from model_engine_server.common.dtos.endpoint_builder import (
    BuildEndpointRequest,
    BuildEndpointResponse,
)
from model_engine_server.common.env_vars import CIRCLECI
from model_engine_server.core.config import infra_config
from model_engine_server.core.fake_notification_gateway import FakeNotificationGateway
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.db.base import get_session_async_null_pool
from model_engine_server.domain.repositories import DockerRepository
from model_engine_server.infra.gateways import (
    ABSFilesystemGateway,
    ASBInferenceAutoscalingMetricsGateway,
    RedisInferenceAutoscalingMetricsGateway,
    S3FilesystemGateway,
)
from model_engine_server.infra.gateways.resources.asb_queue_endpoint_resource_delegate import (
    ASBQueueEndpointResourceDelegate,
)
from model_engine_server.infra.gateways.resources.fake_queue_endpoint_resource_delegate import (
    FakeQueueEndpointResourceDelegate,
)
from model_engine_server.infra.gateways.resources.k8s_endpoint_resource_delegate import (
    set_lazy_load_kubernetes_clients,
)
from model_engine_server.infra.gateways.resources.live_endpoint_resource_gateway import (
    LiveEndpointResourceGateway,
)
from model_engine_server.infra.gateways.resources.queue_endpoint_resource_delegate import (
    QueueEndpointResourceDelegate,
)
from model_engine_server.infra.gateways.resources.sqs_queue_endpoint_resource_delegate import (
    SQSQueueEndpointResourceDelegate,
)
from model_engine_server.infra.repositories import (
    ACRDockerRepository,
    DbModelEndpointRecordRepository,
    ECRDockerRepository,
    FakeDockerRepository,
    RedisFeatureFlagRepository,
    RedisModelEndpointCacheRepository,
)
from model_engine_server.infra.repositories.onprem_docker_repository import OnPremDockerRepository
from model_engine_server.infra.services import LiveEndpointBuilderService
from model_engine_server.service_builder.celery import service_builder_service

# Need to disable lazy loading of k8s clients because each event loop should contain its own k8s
# client, which constructs the aiohttp.ClientSession in the event loop.
set_lazy_load_kubernetes_clients(False)

# Create logger for this module
logger = make_logger(logger_name())
# Get Celery task logger for task-specific logging
task_logger = get_task_logger(__name__)


def get_live_endpoint_builder_service(
    session: Any,
    redis: aioredis.Redis,
):
    queue_delegate: QueueEndpointResourceDelegate
    if CIRCLECI:
        queue_delegate = FakeQueueEndpointResourceDelegate()
    elif infra_config().cloud_provider == "azure":
        queue_delegate = ASBQueueEndpointResourceDelegate()
    else:
        queue_delegate = SQSQueueEndpointResourceDelegate(
            sqs_profile=os.getenv("SQS_PROFILE", hmi_config.sqs_profile)
        )
    notification_gateway = FakeNotificationGateway()
    monitoring_metrics_gateway = get_monitoring_metrics_gateway()
    docker_repository: DockerRepository
    if CIRCLECI:
        docker_repository = FakeDockerRepository()
    elif infra_config().cloud_provider == "azure":
        docker_repository = ACRDockerRepository()
    elif infra_config().cloud_provider == "onprem":
        docker_repository = OnPremDockerRepository()
    else:
        docker_repository = ECRDockerRepository()
    inference_autoscaling_metrics_gateway = (
        ASBInferenceAutoscalingMetricsGateway()
        if infra_config().cloud_provider == "azure"
        else RedisInferenceAutoscalingMetricsGateway(redis_client=redis)
    )
    service = LiveEndpointBuilderService(
        docker_repository=docker_repository,
        resource_gateway=LiveEndpointResourceGateway(
            queue_delegate=queue_delegate,
            inference_autoscaling_metrics_gateway=inference_autoscaling_metrics_gateway,
        ),
        monitoring_metrics_gateway=monitoring_metrics_gateway,
        model_endpoint_record_repository=DbModelEndpointRecordRepository(
            monitoring_metrics_gateway=monitoring_metrics_gateway, session=session, read_only=False
        ),
        model_endpoint_cache_repository=RedisModelEndpointCacheRepository(redis_client=redis),
        filesystem_gateway=(
            ABSFilesystemGateway()
            if infra_config().cloud_provider == "azure"
            else S3FilesystemGateway()
        ),
        notification_gateway=notification_gateway,
        feature_flag_repo=RedisFeatureFlagRepository(redis_client=redis),
    )

    return service


async def _build_endpoint(
    build_endpoint_request: BuildEndpointRequest,
) -> BuildEndpointResponse:
    task_start_time = time.time()
    logger.info(
        "Starting endpoint build process",
        extra={
            "endpoint_name": build_endpoint_request.model_endpoint_record.name,
            "request_id": getattr(build_endpoint_request.model_endpoint_record, "id", "unknown"),
            "user_id": getattr(
                build_endpoint_request.model_endpoint_record, "created_by", "unknown"
            ),
        },
    )

    session = None
    redis = None
    pool = None

    try:
        # Database connection
        if infra_config().debug_mode:  # pragma: no cover
            logger.info("Establishing database session")
        session = get_session_async_null_pool()
        if infra_config().debug_mode:  # pragma: no cover
            logger.info("Database session established successfully")

        # Redis connection
        if infra_config().debug_mode:  # pragma: no cover
            logger.info("Connecting to Redis", extra={"redis_url": hmi_config.cache_redis_url})
        pool = aioredis.BlockingConnectionPool.from_url(hmi_config.cache_redis_url)
        redis = aioredis.Redis(connection_pool=pool)
        if infra_config().debug_mode:  # pragma: no cover
            logger.info("Redis connection established successfully")

        # Service initialization
        if infra_config().debug_mode:  # pragma: no cover
            logger.info("Initializing LiveEndpointBuilderService")
        service: LiveEndpointBuilderService = get_live_endpoint_builder_service(session, redis)
        if infra_config().debug_mode:  # pragma: no cover
            logger.info("LiveEndpointBuilderService initialized successfully")

        # Actual endpoint building
        if infra_config().debug_mode:  # pragma: no cover
            logger.info("Starting endpoint build operation")
        response = await service.build_endpoint(build_endpoint_request)

        build_time = time.time() - task_start_time
        if infra_config().debug_mode:  # pragma: no cover
            logger.info(
                "Endpoint build completed successfully",
                extra={
                    "endpoint_name": build_endpoint_request.model_endpoint_record.name,
                    "build_time_seconds": build_time,
                    "response_status": getattr(response, "status", "unknown"),
                },
            )

        return response

    except Exception as e:
        build_time = time.time() - task_start_time
        error_details = {
            "endpoint_name": build_endpoint_request.model_endpoint_record.name,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "build_time_seconds": build_time,
            "traceback": traceback.format_exc(),
        }

        logger.error("Endpoint build failed with exception", extra=error_details)

        # Re-raise the exception so Celery knows the task failed
        raise

    finally:
        # Cleanup resources
        cleanup_start = time.time()
        try:
            if redis:
                if infra_config().debug_mode:  # pragma: no cover
                    logger.info("Closing Redis connection")
                await redis.close()
                if infra_config().debug_mode:  # pragma: no cover
                    logger.info("Redis connection closed")
        except Exception as e:
            logger.warning(f"Error closing Redis connection: {e}")

        try:
            if pool:
                if infra_config().debug_mode:  # pragma: no cover
                    logger.info("Disconnecting Redis pool")
                await pool.disconnect()
                if infra_config().debug_mode:  # pragma: no cover
                    logger.info("Redis pool disconnected")
        except Exception as e:
            logger.warning(f"Error disconnecting Redis pool: {e}")

        if infra_config().debug_mode:  # pragma: no cover
            cleanup_time = time.time() - cleanup_start
            logger.info(f"Resource cleanup completed in {cleanup_time:.2f} seconds")


@worker_process_init.connect
def init_worker(*args, **kwargs):
    if infra_config().debug_mode:  # pragma: no cover
        logger.info("Initializing Celery worker process")
    # k8s health check
    with open(READYZ_FPATH, "w") as f:
        f.write("READY")
    if infra_config().debug_mode:  # pragma: no cover
        logger.info("Worker process initialized successfully")


@service_builder_service.task(bind=True)
def build_endpoint(self, build_endpoint_request_json: Dict[str, Any]) -> Dict[str, str]:
    task_start_time = time.time()
    task_id = self.request.id

    # Log task start with detailed context
    if infra_config().debug_mode:  # pragma: no cover
        task_logger.info(
            "Task started",
            extra={
                "task_id": task_id,
                "task_name": "build_endpoint",
                "request_data_keys": (
                    list(build_endpoint_request_json.keys()) if build_endpoint_request_json else []
                ),
                "worker_hostname": self.request.hostname,
            },
        )

    try:
        # Parse request
        if infra_config().debug_mode:  # pragma: no cover
            task_logger.info("Parsing build endpoint request", extra={"task_id": task_id})
        build_endpoint_request: BuildEndpointRequest = BuildEndpointRequest.parse_obj(
            build_endpoint_request_json
        )
        if infra_config().debug_mode:  # pragma: no cover
            task_logger.info(
                "Request parsed successfully",
                extra={
                    "task_id": task_id,
                    "endpoint_name": build_endpoint_request.model_endpoint_record.name,
                    "endpoint_type": getattr(
                        build_endpoint_request.model_endpoint_record, "endpoint_type", "unknown"
                    ),
                },
            )

        # Execute the async build process
        if infra_config().debug_mode:  # pragma: no cover
            task_logger.info("Starting async endpoint build", extra={"task_id": task_id})
        result = asyncio.run(_build_endpoint(build_endpoint_request))

        # Log successful completion
        if infra_config().debug_mode:  # pragma: no cover
            task_duration = time.time() - task_start_time
            task_logger.info(
                "Task completed successfully",
                extra={
                    "task_id": task_id,
                    "endpoint_name": build_endpoint_request.model_endpoint_record.name,
                    "task_duration_seconds": task_duration,
                    "result_status": getattr(result, "status", "unknown"),
                },
            )

        return result.dict()

    except Exception as e:
        task_duration = time.time() - task_start_time
        error_info = {
            "task_id": task_id,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "task_duration_seconds": task_duration,
            "full_traceback": traceback.format_exc(),
        }

        # Add request context if available
        try:
            if "build_endpoint_request" in locals():
                error_info["endpoint_name"] = build_endpoint_request.model_endpoint_record.name
                error_info["request_context"] = {
                    "endpoint_type": getattr(
                        build_endpoint_request.model_endpoint_record, "endpoint_type", "unknown"
                    ),
                    "created_by": getattr(
                        build_endpoint_request.model_endpoint_record, "created_by", "unknown"
                    ),
                }
        except Exception:
            pass

        task_logger.error("Task failed with exception", extra=error_info)

        # Re-raise to let Celery handle the failure
        raise
