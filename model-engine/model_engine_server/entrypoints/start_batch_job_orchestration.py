import argparse
import asyncio
import os
from datetime import timedelta

import aioredis
from model_engine_server.api.dependencies import get_monitoring_metrics_gateway
from model_engine_server.common.config import hmi_config
from model_engine_server.common.dtos.model_endpoints import BrokerType
from model_engine_server.common.env_vars import CIRCLECI
from model_engine_server.core.config import infra_config
from model_engine_server.db.base import SessionAsyncNullPool
from model_engine_server.domain.entities import BatchJobSerializationFormat
from model_engine_server.infra.gateways import (
    ABSFilesystemGateway,
    CeleryTaskQueueGateway,
    LiveAsyncModelEndpointInferenceGateway,
    LiveBatchJobProgressGateway,
    LiveModelEndpointInfraGateway,
    LiveModelEndpointsSchemaGateway,
    LiveStreamingModelEndpointInferenceGateway,
    LiveSyncModelEndpointInferenceGateway,
    RedisInferenceAutoscalingMetricsGateway,
    S3FilesystemGateway,
)
from model_engine_server.infra.gateways.resources.asb_queue_endpoint_resource_delegate import (
    ASBQueueEndpointResourceDelegate,
)
from model_engine_server.infra.gateways.resources.fake_queue_endpoint_resource_delegate import (
    FakeQueueEndpointResourceDelegate,
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
    DbBatchJobRecordRepository,
    DbModelEndpointRecordRepository,
    RedisModelEndpointCacheRepository,
)
from model_engine_server.infra.services import (
    LiveBatchJobOrchestrationService,
    LiveModelEndpointService,
)


async def run_batch_job(
    job_id: str,
    owner: str,
    input_path: str,
    serialization_format: BatchJobSerializationFormat,
    timeout_seconds: float,
):
    session = SessionAsyncNullPool
    pool = aioredis.BlockingConnectionPool.from_url(hmi_config.cache_redis_url)
    redis = aioredis.Redis(connection_pool=pool)
    redis_task_queue_gateway = CeleryTaskQueueGateway(broker_type=BrokerType.REDIS)
    sqs_task_queue_gateway = CeleryTaskQueueGateway(broker_type=BrokerType.SQS)
    servicebus_task_queue_gateway = CeleryTaskQueueGateway(broker_type=BrokerType.SERVICEBUS)

    monitoring_metrics_gateway = get_monitoring_metrics_gateway()
    model_endpoint_record_repo = DbModelEndpointRecordRepository(
        monitoring_metrics_gateway=monitoring_metrics_gateway, session=session, read_only=False
    )

    queue_delegate: QueueEndpointResourceDelegate
    if CIRCLECI:
        queue_delegate = FakeQueueEndpointResourceDelegate()
    elif infra_config().cloud_provider == "azure":
        queue_delegate = ASBQueueEndpointResourceDelegate()
    else:
        queue_delegate = SQSQueueEndpointResourceDelegate(
            sqs_profile=os.getenv("SQS_PROFILE", hmi_config.sqs_profile)
        )

    resource_gateway = LiveEndpointResourceGateway(queue_delegate=queue_delegate)
    model_endpoint_cache_repo = RedisModelEndpointCacheRepository(
        redis_client=redis,
    )
    model_endpoint_infra_gateway = LiveModelEndpointInfraGateway(
        resource_gateway=resource_gateway,
        task_queue_gateway=redis_task_queue_gateway,
    )
    async_model_endpoint_inference_gateway = LiveAsyncModelEndpointInferenceGateway(
        task_queue_gateway=servicebus_task_queue_gateway
        if infra_config().cloud_provider == "azure"
        else sqs_task_queue_gateway
    )
    streaming_model_endpoint_inference_gateway = LiveStreamingModelEndpointInferenceGateway(
        use_asyncio=(not CIRCLECI),
    )
    sync_model_endpoint_inference_gateway = LiveSyncModelEndpointInferenceGateway(
        use_asyncio=(not CIRCLECI),
    )
    filesystem_gateway = (
        ABSFilesystemGateway()
        if infra_config().cloud_provider == "azure"
        else S3FilesystemGateway()
    )
    model_endpoints_schema_gateway = LiveModelEndpointsSchemaGateway(
        filesystem_gateway=filesystem_gateway
    )
    inference_autoscaling_metrics_gateway = RedisInferenceAutoscalingMetricsGateway(
        redis_client=redis,
    )
    model_endpoint_service = LiveModelEndpointService(
        model_endpoint_record_repository=model_endpoint_record_repo,
        model_endpoint_infra_gateway=model_endpoint_infra_gateway,
        model_endpoint_cache_repository=model_endpoint_cache_repo,
        async_model_endpoint_inference_gateway=async_model_endpoint_inference_gateway,
        streaming_model_endpoint_inference_gateway=streaming_model_endpoint_inference_gateway,
        sync_model_endpoint_inference_gateway=sync_model_endpoint_inference_gateway,
        model_endpoints_schema_gateway=model_endpoints_schema_gateway,
        inference_autoscaling_metrics_gateway=inference_autoscaling_metrics_gateway,
    )
    batch_job_record_repository = DbBatchJobRecordRepository(session=session, read_only=False)
    batch_job_progress_gateway = LiveBatchJobProgressGateway(filesystem_gateway=filesystem_gateway)

    batch_job_orchestration_service = LiveBatchJobOrchestrationService(
        model_endpoint_service=model_endpoint_service,
        batch_job_record_repository=batch_job_record_repository,
        batch_job_progress_gateway=batch_job_progress_gateway,
        async_model_endpoint_inference_gateway=async_model_endpoint_inference_gateway,
        filesystem_gateway=filesystem_gateway,
    )

    await batch_job_orchestration_service.run_batch_job(
        job_id=job_id,
        owner=owner,
        input_path=input_path,
        serialization_format=serialization_format,
        timeout=timedelta(seconds=timeout_seconds),
    )


def entrypoint():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", "-j", required=True, help="The ID of the batch job to run.")
    parser.add_argument(
        "--owner", "-o", required=True, help="The ID of the user who owns the batch job."
    )
    parser.add_argument("--input-path", "-i", required=True, help="The path to the input data.")
    parser.add_argument(
        "--serialization-format",
        "-s",
        required=True,
        help="The serialization format of the input data.",
    )
    parser.add_argument("--timeout-seconds", "-t", required=True, help="The timeout in seconds.")

    args = parser.parse_args()
    serialization_fmt = BatchJobSerializationFormat(args.serialization_format)
    asyncio.run(
        run_batch_job(
            job_id=args.job_id,
            owner=args.owner,
            input_path=args.input_path,
            serialization_format=serialization_fmt,
            timeout_seconds=float(args.timeout_seconds),
        )
    )


if __name__ == "__main__":
    entrypoint()
