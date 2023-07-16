from typing import Dict, Optional

from llm_engine_server.common.dtos.batch_jobs import CreateBatchJobResourceRequests
from llm_engine_server.core.loggers import filename_wo_ext, make_logger
from llm_engine_server.domain.entities import (
    BatchJob,
    BatchJobProgress,
    BatchJobSerializationFormat,
    BatchJobStatus,
    GpuType,
    ModelEndpointType,
)
from llm_engine_server.domain.exceptions import EndpointResourceInvalidRequestException
from llm_engine_server.domain.services import BatchJobService, ModelEndpointService
from llm_engine_server.infra.gateways import BatchJobOrchestrationGateway, BatchJobProgressGateway
from llm_engine_server.infra.repositories.batch_job_record_repository import BatchJobRecordRepository

logger = make_logger(filename_wo_ext(__file__))

DEFAULT_ENDPOINT_CPUS_BATCH_JOB = 3
DEFAULT_ENDPOINT_MEMORY_BATCH_JOB = "12Gi"
DEFAULT_ENDPOINT_GPUS_BATCH_JOB = 1
DEFAULT_ENDPOINT_GPU_TYPE_BATCH_JOB = GpuType.NVIDIA_TESLA_T4
DEFAULT_ENDPOINT_MAX_WORKERS_BATCH_JOB = 50
DEFAULT_ENDPOINT_PER_WORKER_BATCH_JOB = 40

BATCH_TASK_IDENTIFIER = "batch-task"


def get_batch_task_resource_group_name(model_bundle_id: str, job_id: str):
    return f"{model_bundle_id}-{BATCH_TASK_IDENTIFIER}-{job_id}".replace("_", "-")


class LiveBatchJobService(BatchJobService):
    def __init__(
        self,
        batch_job_record_repository: BatchJobRecordRepository,
        model_endpoint_service: ModelEndpointService,
        batch_job_orchestration_gateway: BatchJobOrchestrationGateway,
        batch_job_progress_gateway: BatchJobProgressGateway,
    ):
        self.batch_job_record_repository = batch_job_record_repository
        self.model_endpoint_service = model_endpoint_service
        self.batch_job_orchestration_gateway = batch_job_orchestration_gateway
        self.batch_job_progress_gateway = batch_job_progress_gateway

    async def create_batch_job(
        self,
        *,
        created_by: str,
        owner: str,
        model_bundle_id: str,
        input_path: str,
        serialization_format: BatchJobSerializationFormat,
        labels: Dict[str, str],
        resource_requests: CreateBatchJobResourceRequests,
        aws_role: str,
        results_s3_bucket: str,
        timeout_seconds: float,
    ) -> str:
        batch_job_record = await self.batch_job_record_repository.create_batch_job_record(
            status=BatchJobStatus.PENDING,
            created_by=created_by,
            owner=owner,
            model_bundle_id=model_bundle_id,
        )
        model_bundle_id = batch_job_record.model_bundle.id
        job_id = batch_job_record.id
        resource_group_name = get_batch_task_resource_group_name(model_bundle_id, job_id)
        cpus = resource_requests.cpus or DEFAULT_ENDPOINT_CPUS_BATCH_JOB
        gpus = (
            resource_requests.gpus
            if resource_requests.gpus is not None
            else DEFAULT_ENDPOINT_GPUS_BATCH_JOB
        )
        memory = resource_requests.memory or DEFAULT_ENDPOINT_MEMORY_BATCH_JOB
        gpu_type = None
        if gpus == 0 and resource_requests.gpu_type is not None:
            raise EndpointResourceInvalidRequestException(
                f"Cannot specify a GPU type {resource_requests.gpu_type} when requesting 0 GPUs"
            )
        elif gpus > 0:
            if resource_requests.gpu_type is None:
                gpu_type = DEFAULT_ENDPOINT_GPU_TYPE_BATCH_JOB
            else:
                gpu_type = resource_requests.gpu_type
        max_workers = resource_requests.max_workers or DEFAULT_ENDPOINT_MAX_WORKERS_BATCH_JOB
        per_worker = resource_requests.per_worker or DEFAULT_ENDPOINT_PER_WORKER_BATCH_JOB

        model_endpoint_record = await self.model_endpoint_service.create_model_endpoint(
            name=resource_group_name,
            created_by=created_by,
            model_bundle_id=model_bundle_id,
            endpoint_type=ModelEndpointType.ASYNC,
            metadata={},
            post_inference_hooks=None,
            child_fn_info=None,
            cpus=cpus,  # type: ignore
            gpus=gpus,  # type: ignore
            memory=memory,  # type: ignore
            gpu_type=gpu_type,  # type: ignore
            storage=resource_requests.storage,
            optimize_costs=False,
            min_workers=0,
            max_workers=max_workers,  # type: ignore
            per_worker=per_worker,  # type: ignore
            labels=labels,
            aws_role=aws_role,
            results_s3_bucket=results_s3_bucket,
            prewarm=True,
            high_priority=False,  # pod spin up should not matter for batch jobs
            owner=owner,
            default_callback_url=None,
            default_callback_auth=None,
        )

        await self.batch_job_record_repository.update_batch_job_record(
            batch_job_id=job_id,
            model_endpoint_id=model_endpoint_record.id,
        )

        await self.batch_job_orchestration_gateway.create_batch_job_orchestrator(
            job_id=batch_job_record.id,
            resource_group_name=resource_group_name,
            owner=owner,
            input_path=input_path,
            serialization_format=serialization_format,
            labels=labels,
            timeout_seconds=timeout_seconds,
        )
        return batch_job_record.id

    async def get_batch_job(self, batch_job_id: str) -> Optional[BatchJob]:
        batch_job_record = await self.batch_job_record_repository.get_batch_job_record(
            batch_job_id=batch_job_id
        )
        if batch_job_record is None:
            return None

        model_endpoint = None
        if batch_job_record.model_endpoint_id is not None:
            model_endpoint = await self.model_endpoint_service.get_model_endpoint(
                model_endpoint_id=batch_job_record.model_endpoint_id,
            )
        if batch_job_record.status != BatchJobStatus.PENDING:
            progress = self.batch_job_progress_gateway.get_progress(
                owner=batch_job_record.owner, batch_job_id=batch_job_record.id
            )
        else:
            progress = BatchJobProgress(
                num_tasks_pending=None,
                num_tasks_completed=None,
            )

        return BatchJob(record=batch_job_record, model_endpoint=model_endpoint, progress=progress)

    async def update_batch_job(self, batch_job_id: str, cancel: bool) -> bool:
        if cancel:
            return await self.cancel_batch_job(batch_job_id=batch_job_id)
        return True

    async def cancel_batch_job(self, batch_job_id: str) -> bool:
        batch_job = await self.get_batch_job(batch_job_id=batch_job_id)
        if batch_job is None:
            return False

        terminal_states = {BatchJobStatus.CANCELLED, BatchJobStatus.SUCCESS}
        if batch_job.model_endpoint is None or batch_job.record.status in terminal_states:
            logger.info(f"Batch job {batch_job_id} is already in a terminal state. Skipping.")
            return True

        success = False
        model_bundle_id = batch_job.record.model_bundle.id
        resource_group_name = get_batch_task_resource_group_name(model_bundle_id, batch_job_id)
        try:
            success = await self.batch_job_orchestration_gateway.delete_batch_job_orchestrator(
                resource_group_name
            )
            await self.batch_job_record_repository.unset_model_endpoint_id(
                batch_job_id=batch_job_id
            )
            await self.model_endpoint_service.delete_model_endpoint(
                model_endpoint_id=batch_job.model_endpoint.record.id,
            )
            await self.batch_job_record_repository.update_batch_job_record(
                batch_job_id=batch_job_id, status=BatchJobStatus.CANCELLED
            )
            success = True
        except:
            logger.exception(f"Failed to cancel batch job {batch_job_id}")
        finally:
            return success
