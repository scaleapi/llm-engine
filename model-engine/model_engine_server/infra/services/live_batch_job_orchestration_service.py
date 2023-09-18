import asyncio
import base64
import csv
import dataclasses
import json
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Union

from model_engine_server.common.dtos.tasks import (
    EndpointPredictV1Request,
    GetAsyncTaskV1Response,
    TaskStatus,
)
from model_engine_server.core.config import infra_config
from model_engine_server.domain.exceptions import ObjectNotFoundException
from model_engine_server.core.loggers import filename_wo_ext, make_logger
from model_engine_server.domain.entities import (
    BatchJobProgress,
    BatchJobRecord,
    BatchJobSerializationFormat,
    BatchJobStatus,
    ModelEndpointStatus,
)
from model_engine_server.domain.gateways import AsyncModelEndpointInferenceGateway
from model_engine_server.domain.services import ModelEndpointService
from model_engine_server.domain.use_cases.async_inference_use_cases import (
    DEFAULT_TASK_TIMEOUT_SECONDS,
)
from model_engine_server.infra.gateways import BatchJobProgressGateway
from model_engine_server.infra.gateways.filesystem_gateway import FilesystemGateway
from model_engine_server.infra.repositories.batch_job_record_repository import (
    BatchJobRecordRepository,
)
from model_engine_server.infra.services.batch_job_orchestration_service import (
    BatchJobOrchestrationService,
)

logger = make_logger(filename_wo_ext(__file__))


@dataclass
class BatchEndpointInferencePrediction:
    request: EndpointPredictV1Request
    reference_id: Optional[str]


@dataclass
class BatchEndpointInProgressTask:
    task_id: str
    reference_id: Optional[str]

    def serialize(self):
        return json.dumps(dataclasses.asdict(self))

    @classmethod
    def deserialize(cls, serialized: str):
        return BatchEndpointInProgressTask(**json.loads(serialized))


@dataclass
class BatchEndpointInferencePredictionResponse:
    response: GetAsyncTaskV1Response
    reference_id: Optional[str]

    def json(self):
        response_dict = self.response.dict()
        response_dict["id"] = self.reference_id
        return json.dumps(response_dict)


class LiveBatchJobOrchestrationService(BatchJobOrchestrationService):
    """
    Service for running batch jobs in a live environment.
    """

    def __init__(
        self,
        model_endpoint_service: ModelEndpointService,
        batch_job_record_repository: BatchJobRecordRepository,
        batch_job_progress_gateway: BatchJobProgressGateway,
        async_model_endpoint_inference_gateway: AsyncModelEndpointInferenceGateway,
        filesystem_gateway: FilesystemGateway,
    ):
        self.model_endpoint_service = model_endpoint_service
        self.batch_job_record_repository = batch_job_record_repository
        self.batch_job_progress_gateway = batch_job_progress_gateway
        self.async_model_endpoint_inference_gateway = async_model_endpoint_inference_gateway
        self.filesystem_gateway = filesystem_gateway

    async def run_batch_job(
        self,
        *,
        job_id: str,
        owner: str,
        input_path: str,
        serialization_format: BatchJobSerializationFormat,
        timeout: timedelta,
    ) -> None:
        try:
            await self._run_batch_job(
                job_id=job_id,
                owner=owner,
                input_path=input_path,
                serialization_format=serialization_format,
                timeout=timeout,
            )
        except ObjectNotFoundException:
            logger.exception(f"Could not find batch job with ID {job_id}.")
            raise
        except TimeoutError:
            await self.batch_job_record_repository.update_batch_job_record(
                batch_job_id=job_id, status=BatchJobStatus.TIMEOUT
            )
            logger.exception(f"Batch job {job_id} timed out.")
        except Exception as e:
            await self.batch_job_record_repository.update_batch_job_record(
                batch_job_id=job_id, status=BatchJobStatus.FAILURE
            )
            logger.exception(f"Batch job {job_id} failed with exception {e}")

    async def _run_batch_job(
        self,
        *,
        job_id: str,
        owner: str,
        input_path: str,
        serialization_format: BatchJobSerializationFormat,
        timeout: timedelta,
    ) -> None:
        current_time = datetime.utcnow()
        timeout_timestamp = current_time + timeout
        logger.info(f"Running batch job {job_id} for owner {owner}")
        batch_job_record = await self.batch_job_record_repository.get_batch_job_record(job_id)

        if batch_job_record is None:
            raise ObjectNotFoundException(f"Batch job {job_id} not found")

        model_endpoint_id = batch_job_record.model_endpoint_id
        # For mypy.
        if model_endpoint_id is None:
            raise ObjectNotFoundException(f"Batch job {job_id} does not have a model endpoint")

        await self._wait_for_endpoint_to_be_ready(model_endpoint_id, timeout_timestamp)
        model_endpoint_record = await self.model_endpoint_service.get_model_endpoint_record(
            model_endpoint_id=model_endpoint_id,
        )
        assert model_endpoint_record is not None
        queue_name = model_endpoint_record.destination

        task_name = model_endpoint_record.current_model_bundle.celery_task_name()

        task_ids = await self._read_or_submit_tasks(
            batch_job_record=batch_job_record,
            queue_name=queue_name,
            input_path=input_path,
            task_name=task_name,
        )

        await self.batch_job_record_repository.update_batch_job_record(
            batch_job_id=job_id,
            status=BatchJobStatus.RUNNING,
        )

        results = self._poll_tasks(
            owner=owner, job_id=job_id, task_ids=task_ids, timeout_timestamp=timeout_timestamp
        )

        result_location = batch_job_record.result_location
        if not result_location:
            result_location = self._get_job_result_location(job_id)

        self._serialize_and_write_results(result_location, serialization_format, results)
        await self.batch_job_record_repository.update_batch_job_record(
            batch_job_id=job_id, result_location=result_location
        )

        # Cleanup the created endpoint
        model_endpoint_id = batch_job_record.model_endpoint_id
        if model_endpoint_id:
            await self.batch_job_record_repository.unset_model_endpoint_id(
                batch_job_id=job_id,
            )
            await self.model_endpoint_service.delete_model_endpoint(model_endpoint_id)

        # Set the status to success if it hasn't been set to failure or timeout
        batch_job_status = BatchJobStatus.SUCCESS
        if datetime.utcnow() > timeout_timestamp:
            batch_job_status = BatchJobStatus.TIMEOUT
        await self.batch_job_record_repository.update_batch_job_record(
            batch_job_id=job_id,
            status=batch_job_status,
            completed_at=datetime.now(),
        )

    async def _wait_for_endpoint_to_be_ready(
        self, model_endpoint_id: str, timeout_timestamp: datetime
    ) -> None:
        model_endpoint = await self.model_endpoint_service.get_model_endpoint_record(
            model_endpoint_id=model_endpoint_id,
        )
        updating = {ModelEndpointStatus.UPDATE_PENDING, ModelEndpointStatus.UPDATE_IN_PROGRESS}

        assert model_endpoint
        while model_endpoint.status in updating:
            logger.info(f"Waiting for model endpoint {model_endpoint_id} to be ready")
            await asyncio.sleep(5)
            model_endpoint = await self.model_endpoint_service.get_model_endpoint_record(
                model_endpoint_id=model_endpoint_id,
            )
            if datetime.utcnow() > timeout_timestamp:
                raise TimeoutError(
                    f"Timed out waiting for model endpoint {model_endpoint_id} to be ready"
                )
            assert model_endpoint

        if model_endpoint.status != ModelEndpointStatus.READY:
            raise ObjectNotFoundException(
                f"Model endpoint {model_endpoint_id} was not successfully created and has status "
                f"{model_endpoint.status}."
            )

    async def _read_or_submit_tasks(
        self,
        *,
        batch_job_record: BatchJobRecord,
        queue_name: str,
        input_path: str,
        task_name: str,
    ) -> List[BatchEndpointInProgressTask]:
        task_ids: Optional[List[BatchEndpointInProgressTask]] = None
        job_id = batch_job_record.id
        try:
            # If the job has already been run and failed halfway, then we want to avoid resubmitting
            # tasks.
            logger.info(f"Checking if batch job {job_id} has already been run")
            pending_task_ids_location = batch_job_record.task_ids_location
            if pending_task_ids_location is not None:
                with self.filesystem_gateway.open(
                    pending_task_ids_location, "r", aws_profile=infra_config().profile_ml_worker
                ) as f:
                    task_ids_serialized = f.read().splitlines()
                    task_ids = [
                        BatchEndpointInProgressTask.deserialize(tid) for tid in task_ids_serialized
                    ]
                    num_task_ids = len(task_ids)  # type:ignore
                    logger.info(f"Found {num_task_ids} pending tasks for batch job {job_id}")
        finally:
            if task_ids is None:
                logger.info(f"Did not find pending tasks. Submitting tasks for batch job {job_id}")
                task_ids = await self._submit_tasks(queue_name, input_path, task_name)
                pending_task_ids_location = self._get_pending_task_ids_location(job_id)
                with self.filesystem_gateway.open(
                    pending_task_ids_location, "w", aws_profile=infra_config().profile_ml_worker
                ) as f:
                    f.write("\n".join([tid.serialize() for tid in task_ids]))
                await self.batch_job_record_repository.update_batch_job_record(
                    batch_job_id=job_id, task_ids_location=pending_task_ids_location
                )

        assert task_ids is not None
        return task_ids

    async def _submit_tasks(
        self, queue_name: str, input_path: str, task_name: str
    ) -> List[BatchEndpointInProgressTask]:
        def _create_task(
            predict_request: BatchEndpointInferencePrediction,
        ) -> BatchEndpointInProgressTask:
            response = self.async_model_endpoint_inference_gateway.create_task(
                topic=queue_name,
                predict_request=predict_request.request,
                task_timeout_seconds=DEFAULT_TASK_TIMEOUT_SECONDS,
                task_name=task_name,
            )
            return BatchEndpointInProgressTask(
                task_id=response.task_id, reference_id=predict_request.reference_id
            )

        inputs: List[BatchEndpointInferencePrediction] = []
        with self.filesystem_gateway.open(
            input_path, "r", aws_profile=infra_config().profile_ml_worker
        ) as f:
            # Increase the CSV reader's field limit size from the default (131072)
            csv.field_size_limit(sys.maxsize)
            reader = csv.DictReader(f)
            for line in reader:
                args = line.get("args")
                if args is not None:
                    args = json.loads(base64.b64decode(args).decode("utf-8"))
                request = EndpointPredictV1Request(
                    url=line.get("url"),
                    args=args,
                    return_pickled=False,
                )
                reference_id = line.get("id")
                inputs.append(
                    BatchEndpointInferencePrediction(request=request, reference_id=reference_id)
                )

        executor = ThreadPoolExecutor()
        task_ids = list(executor.map(_create_task, inputs))
        return task_ids

    def _poll_tasks(
        self,
        owner: str,
        job_id: str,
        task_ids: List[BatchEndpointInProgressTask],
        timeout_timestamp: datetime,
    ) -> List[BatchEndpointInferencePredictionResponse]:
        # Poll the task queue until all tasks are complete.
        # Python multithreading works here because retrieving the tasks is I/O bound.
        task_ids_only = [in_progress_task.task_id for in_progress_task in task_ids]
        task_id_to_ref_id_map = {
            in_progress_task.task_id: in_progress_task.reference_id for in_progress_task in task_ids
        }
        pending_task_ids_set = set(task_ids_only)
        task_id_to_result = {}
        executor = ThreadPoolExecutor()
        progress = BatchJobProgress(
            num_tasks_pending=len(pending_task_ids_set),
            num_tasks_completed=0,
        )
        self.batch_job_progress_gateway.update_progress(owner, job_id, progress)
        while pending_task_ids_set:
            new_results = executor.map(
                self.async_model_endpoint_inference_gateway.get_task, pending_task_ids_set
            )
            has_new_ready_tasks = False
            curr_timestamp = datetime.utcnow()
            terminal_task_states = {TaskStatus.SUCCESS, TaskStatus.FAILURE}
            for r in new_results:
                if r.status in terminal_task_states or curr_timestamp > timeout_timestamp:
                    has_new_ready_tasks = True
                    task_id_to_result[r.task_id] = r
                    pending_task_ids_set.remove(r.task_id)

            if has_new_ready_tasks:
                logger.info(
                    f"Found {len(task_id_to_result)} ready tasks for batch job {job_id}. "
                    f"{len(pending_task_ids_set)} tasks remaining"
                )
                progress = BatchJobProgress(
                    num_tasks_pending=len(pending_task_ids_set),
                    num_tasks_completed=len(task_id_to_result),
                )
                self.batch_job_progress_gateway.update_progress(owner, job_id, progress)

        results = [
            BatchEndpointInferencePredictionResponse(
                response=task_id_to_result[task_id], reference_id=task_id_to_ref_id_map[task_id]
            )
            for task_id in task_ids_only
        ]
        return results

    def _serialize_and_write_results(
        self,
        result_location: str,
        serialization_format: BatchJobSerializationFormat,
        results: List[BatchEndpointInferencePredictionResponse],
    ) -> None:
        # Write results to the output location.
        results_serialized: Union[str, bytes]
        if serialization_format == BatchJobSerializationFormat.JSON:
            results_serialized = "\n".join([result.json() for result in results]).encode()
        else:  # serialization_format = BatchJobSerializationFormat.PICKLE
            results_serialized = pickle.dumps(results)

        with self.filesystem_gateway.open(
            result_location, "wb", aws_profile=infra_config().profile_ml_worker
        ) as f:
            f.write(results_serialized)

    @staticmethod
    def _get_pending_task_ids_location(job_id: str) -> str:
        return f"s3://{infra_config().s3_bucket}/launch/batch-jobs/{job_id}/pending_task_ids.txt"

    @staticmethod
    def _get_job_result_location(job_id: str) -> str:
        return f"s3://{infra_config().s3_bucket}/launch/batch-jobs/{job_id}/result.json"
