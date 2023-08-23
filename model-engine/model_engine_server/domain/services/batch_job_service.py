from abc import ABC, abstractmethod
from typing import Dict, Optional

from model_engine_server.common.dtos.batch_jobs import CreateBatchJobResourceRequests
from model_engine_server.domain.entities import BatchJob, BatchJobSerializationFormat


class BatchJobService(ABC):
    """
    Base class for Batch Job services.
    """

    @abstractmethod
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
        """
        Create a batch job.

        Args:
            created_by: The user who created the batch job.
            owner: The user who owns the batch job.
            model_bundle_id: The ID of the model bundle to use for the batch job.
            input_path: The path to the input data.
            serialization_format: The serialization format of the input data.
            labels: Labels to apply to the batch job.
            resource_requests: The resource requests for the batch job.
            aws_role: The AWS role to use for the batch job.
            results_s3_bucket: The S3 bucket to store results in.
            timeout_seconds: The timeout of the batch job in seconds.

        Returns:
            The ID of the batch job.
        """

    @abstractmethod
    async def get_batch_job(self, batch_job_id: str) -> Optional[BatchJob]:
        """
        Get a batch job.

        Args:
            batch_job_id: The ID of the batch job.

        Returns:
            The batch job, or None if it does not exist.
        """

    @abstractmethod
    async def update_batch_job(self, batch_job_id: str, cancel: bool) -> bool:
        """
        Update a batch job.

        Args:
            batch_job_id: The ID of the batch job.
            cancel: Whether to cancel the batch job.

        Returns:
            Whether the batch job was updated successfully.
        """
