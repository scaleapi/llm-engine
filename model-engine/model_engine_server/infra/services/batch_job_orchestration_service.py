from abc import ABC, abstractmethod
from datetime import timedelta

from model_engine_server.domain.entities import BatchJobSerializationFormat


class BatchJobOrchestrationService(ABC):
    """
    Base class for Batch Job Orchestration services.
    """

    @abstractmethod
    async def run_batch_job(
        self,
        *,
        job_id: str,
        owner: str,
        input_path: str,
        serialization_format: BatchJobSerializationFormat,
        timeout: timedelta,
    ) -> None:
        """
        Run a batch job.

        Args:
            job_id: The ID of the batch job.
            owner: The ID of the user who owns the batch job.
            input_path: The path to the input data.
            serialization_format: The serialization format of the input data.
            timeout: The timeout for the batch job.
        """
