from abc import ABC, abstractmethod

from model_engine_server.domain.entities import BatchJobProgress


class BatchJobProgressGateway(ABC):
    """
    Base class for Batch Job Progress gateways.
    """

    @abstractmethod
    def get_progress(self, owner: str, batch_job_id: str) -> BatchJobProgress:
        """
        Get the progress of a batch job.

        Args:
            owner: The user who owns the batch job.
            batch_job_id: The ID of the batch job.

        Returns:
            The progress of the batch job.
        """

    @abstractmethod
    def update_progress(self, owner: str, batch_job_id: str, progress: BatchJobProgress) -> None:
        """
        Update the progress of a batch job.

        Args:
            owner: The user who owns the batch job.
            batch_job_id: The ID of the batch job.
            progress: The progress of the batch job.
        """
