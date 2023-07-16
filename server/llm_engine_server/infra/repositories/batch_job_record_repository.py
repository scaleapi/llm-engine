from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from llm_engine_server.domain.entities import BatchJobRecord, BatchJobStatus


class BatchJobRecordRepository(ABC):
    """
    Base class for Batch Job repositories.
    """

    @abstractmethod
    async def create_batch_job_record(
        self,
        *,
        status: BatchJobStatus,
        created_by: str,
        owner: str,
        model_bundle_id: str,
    ) -> BatchJobRecord:
        """
        Creates an entry for endpoint tracking data, but not the actual compute resources.
        Assumes that the given model_bundle_id corresponds to a valid Model Bundle.

        Args:
            status: Status of batch job
            created_by: User who created batch job
            owner: User who owns batch job
            model_bundle_id: Model Bundle the batch job uses

        Returns:
            A Batch Job domain entity.
        """

    @abstractmethod
    async def unset_model_endpoint_id(self, batch_job_id: str) -> Optional[BatchJobRecord]:
        """
        Unsets the model_endpoint_id for the Batch Job with the given ID.

        Args:
            batch_job_id: The ID of the batch job.

        Returns:
            A Batch Job domain entity if found, else None.
        """

    @abstractmethod
    async def update_batch_job_record(
        self,
        *,
        batch_job_id: str,
        status: Optional[BatchJobStatus] = None,
        model_endpoint_id: Optional[str] = None,
        task_ids_location: Optional[str] = None,
        result_location: Optional[str] = None,
        completed_at: Optional[datetime] = None,
    ) -> Optional[BatchJobRecord]:
        """
        Updates the entry for endpoint tracking data with the given new values. Only these values are editable.

        Args:
            batch_job_id: Unique ID for the batch job to update
            status: Status of batch job
            model_endpoint_id: Unique ID for the model endpoint the batch job uses
            task_ids_location: Location of task ids
            result_location: Location of results
            completed_at: Time the batch job completed

        Returns:
            A Batch Job domain entity if found, else None.
        """

    @abstractmethod
    async def list_batch_job_records(self, owner: Optional[str]) -> List[BatchJobRecord]:
        """
        Lists all the records of batch jobs given the filters.

        Args:
            owner: The user ID of the creator of the endpoints.

        Returns:
            A list of Batch Job domain entities.
        """

    @abstractmethod
    async def get_batch_job_record(self, batch_job_id: str) -> Optional[BatchJobRecord]:
        """
        Gets a batch job record.

        Args:
            batch_job_id: The unique ID of the Batch Job to get.

        Returns:
            A Batch Job domain entity if found, else None.
        """
