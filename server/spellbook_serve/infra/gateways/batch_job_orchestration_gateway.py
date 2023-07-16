from abc import ABC, abstractmethod
from typing import Dict

from spellbook_serve.domain.entities import BatchJobSerializationFormat


class BatchJobOrchestrationGateway(ABC):
    """
    Abstract base class for gateways that manage batch job orchestrators.
    """

    @abstractmethod
    async def create_batch_job_orchestrator(
        self,
        job_id: str,
        resource_group_name: str,
        owner: str,
        input_path: str,
        serialization_format: BatchJobSerializationFormat,
        labels: Dict[str, str],
        timeout_seconds: float,
    ) -> None:
        """
        Start a running batch job orchestrator.

        Args:
            job_id: The ID of the batch job.
            resource_group_name: The name of the resource group of the batch job (e.g. k8s job name)
            owner: The ID of the user who owns the batch job.
            input_path: The path to the input data.
            serialization_format: The serialization format of the input data.
            labels: Labels to apply to the batch job.
            timeout_seconds: The timeout in seconds.
        """

    @abstractmethod
    async def delete_batch_job_orchestrator(self, resource_group_name: str) -> bool:
        """
        Delete a batch job orchestrator.

        Args:
            resource_group_name: The name of the resource group of the batch job (e.g. k8s job name)

        Returns:
            Whether the batch job orchestrator was successfully deleted.
        """
