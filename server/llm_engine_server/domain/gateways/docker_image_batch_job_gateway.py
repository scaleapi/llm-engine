from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from llm_engine_server.common.dtos.batch_jobs import CreateDockerImageBatchJobResourceRequests
from llm_engine_server.domain.entities.batch_job_entity import DockerImageBatchJob


class DockerImageBatchJobGateway(ABC):
    """
    Base class for docker image batch job gateway

    """

    @abstractmethod
    async def create_docker_image_batch_job(
        self,
        *,
        created_by: str,
        owner: str,
        job_config: Optional[Dict[str, Any]],
        env: Optional[Dict[str, str]],
        command: List[str],
        repo: str,
        tag: str,
        resource_requests: CreateDockerImageBatchJobResourceRequests,
        labels: Dict[str, str],
        mount_location: Optional[str],
    ) -> str:
        """
        Create a docker image batch job

        Args:
            created_by: The user who created the batch job.
            owner: The user who owns the batch job.
            job_config: The user-specified input to the batch job. Exposed as a file mounted at mount_location to the batch job
            env: Optional list of environment variables for the batch job
            command: The command to the docker image running the batch job
            repo: The ECR repo where the docker image running the batch job lies
            tag: The tag of the docker image
            labels: K8s team/product labels
            resource_requests: The resource requests for the batch job.
            mount_location: Location on filesystem where runtime-provided file contents get mounted


        Returns:
            The ID of the batch job.
        """
        pass

    @abstractmethod
    async def get_docker_image_batch_job(self, batch_job_id: str) -> Optional[DockerImageBatchJob]:
        """
        Get a docker image batch job.
        Args:
            batch_job_id: The ID of the batch job.

        Returns:
            The batch job, or None if it does not exist.
        """
        pass

    @abstractmethod
    async def update_docker_image_batch_job(self, batch_job_id: str, cancel: bool) -> bool:
        """
        Update a batch job.

        Args:
            batch_job_id: The ID of the batch job.
            cancel: Whether to cancel the batch job.

        Returns:
            Whether the batch job was updated successfully.

        """
        pass
