from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

from model_engine_server.common.dtos.batch_jobs import CreateDockerImageBatchJobResourceRequests
from model_engine_server.common.dtos.llms import CreateBatchCompletionsEngineRequest
from model_engine_server.common.dtos.llms.batch_completion import BatchCompletionsJob
from model_engine_server.core.auth.authentication_repository import User


@dataclass
class ImageArgs:
    repo: str
    tag: str
    command: List[str]
    env: Optional[Dict[str, str]] = None
    """
    Arguments for a docker image execution

    Args:
        repo: The docker repo where the image is stored
        tag: The tag of the batch completions image
        command: The command to run in the image
        env: Optional environment variables to set in the image

    """


class LLMBatchCompletionsService(ABC):
    """
    Base class for LLM batch completions services.
    """

    @abstractmethod
    async def create_batch_job(
        self,
        *,
        user: User,
        image_repo: str,
        image_tag: str,
        job_request: CreateBatchCompletionsEngineRequest,
        resource_requests: CreateDockerImageBatchJobResourceRequests,
        max_runtime_sec: int = 24 * 60 * 60,
        labels: Dict[str, str] = {},
        priority: Optional[int] = 0,
        num_workers: Optional[int] = 1,
    ) -> BatchCompletionsJob:
        """
        Create a batch completion job.

        Args:
            owner: The user who requested the batch job
            image_repo: The docker repo where the image is stored
            image_tag: The tag of the batch completions image
            job_config: The user-specified input to the batch job. Exposed as a file mounted at mount_location to the batch job
            labels: Labels to apply to the batch job.
            resource_requests: The resource requests for the batch job.
            max_runtime_sec: The timeout of the batch job in seconds.
            num_workers: The number of workers to run in the job.

        Returns:
            The ID of the batch job.
        """
        pass

    @abstractmethod
    async def get_batch_job(self, batch_job_id: str) -> Optional[BatchCompletionsJob]:
        """
        Get a batch job.

        Args:
            batch_job_id: The ID of the batch job.

        Returns:
            The batch job, or None if it does not exist.
        """
        pass

    @abstractmethod
    async def cancel_batch_job(self, batch_job_id: str) -> bool:
        """
        Update a batch job.

        Args:
            batch_job_id: The ID of the batch job.

        Returns:
            Whether the batch job was updated successfully.
        """
        pass
