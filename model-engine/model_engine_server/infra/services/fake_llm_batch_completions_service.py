from typing import Dict, Optional

from model_engine_server.common.dtos.batch_jobs import CreateDockerImageBatchJobResourceRequests
from model_engine_server.common.dtos.llms.batch_completion import (
    BatchCompletionsJob,
    CreateBatchCompletionsEngineRequest,
    UpdateBatchCompletionsV2Request,
)
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.domain.services.llm_batch_completions_service import (
    LLMBatchCompletionsService,
)


class FakeLLMBatchCompletionsService(LLMBatchCompletionsService):
    def __init__(
        self,
    ):
        self.jobs = []

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
        raise NotImplementedError()

    async def get_batch_job(self, batch_job_id: str, user: User) -> Optional[BatchCompletionsJob]:
        """
        Get a batch job.

        Args:
            batch_job_id: The ID of the batch job.

        Returns:
            The batch job, or None if it does not exist.
        """
        raise NotImplementedError()

    async def update_batch_job(
        self, batch_job_id: str, request: UpdateBatchCompletionsV2Request, user: User
    ) -> Optional[BatchCompletionsJob]:
        """
        Get a batch job.

        Args:
            batch_job_id: The ID of the batch job.

        Returns:
            The batch job, or None if it does not exist.
        """
        raise NotImplementedError()

    async def cancel_batch_job(self, batch_job_id: str, user: User) -> bool:
        """
        Update a batch job.

        Args:
            batch_job_id: The ID of the batch job.

        Returns:
            Whether the batch job was updated successfully.
        """
        return False
