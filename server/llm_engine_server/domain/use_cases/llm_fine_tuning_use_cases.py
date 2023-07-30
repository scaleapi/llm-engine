from llm_engine_server.common.dtos.llms import (
    CancelFineTuneJobResponse,
    CreateFineTuneJobRequest,
    CreateFineTuneJobResponse,
    GetFineTuneJobResponse,
    ListFineTuneJobResponse,
)
from llm_engine_server.core.auth.authentication_repository import User
from llm_engine_server.core.domain_exceptions import ObjectNotFoundException
from llm_engine_server.infra.services import DockerImageBatchJobLLMFineTuningService


class CreateFineTuneJobV1UseCase:
    def __init__(self, llm_fine_tuning_service: DockerImageBatchJobLLMFineTuningService):
        self.llm_fine_tuning_service = llm_fine_tuning_service

    async def execute(
        self, user: User, request: CreateFineTuneJobRequest
    ) -> CreateFineTuneJobResponse:
        fine_tune_id = await self.llm_fine_tuning_service.create_fine_tune_job(
            created_by=user.user_id,
            owner=user.team_id,
            training_file=request.training_file,
            validation_file=request.validation_file,
            model_name=request.model_name,
            base_model=request.base_model,
            fine_tuning_method=request.fine_tuning_method,
            hyperparameters=request.hyperparameters,
        )
        return CreateFineTuneJobResponse(
            id=fine_tune_id,
        )


class GetFineTuneJobV1UseCase:
    def __init__(self, llm_fine_tuning_service: DockerImageBatchJobLLMFineTuningService):
        self.llm_fine_tuning_service = llm_fine_tuning_service

    async def execute(self, user: User, fine_tune_id: str) -> GetFineTuneJobResponse:
        di_batch_job = await self.llm_fine_tuning_service.get_fine_tune_job(
            owner=user.team_id,
            fine_tune_id=fine_tune_id,
        )
        if di_batch_job is None:
            raise ObjectNotFoundException
        return GetFineTuneJobResponse(
            id=fine_tune_id,
            status=di_batch_job.status,
        )


class ListFineTuneJobV1UseCase:
    def __init__(self, llm_fine_tuning_service: DockerImageBatchJobLLMFineTuningService):
        self.llm_fine_tuning_service = llm_fine_tuning_service

    async def execute(self, user: User) -> ListFineTuneJobResponse:
        di_batch_jobs = await self.llm_fine_tuning_service.list_fine_tune_jobs(
            owner=user.team_id,
        )
        return ListFineTuneJobResponse(
            jobs=[
                GetFineTuneJobResponse(
                    id==job.id,
                    status=job.status,
                )
                for job in di_batch_jobs
            ]
        )


class CancelFineTuneJobV1UseCase:
    def __init__(self, llm_fine_tuning_service: DockerImageBatchJobLLMFineTuningService):
        self.llm_fine_tuning_service = llm_fine_tuning_service

    async def execute(self, user: User, fine_tune_id: str) -> CancelFineTuneJobResponse:
        success = await self.llm_fine_tuning_service.cancel_fine_tune_job(
            owner=user.team_id,
            fine_tune_id=fine_tune_id,
        )
        return CancelFineTuneJobResponse(
            success=success,
        )
