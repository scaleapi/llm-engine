import os
from typing import Dict, List, Optional

from llm_engine_server.common.dtos.batch_jobs import CreateDockerImageBatchJobResourceRequests
from llm_engine_server.domain.entities.batch_job_entity import DockerImageBatchJob
from llm_engine_server.domain.exceptions import (
    InvalidRequestException,
    LLMFineTuningMethodNotImplementedException,
)
from llm_engine_server.domain.gateways.docker_image_batch_job_gateway import (
    DockerImageBatchJobGateway,
)
from llm_engine_server.domain.repositories.docker_image_batch_job_bundle_repository import (
    DockerImageBatchJobBundleRepository,
)
from llm_engine_server.domain.services.llm_fine_tuning_service import LLMFineTuningService
from llm_engine_server.infra.repositories.llm_fine_tuning_job_repository import (
    LLMFineTuningJobRepository,
)


class DockerImageBatchJobLLMFineTuningService(LLMFineTuningService):
    def __init__(
        self,
        docker_image_batch_job_gateway: DockerImageBatchJobGateway,
        docker_image_batch_job_bundle_repo: DockerImageBatchJobBundleRepository,
        llm_fine_tuning_job_repository: LLMFineTuningJobRepository,
    ):
        self.docker_image_batch_job_gateway = docker_image_batch_job_gateway
        self.docker_image_batch_job_bundle_repo = docker_image_batch_job_bundle_repo
        self.llm_fine_tuning_job_repository = llm_fine_tuning_job_repository

    async def create_fine_tune_job(
        self,
        created_by: str,
        owner: str,
        training_file: str,
        validation_file: str,
        model_name: str,
        base_model: str,
        fine_tuning_method: str,
        hyperparameters: Dict[str, str],
    ) -> str:
        batch_job_template = await self.llm_fine_tuning_job_repository.get_job_template_for_model(
            model_name=base_model, fine_tuning_method=fine_tuning_method
        )
        if batch_job_template is None:
            raise LLMFineTuningMethodNotImplementedException(
                f"Fine-tuning not implemented for the (base model, fine-tuning method) pairing of ({base_model}, {fine_tuning_method})"
            )

        for param in batch_job_template.required_params:
            if param not in hyperparameters:
                raise InvalidRequestException(
                    f"Required param {param} is missing from hyperparameters"
                )
        combined_hyperparameters = {
            **batch_job_template.default_hparams,
            **hyperparameters,
        }

        di_batch_job_bundle = (
            await self.docker_image_batch_job_bundle_repo.get_docker_image_batch_job_bundle(
                docker_image_batch_job_bundle_id=batch_job_template.docker_image_batch_job_bundle_id
            )
        )

        if di_batch_job_bundle is None:
            raise LLMFineTuningMethodNotImplementedException("Fine-tuning job doesn't exist")

        if not di_batch_job_bundle.public and di_batch_job_bundle.owner != owner:
            raise LLMFineTuningMethodNotImplementedException("Fine-tuning job not accessible")

        batch_job_id = await self.docker_image_batch_job_gateway.create_docker_image_batch_job(
            created_by=created_by,
            owner=owner,
            job_config=dict(
                gateway_url=os.getenv("GATEWAY_URL"),
                user_id=owner,
                training_file=training_file,
                validation_file=validation_file,
                model_name=model_name,
                launch_bundle_config=batch_job_template.launch_bundle_config,
                launch_endpoint_config=batch_job_template.launch_endpoint_config,
                hyperparameters=combined_hyperparameters,
            ),
            env=di_batch_job_bundle.env,
            command=di_batch_job_bundle.command,
            repo=di_batch_job_bundle.image_repository,
            tag=di_batch_job_bundle.image_tag,
            resource_requests=CreateDockerImageBatchJobResourceRequests(
                cpus=di_batch_job_bundle.cpus,
                memory=di_batch_job_bundle.memory,
                gpus=di_batch_job_bundle.gpus,
                gpu_type=di_batch_job_bundle.gpu_type,
                storage=di_batch_job_bundle.storage,
            ),
            labels=dict(team="infra", product="llm-fine-tuning"),
            mount_location=di_batch_job_bundle.mount_location,
        )

        return batch_job_id

    async def get_fine_tune_job(
        self, owner: str, fine_tune_id: str
    ) -> Optional[DockerImageBatchJob]:
        di_batch_job = await self.docker_image_batch_job_gateway.get_docker_image_batch_job(
            batch_job_id=fine_tune_id
        )
        if di_batch_job is None or di_batch_job.owner != owner:
            return None
        return di_batch_job

    async def list_fine_tune_jobs(self, owner: str) -> List[DockerImageBatchJob]:
        di_batch_jobs = await self.docker_image_batch_job_gateway.list_docker_image_batch_jobs(
            owner=owner
        )
        return di_batch_jobs

    async def cancel_fine_tune_job(self, owner: str, fine_tune_id: str) -> bool:
        di_batch_job = self.get_fine_tune_job(owner, fine_tune_id)
        if di_batch_job is None:
            return False
        cancel = await self.docker_image_batch_job_gateway.update_docker_image_batch_job(
            batch_job_id=fine_tune_id, cancel=True
        )
        return cancel
