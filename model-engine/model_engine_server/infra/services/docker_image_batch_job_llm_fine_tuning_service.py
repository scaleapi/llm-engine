import os
from typing import Any, Dict, List, Optional

from model_engine_server.common.dtos.batch_jobs import CreateDockerImageBatchJobResourceRequests
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.entities import FineTuneHparamValueType
from model_engine_server.domain.entities.batch_job_entity import DockerImageBatchJob
from model_engine_server.domain.exceptions import (
    InvalidRequestException,
    LLMFineTuningMethodNotImplementedException,
)
from model_engine_server.domain.gateways.docker_image_batch_job_gateway import (
    DockerImageBatchJobGateway,
)
from model_engine_server.domain.repositories.docker_image_batch_job_bundle_repository import (
    DockerImageBatchJobBundleRepository,
)
from model_engine_server.domain.services import LLMFineTuningService
from model_engine_server.infra.repositories.llm_fine_tune_repository import LLMFineTuneRepository

logger = make_logger(logger_name())


class DockerImageBatchJobLLMFineTuningService(LLMFineTuningService):
    def __init__(
        self,
        docker_image_batch_job_gateway: DockerImageBatchJobGateway,
        docker_image_batch_job_bundle_repo: DockerImageBatchJobBundleRepository,
        llm_fine_tune_repository: LLMFineTuneRepository,
    ):
        self.docker_image_batch_job_gateway = docker_image_batch_job_gateway
        self.docker_image_batch_job_bundle_repo = docker_image_batch_job_bundle_repo
        self.llm_fine_tune_repository = llm_fine_tune_repository

    async def create_fine_tune(
        self,
        created_by: str,
        owner: str,
        model: str,
        training_file: str,
        validation_file: Optional[str],
        fine_tuning_method: str,
        hyperparameters: Dict[str, FineTuneHparamValueType],
        fine_tuned_model: str,
        wandb_config: Optional[Dict[str, Any]],
    ) -> str:
        # fine_tuned_model must be a valid k8s label. Leaky implementation detail unfortunately.
        batch_job_template = await self.llm_fine_tune_repository.get_job_template_for_model(
            model_name=model, fine_tuning_method=fine_tuning_method
        )
        if batch_job_template is None:
            raise LLMFineTuningMethodNotImplementedException(
                f"Fine-tuning not implemented for model type {model}"
                # f"Fine-tuning not implemented for the (base model, fine-tuning method) pairing of ({base_model}, {fine_tuning_method})"
            )  # TODO uncomment out error when we support multiple fine tuning methods

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
            raise LLMFineTuningMethodNotImplementedException("Fine-tuning method doesn't exist")

        if not di_batch_job_bundle.public and di_batch_job_bundle.owner != owner:
            raise LLMFineTuningMethodNotImplementedException("Fine-tuning method not accessible")

        # TODO: Pass user-defined labels
        labels = dict(team="egp", product="training.llm_engine_fine_tune")

        logger.info(
            f"Using bundle {di_batch_job_bundle.id} for fine-tune job: {di_batch_job_bundle.image_repository=}, {di_batch_job_bundle.image_tag=}"
        )
        batch_job_id = await self.docker_image_batch_job_gateway.create_docker_image_batch_job(
            created_by=created_by,
            owner=owner,
            job_config=dict(
                **labels,
                gateway_url=os.getenv("GATEWAY_URL"),
                cloud_provider=infra_config().cloud_provider,
                aws_profile=infra_config().profile_ml_worker,
                s3_bucket=infra_config().s3_bucket,
                azure_client_id=os.getenv("AZURE_CLIENT_ID"),
                abs_account_name=os.getenv("ABS_ACCOUNT_NAME"),
                abs_container_name=os.getenv("ABS_CONTAINER_NAME"),
                user_id=owner,
                training_file=training_file,
                validation_file=validation_file,
                model_name=fine_tuned_model,
                launch_endpoint_config=batch_job_template.launch_endpoint_config,
                hyperparameters=combined_hyperparameters,
                wandb_config=wandb_config,
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
            labels=labels,
            annotations=dict(fine_tuned_model=fine_tuned_model),
            mount_location=di_batch_job_bundle.mount_location,
        )

        return batch_job_id

    async def get_fine_tune(self, owner: str, fine_tune_id: str) -> Optional[DockerImageBatchJob]:
        di_batch_job = await self.docker_image_batch_job_gateway.get_docker_image_batch_job(
            batch_job_id=fine_tune_id
        )
        if di_batch_job is None or di_batch_job.owner != owner:
            return None
        return di_batch_job

    async def list_fine_tunes(self, owner: str) -> List[DockerImageBatchJob]:
        di_batch_jobs = await self.docker_image_batch_job_gateway.list_docker_image_batch_jobs(
            owner=owner
        )
        return di_batch_jobs

    async def cancel_fine_tune(self, owner: str, fine_tune_id: str) -> bool:
        di_batch_job = await self.get_fine_tune(owner, fine_tune_id)
        if di_batch_job is None:
            return False
        cancel = await self.docker_image_batch_job_gateway.update_docker_image_batch_job(
            batch_job_id=fine_tune_id, cancel=True
        )
        return cancel

    async def get_fine_tune_model_name_from_id(
        self, owner: str, fine_tune_id: str
    ) -> Optional[str]:
        di_batch_job = await self.get_fine_tune(owner, fine_tune_id)
        if di_batch_job is None or di_batch_job.annotations is None:
            return None
        return di_batch_job.annotations["fine_tuned_model"]
