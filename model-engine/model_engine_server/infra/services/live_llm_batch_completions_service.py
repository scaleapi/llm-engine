from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional

from model_engine_server.common.dtos.batch_jobs import CreateDockerImageBatchJobResourceRequests
from model_engine_server.common.dtos.llms import (
    BatchCompletionsJob,
    BatchCompletionsJobStatus,
    CreateBatchCompletionsEngineRequest,
)
from model_engine_server.common.dtos.llms.batch_completion import (
    UpdateBatchCompletionsV2Request,
    UpdateBatchCompletionsV2Response,
)
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.domain.entities.batch_job_entity import BatchJobStatus
from model_engine_server.domain.gateways.docker_image_batch_job_gateway import (
    DockerImageBatchJobGateway,
)
from model_engine_server.domain.services.llm_batch_completions_service import (
    LLMBatchCompletionsService,
)


def to_dto(status: BatchJobStatus) -> BatchCompletionsJobStatus:
    if status == BatchJobStatus.PENDING:
        return BatchCompletionsJobStatus.Queued
    if status == BatchJobStatus.RUNNING:
        return BatchCompletionsJobStatus.Running
    if status == BatchJobStatus.FAILURE:
        return BatchCompletionsJobStatus.Failed
    if status == BatchJobStatus.SUCCESS:
        return BatchCompletionsJobStatus.Completed
    if status == BatchJobStatus.CANCELLED:
        return BatchCompletionsJobStatus.Cancelled
    if status == BatchJobStatus.TIMEOUT:
        return BatchCompletionsJobStatus.Failed

    return BatchCompletionsJobStatus.Unknown


@dataclass
class CustomJobMetadata:
    """
    This is a workaround to the current DockerImageBatchJobGateway implementation
    which doesn't store additional metadata we need for batch completions v2
    """

    input_data_path: Optional[str]
    output_data_path: str
    expires_at: str
    priority: Optional[str]
    labels: Dict[str, str]


NULL_TOKEN = "null"


class LiveLLMBatchCompletionsService(LLMBatchCompletionsService):
    def __init__(
        self,
        docker_image_batch_job_gateway: DockerImageBatchJobGateway,
    ):
        self.docker_image_batch_job_gateway = docker_image_batch_job_gateway

    def encode_metadata(self, metadata: CustomJobMetadata) -> Dict[str, str]:
        return {
            "__INT_input_data_path": metadata.input_data_path or NULL_TOKEN,
            "__INT_output_data_path": metadata.output_data_path,
            "__INT_expires_at": metadata.expires_at,
            "__INT_priority": metadata.priority or NULL_TOKEN,
            **{f"__LABEL_{key}": value for key, value in metadata.labels.items()},
        }

    def decode_metadata(self, metadata: Dict[str, str]) -> CustomJobMetadata:
        labels = {
            key.replace("__LABEL_", ""): value
            for key, value in metadata.items()
            if key.startswith("__LABEL")
        }

        return CustomJobMetadata(
            input_data_path=metadata.get("__INT_input_data_path", "unknown"),
            output_data_path=metadata.get("__INT_output_data_path", "unknown"),
            expires_at=metadata.get("__INT_expires_at", "unknown"),
            priority=metadata.get("__INT_priority", "unknown"),
            labels=labels,
        )

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
    ):
        config_file_path = "/opt/config.json"
        env = {"CONFIG_FILE": config_file_path}
        command = [
            "dumb-init",
            "--",
            "/bin/bash",
            "-c",
            "ddtrace-run python vllm_batch.py",
        ]

        expires_at = datetime.now() + timedelta(seconds=max_runtime_sec)
        job_id = await self.docker_image_batch_job_gateway.create_docker_image_batch_job(
            created_by=user.user_id,
            owner=user.team_id,
            job_config=job_request.model_dump(by_alias=True),
            env=env,
            command=command,
            repo=image_repo,
            tag=image_tag,
            mount_location=config_file_path,
            resource_requests=resource_requests,
            labels=labels,
            override_job_max_runtime_s=max_runtime_sec,
            num_workers=num_workers,
            annotations=self.encode_metadata(
                CustomJobMetadata(
                    input_data_path=job_request.input_data_path,
                    output_data_path=job_request.output_data_path,
                    expires_at=expires_at.isoformat(),
                    priority=job_request.priority,
                    labels=job_request.labels,
                )
            ),
        )
        return BatchCompletionsJob(
            job_id=job_id,
            input_data_path=job_request.input_data_path,
            output_data_path=job_request.output_data_path,
            model_config=job_request.model_cfg,
            priority=job_request.priority,
            status=BatchCompletionsJobStatus.Queued,
            created_at=datetime.now().isoformat(),
            expires_at=expires_at.isoformat(),
            completed_at=None,
            metadata={"labels": job_request.labels},
        )

    async def get_batch_job(self, batch_job_id: str, user: User) -> Optional[BatchCompletionsJob]:
        job = await self.docker_image_batch_job_gateway.get_docker_image_batch_job(
            batch_job_id=batch_job_id
        )

        if job is None:
            return None

        custom_metadata = self.decode_metadata(job.annotations or {})
        model_config = "[Cannot retrieve] -- please check the job logs"

        return BatchCompletionsJob(
            job_id=batch_job_id,
            input_data_path=custom_metadata.input_data_path,
            output_data_path=custom_metadata.output_data_path,
            model_config=model_config,
            priority=custom_metadata.priority,
            status=to_dto(job.status),
            created_at=job.created_at,
            expires_at=custom_metadata.expires_at,
            completed_at=job.completed_at,
            metadata={"labels": custom_metadata.labels},
        )

    async def update_batch_job(
        self, batch_job_id: str, request: UpdateBatchCompletionsV2Request, user: User
    ) -> UpdateBatchCompletionsV2Response:
        raise NotImplementedError("Not supported")

    async def cancel_batch_job(self, batch_job_id: str, user: User) -> bool:
        return await self.docker_image_batch_job_gateway.update_docker_image_batch_job(
            batch_job_id=batch_job_id, cancel=True
        )
