from datetime import datetime, timedelta
from typing import Dict, Optional

from model_engine_server.common.dtos.batch_jobs import CreateDockerImageBatchJobResourceRequests
from model_engine_server.common.dtos.llms import (
    BatchCompletionsJob,
    BatchCompletionsJobStatus,
    CreateBatchCompletionsEngineRequest,
)
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.domain.gateways.docker_image_batch_job_gateway import (
    DockerImageBatchJobGateway,
)
from model_engine_server.domain.services.llm_batch_completions_service import (
    LLMBatchCompletionsService,
)


class LiveLLMBatchCompletionsService(LLMBatchCompletionsService):
    def __init__(
        self,
        docker_image_batch_job_gateway: DockerImageBatchJobGateway,
    ):
        self.docker_image_batch_job_gateway = docker_image_batch_job_gateway

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
        )
        return BatchCompletionsJob(
            job_id=job_id,
            input_data_path=job_request.input_data_path,
            output_data_path=job_request.output_data_path,
            model_config=job_request.model_cfg,
            priority=job_request.priority,
            status=BatchCompletionsJobStatus.Queued,
            created_at=datetime.now().isoformat(),
            expires_at=(datetime.now() + timedelta(seconds=max_runtime_sec)).isoformat(),
            completed_at=None,
            metadata={"labels": job_request.labels},
        )

    async def get_batch_job(self, batch_job_id: str) -> Optional[BatchCompletionsJob]:
        raise NotImplementedError("Not implemented")

    async def cancel_batch_job(self, batch_job_id: str) -> bool:
        # TODO: implement
        raise NotImplementedError("Not implemented")
