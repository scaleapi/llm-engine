from typing import Optional

from model_engine_server.common.dtos.batch_jobs import (
    CreateDockerImageBatchJobBundleV1Request,
    CreateDockerImageBatchJobBundleV1Response,
    DockerImageBatchJobBundleV1Response,
    ListDockerImageBatchJobBundleV1Response,
)
from model_engine_server.common.dtos.model_bundles import ModelBundleOrderBy
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.domain.authorization.live_authorization_module import (
    LiveAuthorizationModule,
)
from model_engine_server.domain.exceptions import (
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
)
from model_engine_server.domain.repositories import DockerImageBatchJobBundleRepository


class CreateDockerImageBatchJobBundleV1UseCase:
    """Use case for creating a Docker Image Batch Job Bundle"""

    def __init__(self, docker_image_batch_job_bundle_repo: DockerImageBatchJobBundleRepository):
        self.docker_image_batch_job_bundle_repo = docker_image_batch_job_bundle_repo

    async def execute(
        self, user: User, request: CreateDockerImageBatchJobBundleV1Request
    ) -> CreateDockerImageBatchJobBundleV1Response:
        # TODO should we verify gpu_type, cpus, memory, gpus compatibility?

        # type conversions

        # cast cpu, memory, storage to Optional[str] for db storage.
        # This shouldn't break compatibility with k8s
        if request.resource_requests.cpus is not None:
            request.resource_requests.cpus = str(request.resource_requests.cpus)
        if request.resource_requests.memory is not None:
            # TODO floats probably break k8s,
            #   underlying type might be too permissive
            request.resource_requests.memory = str(request.resource_requests.memory)
        if request.resource_requests.storage is not None:
            request.resource_requests.storage = str(request.resource_requests.storage)

        batch_bundle = (
            await self.docker_image_batch_job_bundle_repo.create_docker_image_batch_job_bundle(
                name=request.name,
                created_by=user.user_id,
                owner=user.team_id,
                image_repository=request.image_repository,
                image_tag=request.image_tag,
                command=request.command,
                env=request.env,
                mount_location=request.mount_location,
                cpus=request.resource_requests.cpus,
                memory=request.resource_requests.memory,
                storage=request.resource_requests.storage,
                gpus=request.resource_requests.gpus,
                gpu_type=request.resource_requests.gpu_type,
                public=request.public,
            )
        )
        return CreateDockerImageBatchJobBundleV1Response(
            docker_image_batch_job_bundle_id=batch_bundle.id
        )


class ListDockerImageBatchJobBundleV1UseCase:
    def __init__(self, docker_image_batch_job_bundle_repo: DockerImageBatchJobBundleRepository):
        self.docker_image_batch_job_bundle_repo = docker_image_batch_job_bundle_repo

    async def execute(
        self, user: User, bundle_name: Optional[str], order_by: Optional[ModelBundleOrderBy]
    ) -> ListDockerImageBatchJobBundleV1Response:
        batch_bundles = (
            await self.docker_image_batch_job_bundle_repo.list_docker_image_batch_job_bundles(
                owner=user.team_id, name=bundle_name, order_by=order_by
            )
        )
        return ListDockerImageBatchJobBundleV1Response(
            docker_image_batch_job_bundles=[
                DockerImageBatchJobBundleV1Response.from_orm(batch_bundle)
                for batch_bundle in batch_bundles
            ]
        )


class GetDockerImageBatchJobBundleByIdV1UseCase:
    def __init__(self, docker_image_batch_job_bundle_repo: DockerImageBatchJobBundleRepository):
        self.docker_image_batch_job_bundle_repo = docker_image_batch_job_bundle_repo
        self.authz_module = LiveAuthorizationModule()

    async def execute(
        self, user: User, docker_image_batch_job_bundle_id: str
    ) -> DockerImageBatchJobBundleV1Response:
        batch_bundle = (
            await self.docker_image_batch_job_bundle_repo.get_docker_image_batch_job_bundle(
                docker_image_batch_job_bundle_id=docker_image_batch_job_bundle_id
            )
        )
        if batch_bundle is None:
            raise ObjectNotFoundException
        if not self.authz_module.check_access_read_owned_entity(user, batch_bundle):
            raise ObjectNotAuthorizedException

        return DockerImageBatchJobBundleV1Response.from_orm(batch_bundle)


class GetLatestDockerImageBatchJobBundleByNameV1UseCase:
    def __init__(self, docker_image_batch_job_bundle_repo: DockerImageBatchJobBundleRepository):
        self.docker_image_batch_job_bundle_repo = docker_image_batch_job_bundle_repo

    async def execute(self, user: User, bundle_name: str) -> DockerImageBatchJobBundleV1Response:
        batch_bundle = (
            await self.docker_image_batch_job_bundle_repo.get_latest_docker_image_batch_job_bundle(
                owner=user.team_id, name=bundle_name
            )
        )
        if batch_bundle is None:
            raise ObjectNotFoundException
        return DockerImageBatchJobBundleV1Response.from_orm(batch_bundle)
