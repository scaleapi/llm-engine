from datetime import datetime
from typing import Optional

from model_engine_server.common.dtos.batch_jobs import (
    CreateBatchJobV1Request,
    CreateBatchJobV1Response,
    CreateDockerImageBatchJobResourceRequests,
    CreateDockerImageBatchJobV1Request,
    CreateDockerImageBatchJobV1Response,
    GetBatchJobV1Response,
    GetDockerImageBatchJobV1Response,
    ListDockerImageBatchJobsV1Response,
    UpdateBatchJobV1Request,
    UpdateBatchJobV1Response,
    UpdateDockerImageBatchJobV1Request,
    UpdateDockerImageBatchJobV1Response,
)
from model_engine_server.common.resource_limits import validate_resource_requests
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.core.loggers import filename_wo_ext, make_logger
from model_engine_server.domain.authorization.live_authorization_module import (
    LiveAuthorizationModule,
)
from model_engine_server.domain.entities import ModelEndpointType
from model_engine_server.domain.exceptions import (
    DockerImageNotFoundException,
    ObjectHasInvalidValueException,
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
)
from model_engine_server.domain.gateways import CronJobGateway, DockerImageBatchJobGateway
from model_engine_server.domain.repositories import (
    DockerImageBatchJobBundleRepository,
    DockerRepository,
    ModelBundleRepository,
    TriggerRepository,
)
from model_engine_server.domain.services import BatchJobService, ModelEndpointService
from model_engine_server.domain.use_cases.model_endpoint_use_cases import (
    validate_deployment_resources,
    validate_labels,
)

logger = make_logger(filename_wo_ext(__file__))


class CreateBatchJobV1UseCase:
    """
    Use case for creating a batch job.
    """

    def __init__(
        self,
        model_bundle_repository: ModelBundleRepository,
        model_endpoint_service: ModelEndpointService,
        batch_job_service: BatchJobService,
    ):
        self.batch_job_service = batch_job_service
        self.model_bundle_repository = model_bundle_repository
        self.model_endpoint_service = model_endpoint_service
        self.authz_module = LiveAuthorizationModule()

    async def execute(
        self, user: User, request: CreateBatchJobV1Request
    ) -> CreateBatchJobV1Response:
        validate_labels(request.labels)
        validate_deployment_resources(
            min_workers=0,
            max_workers=request.resource_requests.max_workers,
            endpoint_type=ModelEndpointType.ASYNC,
        )

        bundle = await self.model_bundle_repository.get_model_bundle(
            model_bundle_id=request.model_bundle_id
        )
        if bundle is None:
            raise ObjectNotFoundException
        if not self.authz_module.check_access_read_owned_entity(user, bundle):
            raise ObjectNotAuthorizedException

        validate_resource_requests(
            bundle=bundle,
            cpus=request.resource_requests.cpus,
            memory=request.resource_requests.memory,
            storage=None,
            gpus=request.resource_requests.gpus,
            gpu_type=request.resource_requests.gpu_type,
        )

        aws_role = self.authz_module.get_aws_role_for_user(user)
        results_s3_bucket = self.authz_module.get_s3_bucket_for_user(user)

        batch_job_id = await self.batch_job_service.create_batch_job(
            created_by=user.user_id,
            owner=user.team_id,
            model_bundle_id=request.model_bundle_id,
            input_path=request.input_path,
            serialization_format=request.serialization_format,
            labels=request.labels,
            resource_requests=request.resource_requests,
            aws_role=aws_role,
            results_s3_bucket=results_s3_bucket,
            timeout_seconds=request.timeout_seconds,
        )

        return CreateBatchJobV1Response(job_id=batch_job_id)


class GetBatchJobV1UseCase:
    """
    Use case for getting a batch job.
    """

    def __init__(self, batch_job_service: BatchJobService):
        self.authz_module = LiveAuthorizationModule()
        self.batch_job_service = batch_job_service

    async def execute(self, user: User, batch_job_id: str) -> GetBatchJobV1Response:
        batch_job = await self.batch_job_service.get_batch_job(batch_job_id=batch_job_id)
        if batch_job is None:
            raise ObjectNotFoundException
        record = batch_job.record
        if not self.authz_module.check_access_read_owned_entity(user, record):
            raise ObjectNotAuthorizedException

        progress = batch_job.progress
        if record.completed_at is not None:
            duration = record.completed_at - record.created_at
        else:
            duration = datetime.now(tz=record.created_at.tzinfo) - record.created_at

        response = GetBatchJobV1Response(
            status=record.status,
            result=record.result_location,
            duration=duration,
            num_tasks_pending=progress.num_tasks_pending,
            num_tasks_completed=progress.num_tasks_completed,
        )
        return response


class UpdateBatchJobV1UseCase:
    """
    Use case for cancelling a batch job.
    """

    def __init__(self, batch_job_service: BatchJobService):
        self.batch_job_service = batch_job_service
        self.authz_module = LiveAuthorizationModule()

    async def execute(
        self, user: User, batch_job_id: str, request: UpdateBatchJobV1Request
    ) -> UpdateBatchJobV1Response:
        batch_job = await self.batch_job_service.get_batch_job(batch_job_id=batch_job_id)
        if batch_job is None:
            raise ObjectNotFoundException
        if not self.authz_module.check_access_read_owned_entity(user, batch_job.record):
            raise ObjectNotAuthorizedException

        await self.batch_job_service.update_batch_job(
            batch_job_id=batch_job_id, cancel=request.cancel
        )
        return UpdateBatchJobV1Response(success=True)


class CreateDockerImageBatchJobV1UseCase:
    def __init__(
        self,
        docker_image_batch_job_gateway: DockerImageBatchJobGateway,
        docker_image_batch_job_bundle_repository: DockerImageBatchJobBundleRepository,
        docker_repository: DockerRepository,
    ):
        self.docker_image_batch_job_gateway = docker_image_batch_job_gateway
        self.docker_image_batch_job_bundle_repository = docker_image_batch_job_bundle_repository
        self.docker_repository = docker_repository
        self.authz_module = LiveAuthorizationModule()

    async def execute(
        self, user: User, request: CreateDockerImageBatchJobV1Request
    ) -> CreateDockerImageBatchJobV1Response:
        if request.docker_image_batch_job_bundle_id is not None:
            batch_bundle = await self.docker_image_batch_job_bundle_repository.get_docker_image_batch_job_bundle(
                request.docker_image_batch_job_bundle_id
            )
        else:
            # Given that ...bundle_id is none,
            #   we've validated ...bundle_name isn't none when the request was constructed.
            # Following block is for mypy, and won't get executed as long as validation is done.
            if request.docker_image_batch_job_bundle_name is None:
                raise ObjectHasInvalidValueException("Please specify one of bundle id and name")
            batch_bundle = await self.docker_image_batch_job_bundle_repository.get_latest_docker_image_batch_job_bundle(
                user.team_id, request.docker_image_batch_job_bundle_name
            )

        if batch_bundle is None:
            raise ObjectNotFoundException("The specified batch job bundle could not be found")
        if not self.authz_module.check_access_read_owned_entity(user, batch_bundle):
            raise ObjectNotAuthorizedException(
                f"User {user} does not have permission for the specified batch job bundle"
            )

        if not self.docker_repository.image_exists(
            image_tag=batch_bundle.image_tag, repository_name=batch_bundle.image_repository
        ):
            raise DockerImageNotFoundException(
                repository=batch_bundle.image_repository,
                tag=batch_bundle.image_tag,
            )  # Error if docker image could not be found either

        original_requests = CreateDockerImageBatchJobResourceRequests(
            cpus=batch_bundle.cpus,
            memory=batch_bundle.memory,
            gpus=batch_bundle.gpus,
            gpu_type=batch_bundle.gpu_type,
            storage=batch_bundle.storage,
        )
        final_requests = CreateDockerImageBatchJobResourceRequests.merge_requests(
            original_requests, request.resource_requests
        )
        overridden_parameters = CreateDockerImageBatchJobResourceRequests.common_requests(
            original_requests, request.resource_requests
        )
        logger.info(f"Common parameters: {overridden_parameters}")

        # to override a default gpu instance to a cpu instance, can specify gpus=0
        # have to do this since a not-none gpu type won't get overridden by a None gpu type
        if final_requests.gpus == 0:
            final_requests.gpus = None
            final_requests.gpu_type = None

        # check resources exist
        if None in [final_requests.cpus, final_requests.memory]:
            raise ObjectHasInvalidValueException("Must specify value for cpus and memory")
        # check they're valid for the cluster also
        validate_resource_requests(
            bundle=batch_bundle,
            cpus=final_requests.cpus,
            memory=final_requests.memory,
            storage=final_requests.storage,
            gpus=final_requests.gpus,
            gpu_type=final_requests.gpu_type,
        )

        validate_labels(request.labels)

        if (
            request.override_job_max_runtime_s is not None
            and request.override_job_max_runtime_s < 1
        ):
            raise ObjectHasInvalidValueException(
                "Please supply a positive integer value for batch job's maximum runtime (`override_job_max_runtime_s`)"
            )

        job_id = await self.docker_image_batch_job_gateway.create_docker_image_batch_job(
            created_by=user.user_id,
            owner=user.team_id,
            job_config=request.job_config,
            env=batch_bundle.env,
            command=batch_bundle.command,
            repo=batch_bundle.image_repository,
            tag=batch_bundle.image_tag,
            resource_requests=final_requests,
            labels=request.labels,
            mount_location=batch_bundle.mount_location,
            override_job_max_runtime_s=request.override_job_max_runtime_s,
        )
        return CreateDockerImageBatchJobV1Response(job_id=job_id)


class GetDockerImageBatchJobV1UseCase:
    """
    Use case for getting a batch job.
    """

    def __init__(self, docker_image_batch_job_gateway: DockerImageBatchJobGateway):
        self.docker_image_batch_job_gateway = docker_image_batch_job_gateway

    async def execute(self, user: User, batch_job_id: str) -> GetDockerImageBatchJobV1Response:
        # ids get validated inside the gateway since that's where they get created
        job = await self.docker_image_batch_job_gateway.get_docker_image_batch_job(
            batch_job_id=batch_job_id
        )
        if job is None:
            raise ObjectNotFoundException(
                f"The specified batch job {batch_job_id} could not be found"
            )
        if job.owner != user.team_id:
            raise ObjectNotAuthorizedException(
                f"User {user} is not authorized for batch job {batch_job_id}"
            )  # Report job isn't owned, but API layer returns 404
        return GetDockerImageBatchJobV1Response(status=job.status)


class ListDockerImageBatchJobsV1UseCase:
    def __init__(
        self,
        trigger_repository: TriggerRepository,
        cron_job_gateway: CronJobGateway,
    ):
        self.trigger_repository = trigger_repository
        self.cron_job_gateway = cron_job_gateway
        self.authz_module = LiveAuthorizationModule()

    async def execute(
        self, user: User, trigger_id: Optional[str]
    ) -> ListDockerImageBatchJobsV1Response:
        if trigger_id:
            trigger = await self.trigger_repository.get_trigger(trigger_id=trigger_id)
            if trigger is None:
                raise ObjectNotFoundException
            if not self.authz_module.check_access_read_owned_entity(user, trigger):
                raise ObjectNotAuthorizedException(
                    f"User {user} is not authorized for trigger {trigger_id}"
                )

        jobs = await self.cron_job_gateway.list_jobs(owner=user.team_id, trigger_id=trigger_id)
        return ListDockerImageBatchJobsV1Response(jobs=jobs)


class UpdateDockerImageBatchJobV1UseCase:
    """
    Use case for cancelling a batch job.
    """

    def __init__(self, docker_image_batch_job_gateway: DockerImageBatchJobGateway):
        self.docker_image_batch_job_gateway = docker_image_batch_job_gateway

    async def execute(
        self, user: User, batch_job_id: str, request: UpdateDockerImageBatchJobV1Request
    ) -> UpdateDockerImageBatchJobV1Response:
        job = await self.docker_image_batch_job_gateway.get_docker_image_batch_job(
            batch_job_id=batch_job_id
        )
        if job is None:
            raise ObjectNotFoundException(
                f"The specified batch job {batch_job_id} could not be found"
            )
        if job.owner != user.team_id:
            raise ObjectNotAuthorizedException(
                f"User {user} is not authorized for batch job {batch_job_id}"
            )  # Report job isn't owned, but API layer returns 404
        response = await self.docker_image_batch_job_gateway.update_docker_image_batch_job(
            batch_job_id=batch_job_id, cancel=request.cancel
        )
        return UpdateDockerImageBatchJobV1Response(success=response)
