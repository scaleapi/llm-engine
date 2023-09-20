import os

from croniter import croniter
from model_engine_server.common.dtos.triggers import (
    CreateTriggerV1Request,
    CreateTriggerV1Response,
    DeleteTriggerV1Response,
    GetTriggerV1Response,
    ListTriggersV1Response,
    UpdateTriggerV1Request,
    UpdateTriggerV1Response,
)
from model_engine_server.common.resource_limits import validate_resource_requests
from model_engine_server.common.settings import REQUIRED_ENDPOINT_LABELS
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.core.config import infra_config
from model_engine_server.domain.authorization.live_authorization_module import (
    LiveAuthorizationModule,
)
from model_engine_server.domain.exceptions import (
    CronSyntaxException,
    DockerImageNotFoundException,
    EndpointLabelsException,
    ObjectHasInvalidValueException,
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
)
from model_engine_server.domain.gateways.cron_job_gateway import CronJobGateway
from model_engine_server.domain.repositories import (
    DockerImageBatchJobBundleRepository,
    DockerRepository,
    TriggerRepository,
)
from model_engine_server.domain.use_cases.model_endpoint_use_cases import validate_labels

DEFAULT_HOST = f"https://model-engine.{infra_config().dns_host_domain}"

ALLOWED_CRON_MACROS = set(
    [
        "@yearly",
        "@annually",
        "@monthly",
        "@weekly",
        "@daily",
        "@midnight",
        "@hourly",
    ]
)


def validate_cron(cron: str) -> None:
    if len(cron) == 0:
        raise CronSyntaxException("Cron expression cannot be empty.")

    if cron not in ALLOWED_CRON_MACROS:
        # case on presence of macro identifier
        if cron[0] == "@":
            raise CronSyntaxException(
                f"Unsupported macro supplied: '{cron}'. Please select from the following list, {ALLOWED_CRON_MACROS}."
            )
        elif not croniter.is_valid(cron):
            raise CronSyntaxException(
                f"Invalid Cron syntax: '{cron}'. Please see https://crontab.guru."
            )


class CreateTriggerUseCase:
    """Use case for creating a Trigger"""

    def __init__(
        self,
        trigger_repository: TriggerRepository,
        cron_job_gateway: CronJobGateway,
        docker_image_batch_job_bundle_repository: DockerImageBatchJobBundleRepository,
        docker_repository: DockerRepository,
    ):
        self.trigger_repository = trigger_repository
        self.cron_job_gateway = cron_job_gateway
        self.docker_image_batch_job_bundle_repository = docker_image_batch_job_bundle_repository
        self.docker_repository = docker_repository
        self.authz_module = LiveAuthorizationModule()

    async def execute(
        self,
        user: User,
        request: CreateTriggerV1Request,
    ) -> CreateTriggerV1Response:
        batch_bundle = (
            await self.docker_image_batch_job_bundle_repository.get_docker_image_batch_job_bundle(
                request.bundle_id
            )
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

        # check if required resources exist
        if None in [batch_bundle.cpus, batch_bundle.memory]:
            raise ObjectHasInvalidValueException("Bundle must specify value for cpus and memory")
        # validate resource request in cluster also
        validate_resource_requests(
            bundle=batch_bundle,
            cpus=batch_bundle.cpus,
            memory=batch_bundle.memory,
            storage=batch_bundle.storage,
            gpus=batch_bundle.gpus,
            gpu_type=batch_bundle.gpu_type,
        )

        if request.default_job_metadata is None:
            raise EndpointLabelsException(
                f"Missing labels in default_job_metadata. These are all required: {REQUIRED_ENDPOINT_LABELS}"
            )

        validate_labels(request.default_job_metadata)
        validate_cron(request.cron_schedule)

        trigger = await self.trigger_repository.create_trigger(
            name=request.name,
            created_by=user.user_id,
            owner=user.team_id,
            cron_schedule=request.cron_schedule,
            docker_image_batch_job_bundle_id=request.bundle_id,
            default_job_config=request.default_job_config,
            default_job_metadata=request.default_job_metadata,
        )

        request.default_job_metadata["trigger_id"] = trigger.id
        await self.cron_job_gateway.create_cronjob(
            request_host=os.getenv("GATEWAY_URL") or DEFAULT_HOST,
            trigger_id=trigger.id,
            created_by=user.user_id,
            owner=user.team_id,
            cron_schedule=request.cron_schedule,
            docker_image_batch_job_bundle_id=request.bundle_id,
            default_job_config=request.default_job_config,
            default_job_metadata=request.default_job_metadata,
        )

        return CreateTriggerV1Response(trigger_id=trigger.id)


class ListTriggersUseCase:
    def __init__(self, trigger_repository: TriggerRepository):
        self.trigger_repository = trigger_repository

    async def execute(self, user: User) -> ListTriggersV1Response:
        triggers = await self.trigger_repository.list_triggers(owner=user.team_id)
        return ListTriggersV1Response(
            triggers=[GetTriggerV1Response.from_orm(trigger) for trigger in triggers]
        )


class GetTriggerUseCase:
    def __init__(self, trigger_repository: TriggerRepository):
        self.trigger_repository = trigger_repository
        self.authz_module = LiveAuthorizationModule()

    async def execute(self, user: User, trigger_id: str) -> GetTriggerV1Response:
        trigger = await self.trigger_repository.get_trigger(trigger_id=trigger_id)
        if trigger is None:
            raise ObjectNotFoundException
        if not self.authz_module.check_access_read_owned_entity(user, trigger):
            raise ObjectNotAuthorizedException(
                f"User {user} is not authorized for trigger {trigger_id}"
            )

        return GetTriggerV1Response.from_orm(trigger)


class UpdateTriggerUseCase:
    def __init__(
        self,
        trigger_repository: TriggerRepository,
        cron_job_gateway: CronJobGateway,
    ):
        self.trigger_repository = trigger_repository
        self.cron_job_gateway = cron_job_gateway
        self.authz_module = LiveAuthorizationModule()

    async def execute(
        self, user: User, trigger_id: str, request: UpdateTriggerV1Request
    ) -> UpdateTriggerV1Response:
        trigger = await self.trigger_repository.get_trigger(trigger_id=trigger_id)
        if trigger is None:
            raise ObjectNotFoundException
        if not self.authz_module.check_access_read_owned_entity(user, trigger):
            raise ObjectNotAuthorizedException(
                f"User {user} is not authorized for trigger {trigger_id}"
            )

        success = True
        if request.cron_schedule is not None:
            validate_cron(request.cron_schedule)
            success = await self.trigger_repository.update_trigger(
                trigger_id=trigger_id, cron_schedule=request.cron_schedule
            )

        if success:
            await self.cron_job_gateway.update_cronjob(
                trigger_id=trigger.id,
                cron_schedule=request.cron_schedule,
                suspend=request.suspend,
            )

        return UpdateTriggerV1Response(success=success)


class DeleteTriggerUseCase:
    def __init__(
        self,
        trigger_repository: TriggerRepository,
        cron_job_gateway: CronJobGateway,
    ):
        self.trigger_repository = trigger_repository
        self.cron_job_gateway = cron_job_gateway
        self.authz_module = LiveAuthorizationModule()

    async def execute(self, user: User, trigger_id: str) -> DeleteTriggerV1Response:
        trigger = await self.trigger_repository.get_trigger(trigger_id=trigger_id)
        if trigger is None:
            raise ObjectNotFoundException
        if not self.authz_module.check_access_read_owned_entity(user, trigger):
            raise ObjectNotAuthorizedException(
                f"User {user} is not authorized for trigger {trigger_id}"
            )

        success = await self.trigger_repository.delete_trigger(trigger_id=trigger_id)
        if success:
            await self.cron_job_gateway.delete_cronjob(trigger_id=trigger_id)

        return DeleteTriggerV1Response(success=success)
