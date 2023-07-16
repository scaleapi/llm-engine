"""
TODO figure out how to do: (or if we want to do it)
List model endpoint history: GET model-endpoints/<endpoint id>/history
Read model endpoint creation logs: GET model-endpoints/<endpoint id>/creation-logs
"""

from typing import List, Optional

from llm_engine_server.common.constants import CALLBACK_POST_INFERENCE_HOOK
from llm_engine_server.common.dtos.model_endpoints import (
    CreateModelEndpointV1Request,
    CreateModelEndpointV1Response,
    DeleteModelEndpointV1Response,
    GetModelEndpointV1Response,
    ListModelEndpointsV1Response,
    ModelEndpointOrderBy,
    UpdateModelEndpointV1Request,
    UpdateModelEndpointV1Response,
)
from llm_engine_server.common.resource_limits import MAX_ENDPOINT_SIZE, validate_resource_requests
from llm_engine_server.core.auth.authentication_repository import User
from llm_engine_server.core.domain_exceptions import (
    ObjectHasInvalidValueException,
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
)
from llm_engine_server.core.loggers import filename_wo_ext, make_logger
from llm_engine_server.domain.authorization.scale_authorization_module import ScaleAuthorizationModule
from llm_engine_server.domain.entities import (
    ModelEndpoint,
    ModelEndpointType,
    StreamingEnhancedRunnableImageFlavor,
)
from llm_engine_server.domain.exceptions import (
    EndpointInfraStateNotFound,
    EndpointLabelsException,
    EndpointResourceInvalidRequestException,
)
from llm_engine_server.domain.repositories import ModelBundleRepository
from llm_engine_server.domain.services import ModelEndpointService

CONVERTED_FROM_ARTIFACT_LIKE_KEY = "_CONVERTED_FROM_ARTIFACT_LIKE"

logger = make_logger(filename_wo_ext(__name__))


def model_endpoint_entity_to_get_model_endpoint_response(
    model_endpoint: ModelEndpoint,
) -> GetModelEndpointV1Response:
    infra_state = model_endpoint.infra_state
    endpoint_config = None if infra_state is None else infra_state.user_config_state.endpoint_config
    post_inference_hooks = [] if endpoint_config is None else endpoint_config.post_inference_hooks
    default_callback_url = None if endpoint_config is None else endpoint_config.default_callback_url
    default_callback_auth = (
        None if endpoint_config is None else endpoint_config.default_callback_auth
    )
    return GetModelEndpointV1Response(
        id=model_endpoint.record.id,
        name=model_endpoint.record.name,
        endpoint_type=model_endpoint.record.endpoint_type,
        destination=model_endpoint.record.destination,
        deployment_name=(None if infra_state is None else infra_state.deployment_name),
        metadata=model_endpoint.record.metadata,
        bundle_name=model_endpoint.record.current_model_bundle.name,
        status=model_endpoint.record.status,
        post_inference_hooks=post_inference_hooks,
        default_callback_url=default_callback_url,  # type: ignore
        default_callback_auth=default_callback_auth,
        labels=(None if infra_state is None else infra_state.labels),
        aws_role=(None if infra_state is None else infra_state.aws_role),
        results_s3_bucket=(None if infra_state is None else infra_state.results_s3_bucket),
        created_by=model_endpoint.record.created_by,
        created_at=model_endpoint.record.created_at,
        last_updated_at=model_endpoint.record.last_updated_at,  # type: ignore
        deployment_state=(None if infra_state is None else infra_state.deployment_state),
        resource_state=(None if infra_state is None else infra_state.resource_state),
        num_queued_items=(None if infra_state is None else infra_state.num_queued_items),
        public_inference=model_endpoint.record.public_inference,
    )


def _handle_post_inference_hooks(
    created_by: str,
    name: str,
    post_inference_hooks: Optional[List[str]],
) -> None:
    if not post_inference_hooks:
        return
    for hook in post_inference_hooks:
        hook = hook.lower()


def validate_deployment_resources(
    min_workers: Optional[int],
    max_workers: Optional[int],
    endpoint_type: ModelEndpointType,
) -> None:
    # TODO: we should be also validating the update request against the existing state in k8s (e.g.
    #  so min_workers <= max_workers always) maybe this occurs already in update_model_endpoint.
    min_endpoint_size = 0 if endpoint_type == ModelEndpointType.ASYNC else 1
    if min_workers is not None and min_workers < min_endpoint_size:
        raise EndpointResourceInvalidRequestException(
            f"Requested min workers {min_workers} too low"
        )
    if max_workers is not None and max_workers > MAX_ENDPOINT_SIZE:
        raise EndpointResourceInvalidRequestException(
            f"Requested max workers {max_workers} too high"
        )


def validate_post_inference_hooks(user: User, post_inference_hooks: Optional[List[str]]) -> None:
    # We're going to ask for user-specified auth for callbacks instead of providing default auth
    # from LLMEngine. Otherwise, we'd want to prevent non-privileged users from using the
    # callback post-inference hook.
    if post_inference_hooks is None:
        return

    for hook in post_inference_hooks:
        if hook not in [
            CALLBACK_POST_INFERENCE_HOOK,
        ]:
            raise ValueError(f"Unsupported post-inference hook {hook}")


class CreateModelEndpointV1UseCase:
    def __init__(
        self,
        model_bundle_repository: ModelBundleRepository,
        model_endpoint_service: ModelEndpointService,
    ):
        self.model_bundle_repository = model_bundle_repository
        self.model_endpoint_service = model_endpoint_service
        self.authz_module = ScaleAuthorizationModule()

    async def execute(
        self, user: User, request: CreateModelEndpointV1Request
    ) -> CreateModelEndpointV1Response:
        validate_resource_requests(
            cpus=request.cpus,
            memory=request.memory,
            storage=request.storage,
            gpus=request.gpus,
            gpu_type=request.gpu_type,
        )
        validate_deployment_resources(
            min_workers=request.min_workers,
            max_workers=request.max_workers,
            endpoint_type=request.endpoint_type,
        )
        if request.labels is None:
            raise EndpointLabelsException("Endpoint labels cannot be None!")
        validate_post_inference_hooks(user, request.post_inference_hooks)
        bundle = await self.model_bundle_repository.get_model_bundle(
            model_bundle_id=request.model_bundle_id
        )
        if bundle is None:
            raise ObjectNotFoundException
        if not self.authz_module.check_access_read_owned_entity(user, bundle):
            raise ObjectNotAuthorizedException
        if not isinstance(bundle.flavor, StreamingEnhancedRunnableImageFlavor) and (
            request.endpoint_type == ModelEndpointType.STREAMING
        ):
            raise ObjectHasInvalidValueException(
                "Cannot create a streaming endpoint for a non-streaming model bundle."
            )
        if request.endpoint_type == ModelEndpointType.STREAMING and request.per_worker != 1:
            raise ObjectHasInvalidValueException(
                "Cannot create a streaming endpoint with per_worker != 1."
            )
        if (
            isinstance(bundle.flavor, StreamingEnhancedRunnableImageFlavor)
            and request.endpoint_type in {ModelEndpointType.SYNC, ModelEndpointType.ASYNC}
            and len(bundle.flavor.command) == 0
        ):
            raise ObjectHasInvalidValueException(
                "Cannot create a non-streaming endpoint with no bundle command."
            )
        if CONVERTED_FROM_ARTIFACT_LIKE_KEY in request.metadata:
            raise ObjectHasInvalidValueException(
                f"{CONVERTED_FROM_ARTIFACT_LIKE_KEY} is a reserved metadata key and cannot be used by user."
            )

        prewarm = request.prewarm
        if prewarm is None:
            prewarm = True

        high_priority = request.high_priority
        if high_priority is None:
            high_priority = False

        aws_role = self.authz_module.get_aws_role_for_user(user)
        results_s3_bucket = self.authz_module.get_s3_bucket_for_user(user)

        model_endpoint_record = await self.model_endpoint_service.create_model_endpoint(
            name=request.name,
            created_by=user.user_id,
            model_bundle_id=request.model_bundle_id,
            endpoint_type=request.endpoint_type,
            metadata=request.metadata,
            post_inference_hooks=request.post_inference_hooks,
            child_fn_info=None,
            cpus=request.cpus,
            gpus=request.gpus,
            memory=request.memory,
            gpu_type=request.gpu_type,
            storage=request.storage,
            optimize_costs=bool(request.optimize_costs),
            min_workers=request.min_workers,
            max_workers=request.max_workers,
            per_worker=request.per_worker,
            labels=request.labels,
            aws_role=aws_role,
            results_s3_bucket=results_s3_bucket,
            prewarm=prewarm,
            high_priority=high_priority,
            owner=user.team_id,
            default_callback_url=request.default_callback_url,
            default_callback_auth=request.default_callback_auth,
            public_inference=request.public_inference,
        )
        _handle_post_inference_hooks(
            created_by=user.user_id,
            name=request.name,
            post_inference_hooks=request.post_inference_hooks,
        )

        return CreateModelEndpointV1Response(
            endpoint_creation_task_id=model_endpoint_record.creation_task_id  # type: ignore
        )


class UpdateModelEndpointByIdV1UseCase:
    def __init__(
        self,
        model_bundle_repository: ModelBundleRepository,
        model_endpoint_service: ModelEndpointService,
    ):
        self.model_bundle_repository = model_bundle_repository
        self.model_endpoint_service = model_endpoint_service
        self.authz_module = ScaleAuthorizationModule()

    async def execute(
        self, user: User, model_endpoint_id: str, request: UpdateModelEndpointV1Request
    ) -> UpdateModelEndpointV1Response:

        validate_post_inference_hooks(user, request.post_inference_hooks)

        if request.model_bundle_id is not None:
            bundle = await self.model_bundle_repository.get_model_bundle(
                model_bundle_id=request.model_bundle_id
            )
            if bundle is None:
                raise ObjectNotFoundException
            if not self.authz_module.check_access_read_owned_entity(user, bundle):
                raise ObjectNotAuthorizedException

        endpoint = await self.model_endpoint_service.get_model_endpoint(
            model_endpoint_id=model_endpoint_id
        )

        if endpoint is None:
            logger.error(f"Endpoint not found for {model_endpoint_id=}")
            raise ObjectNotFoundException

        endpoint_record = endpoint.record
        if endpoint_record is None:
            logger.error(f"Endpoint record not found for {model_endpoint_id=}")
            raise ObjectNotFoundException
        if not self.authz_module.check_access_write_owned_entity(user, endpoint_record):
            logger.error(
                f"{user=} is not authorized to write to endpoint record with {model_endpoint_id=}"
            )
            raise ObjectNotAuthorizedException

        # TODO: We may want to consider what happens if an endpoint gets stuck in UPDATE_PENDING
        #  on first creating it, and we need to find a way to get it unstuck. This would end up
        # causing endpoint.infra_state to be None.
        if endpoint.infra_state is None:
            error_msg = f"Endpoint infra state not found for {model_endpoint_id=}"
            logger.error(error_msg)
            raise EndpointInfraStateNotFound(error_msg)

        infra_state = endpoint.infra_state

        # For resources that are not specified in the update endpoint request, pass in resource from
        # infra_state to make sure that after the update, all resources are valid and in sync.
        # E.g. If user only want to update gpus and leave gpu_type as None, we use the existing gpu_type
        # from infra_state to avoid passing in None to validate_resource_requests.
        validate_resource_requests(
            cpus=request.cpus or infra_state.resource_state.cpus,
            memory=request.memory or infra_state.resource_state.memory,
            storage=request.storage or infra_state.resource_state.storage,
            gpus=request.gpus or infra_state.resource_state.gpus,
            gpu_type=request.gpu_type or infra_state.resource_state.gpu_type,
        )

        validate_deployment_resources(
            min_workers=request.min_workers,
            max_workers=request.max_workers,
            endpoint_type=endpoint_record.endpoint_type,
        )

        if request.metadata is not None and CONVERTED_FROM_ARTIFACT_LIKE_KEY in request.metadata:
            raise ObjectHasInvalidValueException(
                f"{CONVERTED_FROM_ARTIFACT_LIKE_KEY} is a reserved metadata key and cannot be used by user."
            )

        updated_endpoint_record = await self.model_endpoint_service.update_model_endpoint(
            model_endpoint_id=model_endpoint_id,
            model_bundle_id=request.model_bundle_id,
            metadata=request.metadata,
            post_inference_hooks=request.post_inference_hooks,
            cpus=request.cpus,
            gpus=request.gpus,
            memory=request.memory,
            gpu_type=request.gpu_type,
            storage=request.storage,
            optimize_costs=request.optimize_costs,
            min_workers=request.min_workers,
            max_workers=request.max_workers,
            per_worker=request.per_worker,
            labels=request.labels,
            prewarm=request.prewarm,
            high_priority=request.high_priority,
            default_callback_url=request.default_callback_url,
            default_callback_auth=request.default_callback_auth,
            public_inference=request.public_inference,
        )
        _handle_post_inference_hooks(
            created_by=endpoint_record.created_by,
            name=updated_endpoint_record.name,
            post_inference_hooks=request.post_inference_hooks,
        )

        return UpdateModelEndpointV1Response(
            endpoint_creation_task_id=updated_endpoint_record.creation_task_id  # type: ignore
        )


class ListModelEndpointsV1UseCase:
    """
    Use case for listing all versions of a Model Endpoint of a given user and model endpoint name.
    """

    def __init__(self, model_endpoint_service: ModelEndpointService):
        self.model_endpoint_service = model_endpoint_service

    async def execute(
        self, user: User, name: Optional[str], order_by: Optional[ModelEndpointOrderBy]
    ) -> ListModelEndpointsV1Response:
        """
        Runs the use case to list all Model Endpoints owned by the user with the given name.

        Args:
            user: The owner of the model endpoint(s).
            name: The name of the Model Endpoint(s).
            order_by: An optional argument to specify the output ordering of the model endpoints.

        Returns:
            A response object that contains the model endpoints.
        """
        model_endpoints = await self.model_endpoint_service.list_model_endpoints(
            owner=user.team_id, name=name, order_by=order_by
        )
        return ListModelEndpointsV1Response(
            model_endpoints=[
                model_endpoint_entity_to_get_model_endpoint_response(m) for m in model_endpoints
            ]
        )


class GetModelEndpointByIdV1UseCase:
    """
    Use case for getting a Model Endpoint of a given user by ID.
    """

    def __init__(self, model_endpoint_service: ModelEndpointService):
        self.model_endpoint_service = model_endpoint_service
        self.authz_module = ScaleAuthorizationModule()

    async def execute(self, user: User, model_endpoint_id: str) -> GetModelEndpointV1Response:
        """
        Runs the use case to list all Model Endpoints owned by the user with the given name.

        Args:
            user: The owner of the model endpoint.
            model_endpoint_id: The ID of the model endpoint.

        Returns:
            A response object that contains the model endpoint.

        Raises:
            ObjectNotFoundException: If a model endpoint with the given ID could not be found.
            ObjectNotAuthorizedException: If the owner does not own the model endpoint.
        """
        model_endpoint = await self.model_endpoint_service.get_model_endpoint(model_endpoint_id)
        if not model_endpoint:
            raise ObjectNotFoundException
        if not self.authz_module.check_access_read_owned_entity(user, model_endpoint.record):
            raise ObjectNotAuthorizedException
        return model_endpoint_entity_to_get_model_endpoint_response(model_endpoint)


class DeleteModelEndpointByIdV1UseCase:
    def __init__(self, model_endpoint_service: ModelEndpointService):
        self.model_endpoint_service = model_endpoint_service
        self.authz_module = ScaleAuthorizationModule()

    async def execute(self, user: User, model_endpoint_id: str) -> DeleteModelEndpointV1Response:
        model_endpoint = await self.model_endpoint_service.get_model_endpoint_record(
            model_endpoint_id=model_endpoint_id
        )
        if model_endpoint is None:
            raise ObjectNotFoundException
        if not self.authz_module.check_access_write_owned_entity(user, model_endpoint):
            raise ObjectNotAuthorizedException
        await self.model_endpoint_service.delete_model_endpoint(model_endpoint_id)
        return DeleteModelEndpointV1Response(deleted=True)


class ListModelEndpointHistoryByIdV1UseCase:
    # Not implemented as of now
    pass


class ReadModelEndpointCreationLogsByIdV1UseCase:
    # Not really core CRUD ops on an endpoint
    pass
