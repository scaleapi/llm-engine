from model_engine_server.common.dtos.tasks import (
    CreateAsyncTaskV1Response,
    EndpointPredictV1Request,
    GetAsyncTaskV1Response,
)
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.core.domain_exceptions import (
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
)
from model_engine_server.domain.authorization.live_authorization_module import (
    LiveAuthorizationModule,
)
from model_engine_server.domain.entities import ModelEndpointType
from model_engine_server.domain.exceptions import EndpointUnsupportedInferenceTypeException
from model_engine_server.domain.services.model_endpoint_service import ModelEndpointService

DEFAULT_TASK_TIMEOUT_SECONDS = 86400


class CreateAsyncInferenceTaskV1UseCase:
    """
    Use case for creating an async inference for an endpoint.
    """

    def __init__(self, model_endpoint_service: ModelEndpointService):
        self.model_endpoint_service = model_endpoint_service
        self.authz_module = LiveAuthorizationModule()

    async def execute(
        self, user: User, model_endpoint_id: str, request: EndpointPredictV1Request
    ) -> CreateAsyncTaskV1Response:
        """
        Runs the use case to create an async inference task.

        Args:
            user: The user who is creating the async inference task.
            model_endpoint_id: The ID of the model endpoint for the task.
            request: The body of the request to forward to the endpoint.

        Returns:
            A response object that contains the ID of the created task.

        Raises:
            ObjectNotFoundException: If a model endpoint with the given ID could not be found.
            ObjectNotAuthorizedException: If the owner does not own the model endpoint.
        """
        model_endpoint = await self.model_endpoint_service.get_model_endpoint(
            model_endpoint_id=model_endpoint_id
        )
        if model_endpoint is None:
            raise ObjectNotFoundException

        if not self.authz_module.check_access_read_owned_entity(
            user, model_endpoint.record
        ) and not self.authz_module.check_endpoint_public_inference_for_user(
            user, model_endpoint.record
        ):
            raise ObjectNotAuthorizedException

        if model_endpoint.record.endpoint_type != ModelEndpointType.ASYNC:
            raise EndpointUnsupportedInferenceTypeException(
                f"Endpoint {model_endpoint_id} is not an async endpoint."
            )

        task_name = model_endpoint.record.current_model_bundle.celery_task_name()

        inference_gateway = self.model_endpoint_service.get_async_model_endpoint_inference_gateway()
        return inference_gateway.create_task(
            topic=model_endpoint.record.destination,
            predict_request=request,
            task_timeout_seconds=DEFAULT_TASK_TIMEOUT_SECONDS,
            task_name=task_name,
        )


class GetAsyncInferenceTaskV1UseCase:
    def __init__(self, model_endpoint_service: ModelEndpointService):
        self.model_endpoint_service = model_endpoint_service

    def execute(self, user: User, task_id: str) -> GetAsyncTaskV1Response:
        """
        Runs the use case to create an async inference task.

        Args:
            user: The user who is creating the async inference task.
            task_id: The ID of the task.

        Returns:
            A response object that contains the status and result of the task.

        Raises:
            ObjectNotFoundException: If a task with the given ID could not be found.
            ObjectNotAuthorizedException: If the owner does not own the task.
        """
        # TODO: check that user is authorized to access this task.
        inference_gateway = self.model_endpoint_service.get_async_model_endpoint_inference_gateway()
        return inference_gateway.get_task(task_id=task_id)
