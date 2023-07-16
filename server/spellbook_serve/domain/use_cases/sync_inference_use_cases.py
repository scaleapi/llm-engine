from spellbook_serve.common.dtos.tasks import (
    EndpointPredictV1Request,
    SyncEndpointPredictV1Response,
)
from spellbook_serve.core.auth.authentication_repository import User
from spellbook_serve.core.domain_exceptions import (
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
)
from spellbook_serve.domain.authorization.scale_authorization_module import ScaleAuthorizationModule
from spellbook_serve.domain.entities import ModelEndpointType
from spellbook_serve.domain.exceptions import EndpointUnsupportedInferenceTypeException
from spellbook_serve.domain.services.model_endpoint_service import ModelEndpointService


class CreateSyncInferenceTaskV1UseCase:
    """
    Use case for creating a sync inference for an endpoint.
    """

    def __init__(self, model_endpoint_service: ModelEndpointService):
        self.model_endpoint_service = model_endpoint_service
        self.authz_module = ScaleAuthorizationModule()

    async def execute(
        self, user: User, model_endpoint_id: str, request: EndpointPredictV1Request
    ) -> SyncEndpointPredictV1Response:
        """
        Runs the use case to create a sync inference task.

        Args:
            user: The user who is creating the sync inference task.
            model_endpoint_id: The ID of the model endpoint for the task.
            request: The body of the request to forward to the endpoint.

        Returns:
            A response object that contains the status and result of the task.

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

        if (
            model_endpoint.record.endpoint_type is not ModelEndpointType.SYNC
            and model_endpoint.record.endpoint_type is not ModelEndpointType.STREAMING
        ):
            raise EndpointUnsupportedInferenceTypeException(
                f"Endpoint {model_endpoint_id} does not serve sync tasks."
            )

        inference_gateway = self.model_endpoint_service.get_sync_model_endpoint_inference_gateway()
        return await inference_gateway.predict(
            topic=model_endpoint.record.destination, predict_request=request
        )
