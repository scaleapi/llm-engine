from typing import AsyncIterable

from model_engine_server.common.dtos.tasks import (
    SyncEndpointPredictV1Request,
    SyncEndpointPredictV1Response,
)
from model_engine_server.core.auth.authentication_repository import User
from model_engine_server.domain.authorization.live_authorization_module import (
    LiveAuthorizationModule,
)
from model_engine_server.domain.entities import ModelEndpointType
from model_engine_server.domain.exceptions import (
    EndpointUnsupportedInferenceTypeException,
    ObjectNotAuthorizedException,
    ObjectNotFoundException,
)
from model_engine_server.domain.services.model_endpoint_service import ModelEndpointService


class CreateStreamingInferenceTaskV1UseCase:
    """
    Use case for creating a sync inference for an endpoint.
    """

    def __init__(self, model_endpoint_service: ModelEndpointService):
        self.model_endpoint_service = model_endpoint_service
        self.authz_module = LiveAuthorizationModule()

    async def execute(
        self, user: User, model_endpoint_id: str, request: SyncEndpointPredictV1Request
    ) -> AsyncIterable[SyncEndpointPredictV1Response]:
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

        if not self.authz_module.check_access_read_owned_entity(user, model_endpoint.record):
            raise ObjectNotAuthorizedException

        if model_endpoint.record.endpoint_type != ModelEndpointType.STREAMING:
            raise EndpointUnsupportedInferenceTypeException(
                f"Endpoint {model_endpoint_id} is not a streaming endpoint."
            )

        inference_gateway = (
            self.model_endpoint_service.get_streaming_model_endpoint_inference_gateway()
        )
        autoscaling_metrics_gateway = (
            self.model_endpoint_service.get_inference_autoscaling_metrics_gateway()
        )
        await autoscaling_metrics_gateway.emit_inference_autoscaling_metric(
            endpoint_id=model_endpoint_id
        )
        # TODO handle lws
        return inference_gateway.streaming_predict(
            topic=model_endpoint.record.destination, predict_request=request
        )
