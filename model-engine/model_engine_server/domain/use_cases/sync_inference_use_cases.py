from model_engine_server.common.config import hmi_config
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


class CreateSyncInferenceTaskV1UseCase:
    """
    Use case for creating a sync inference for an endpoint.
    """

    def __init__(self, model_endpoint_service: ModelEndpointService):
        self.model_endpoint_service = model_endpoint_service
        self.authz_module = LiveAuthorizationModule()

    async def execute(
        self, user: User, model_endpoint_id: str, request: SyncEndpointPredictV1Request
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
            asyncio.exceptions.TimeoutError: If the task times out.
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
        autoscaling_metrics_gateway = (
            self.model_endpoint_service.get_inference_autoscaling_metrics_gateway()
        )
        await autoscaling_metrics_gateway.emit_inference_autoscaling_metric(
            endpoint_id=model_endpoint_id
        )
        # Hack: manually resolve dns if istio is present. Since we do not inject istio for multinode,
        # empirically we find that without manual dns resolution, requests to the k8s service DNS name fail,
        # likely because the requests are getting changed by Istio. A fix is to resolve the service DNS name
        # (e.g. model-endpoint-foo.namespace.svc.cluster.local) to the actual IP address of the service
        manually_resolve_dns = (
            model_endpoint.infra_state is not None
            and model_endpoint.infra_state.resource_state.nodes_per_worker > 1
            and hmi_config.istio_enabled
        )
        return await inference_gateway.predict(
            topic=model_endpoint.record.destination,
            predict_request=request,
            manually_resolve_dns=manually_resolve_dns,
            endpoint_name=model_endpoint.record.name,
        )
