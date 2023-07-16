from llm_engine_server.common.dtos.model_endpoints import GetModelEndpointsSchemaV1Response
from llm_engine_server.core.auth.authentication_repository import User
from llm_engine_server.domain.authorization.scale_authorization_module import (
    ScaleAuthorizationModule,
)
from llm_engine_server.domain.services import ModelEndpointService


class GetModelEndpointsSchemaV1UseCase:
    """
    Use case for getting the OpenAPI schema for the model endpoints.
    """

    def __init__(self, model_endpoint_service: ModelEndpointService):
        self.model_endpoint_service = model_endpoint_service
        self.authz_module = ScaleAuthorizationModule()

    async def execute(self, user: User) -> GetModelEndpointsSchemaV1Response:
        """Execute the use case.

        Args:
            user: The user who is requesting the schema.

        Returns:
            A response object that contains the OpenAPI schema for the model endpoints.
        """
        schema = await self.model_endpoint_service.get_model_endpoints_schema(user.team_id)
        return GetModelEndpointsSchemaV1Response(model_endpoints_schema=schema)
