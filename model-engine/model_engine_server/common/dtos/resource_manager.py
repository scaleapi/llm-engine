from model_engine_server.common.dtos.endpoint_builder import BuildEndpointRequest
from model_engine_server.common.pydantic_types import BaseModel


class CreateOrUpdateResourcesRequest(BaseModel):
    build_endpoint_request: BuildEndpointRequest
    image: str
