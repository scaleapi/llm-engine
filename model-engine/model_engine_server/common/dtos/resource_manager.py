from model_engine_server.common.dtos.endpoint_builder import BuildEndpointRequest
from pydantic.v1 import BaseModel


class CreateOrUpdateResourcesRequest(BaseModel):
    build_endpoint_request: BuildEndpointRequest
    image: str
