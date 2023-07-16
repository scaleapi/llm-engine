from pydantic import BaseModel

from llm_engine_server.common.dtos.endpoint_builder import BuildEndpointRequest


class CreateOrUpdateResourcesRequest(BaseModel):
    build_endpoint_request: BuildEndpointRequest
    image: str
