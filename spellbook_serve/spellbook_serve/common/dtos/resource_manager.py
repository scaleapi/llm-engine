from pydantic import BaseModel

from spellbook_serve.common.dtos.endpoint_builder import BuildEndpointRequest


class CreateOrUpdateResourcesRequest(BaseModel):
    build_endpoint_request: BuildEndpointRequest
    image: str
