from typing import Dict, Tuple

from model_engine_server.domain.entities import ModelEndpointInfraState
from model_engine_server.infra.gateways.resources.endpoint_resource_gateway import (
    EndpointResourceGateway,
)
from model_engine_server.infra.repositories.model_endpoint_cache_repository import (
    ModelEndpointCacheRepository,
)
from model_engine_server.infra.services.image_cache_service import ImageCacheService


class ModelEndpointCacheWriteService:
    """
    Represents reading from k8s and writing data to the global cache
    """

    def __init__(
        self,
        model_endpoint_cache_repository: ModelEndpointCacheRepository,
        resource_gateway: EndpointResourceGateway,
        image_cache_service: ImageCacheService,
    ):
        self.model_endpoint_cache_repository = model_endpoint_cache_repository
        self.resource_gateway = resource_gateway
        self.image_cache_service = image_cache_service

    async def execute(self, ttl_seconds: float):
        endpoint_infra_states: Dict[str, Tuple[bool, ModelEndpointInfraState]] = (
            await self.resource_gateway.get_all_resources()
        )

        for key, (is_key_an_endpoint_id, state) in endpoint_infra_states.items():
            if is_key_an_endpoint_id:
                await self.model_endpoint_cache_repository.write_endpoint_info(
                    endpoint_id=key, endpoint_info=state, ttl_seconds=ttl_seconds
                )
            else:
                # TODO: Once we've backfilled all k8s resources to have an endpoint_id label, then
                # we can get rid of this branch (also in the write_endpoint_info method, as well as
                # simplifying the return type of get_all_resources() to not require the bool).
                await self.model_endpoint_cache_repository.write_endpoint_info(
                    endpoint_id="", endpoint_info=state, ttl_seconds=ttl_seconds
                )

        await self.image_cache_service.execute(endpoint_infra_states=endpoint_infra_states)
