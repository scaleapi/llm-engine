from abc import ABC, abstractmethod
from typing import Optional

from llm_engine_server.domain.entities import ModelEndpointInfraState


class ModelEndpointCacheRepository(ABC):
    @abstractmethod
    async def write_endpoint_info(
        self,
        endpoint_id: str,
        endpoint_info: ModelEndpointInfraState,
        ttl_seconds: float,
    ):
        """
        Writes the endpoint info to a cache
        Args:
            endpoint_id: The ID of the endpoint.
            endpoint_info: a ModelEndpointInfraState that we want cached
            ttl_seconds: TTL on the cache entry
        Returns:
            None
        """
        pass

    @abstractmethod
    async def read_endpoint_info(
        self, endpoint_id: str, deployment_name: str
    ) -> Optional[ModelEndpointInfraState]:
        """
        Reads the endpoint info from the cache
        Args:
            endpoint_id: The ID of the endpoint.
            deployment_name: (Deprecated) K8s name of deployment that we want to retrieve cache value from

        Returns:
            ModelEndpointInfraState if it's available in the cache
        """
        pass
