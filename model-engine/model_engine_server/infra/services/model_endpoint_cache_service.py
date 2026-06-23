from typing import Dict, Tuple

from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.entities import ModelEndpointInfraState
from model_engine_server.domain.gateways.monitoring_metrics_gateway import MonitoringMetricsGateway
from model_engine_server.infra.gateways.resources.endpoint_resource_gateway import (
    EndpointResourceGateway,
)
from model_engine_server.infra.repositories.model_endpoint_cache_repository import (
    ModelEndpointCacheRepository,
)
from model_engine_server.infra.services.image_cache_service import ImageCacheService

logger = make_logger(logger_name())


class ModelEndpointCacheWriteService:
    """
    Represents reading from k8s and writing data to the global cache
    """

    def __init__(
        self,
        model_endpoint_cache_repository: ModelEndpointCacheRepository,
        resource_gateway: EndpointResourceGateway,
        image_cache_service: ImageCacheService,
        monitoring_metrics_gateway: MonitoringMetricsGateway,
    ):
        self.model_endpoint_cache_repository = model_endpoint_cache_repository
        self.resource_gateway = resource_gateway
        self.image_cache_service = image_cache_service
        self.monitoring_metrics_gateway = monitoring_metrics_gateway

    async def execute(self, ttl_seconds: float):
        endpoint_infra_states: Dict[str, Tuple[bool, ModelEndpointInfraState]] = (
            await self.resource_gateway.get_all_resources()
        )

        try:
            for key, (is_key_an_endpoint_id, state) in endpoint_infra_states.items():
                if is_key_an_endpoint_id:
                    await self.model_endpoint_cache_repository.write_endpoint_info(
                        endpoint_id=key, endpoint_info=state, ttl_seconds=ttl_seconds
                    )
                else:
                    # TODO: Once we've backfilled all k8s resources to have an endpoint_id label,
                    # then we can get rid of this branch (also in the write_endpoint_info method, as
                    # well as simplifying the return type of get_all_resources() to not require the
                    # bool).
                    await self.model_endpoint_cache_repository.write_endpoint_info(
                        endpoint_id="", endpoint_info=state, ttl_seconds=ttl_seconds
                    )
        except Exception:
            # A silent Redis-write failure here lets cache entries expire, after which the Gateway
            # reports endpoint status as `unknown`. Surface the cause (log + metric) before
            # re-raising so the failure is observable rather than deceptive.
            logger.exception("Failed to write endpoint info to Redis cache")
            self.monitoring_metrics_gateway.emit_cache_write_failure_metric()
            raise

        await self.image_cache_service.execute(endpoint_infra_states=endpoint_infra_states)
