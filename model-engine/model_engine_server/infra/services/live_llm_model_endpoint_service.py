from typing import List, Optional

from model_engine_server.common.dtos.model_endpoints import ModelEndpointOrderBy
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.entities import ModelEndpoint
from model_engine_server.domain.services import LLMModelEndpointService
from model_engine_server.infra.repositories.model_endpoint_record_repository import (
    ModelEndpointRecordRepository,
)
from model_engine_server.infra.services import LiveModelEndpointService

logger = make_logger(logger_name())


class LiveLLMModelEndpointService(LLMModelEndpointService):
    def __init__(
        self,
        model_endpoint_record_repository: ModelEndpointRecordRepository,
        model_endpoint_service: LiveModelEndpointService,
    ):
        self.model_endpoint_record_repository = model_endpoint_record_repository
        self.model_endpoint_service = model_endpoint_service

    async def list_llm_model_endpoints(
        self,
        owner: Optional[str],
        name: Optional[str],
        order_by: Optional[ModelEndpointOrderBy],
    ) -> List[ModelEndpoint]:
        # Will read from cache at first
        records = await self.model_endpoint_record_repository.list_llm_model_endpoint_records(
            owner=owner,
            name=name,
            order_by=order_by,
        )
        endpoints: List[ModelEndpoint] = []
        for record in records:
            infra_state = await self.model_endpoint_service._get_model_endpoint_infra_state(
                record=record, use_cache=True
            )
            endpoints.append(ModelEndpoint(record=record, infra_state=infra_state))
        return endpoints

    async def get_llm_model_endpoint(self, model_endpoint_name: str) -> Optional[ModelEndpoint]:
        model_endpoint_record = (
            await self.model_endpoint_record_repository.get_llm_model_endpoint_record(
                model_endpoint_name=model_endpoint_name
            )
        )
        if model_endpoint_record is None:
            return None

        model_endpoint_infra_state = (
            await self.model_endpoint_service._get_model_endpoint_infra_state(
                record=model_endpoint_record, use_cache=True
            )
        )
        return ModelEndpoint(record=model_endpoint_record, infra_state=model_endpoint_infra_state)
