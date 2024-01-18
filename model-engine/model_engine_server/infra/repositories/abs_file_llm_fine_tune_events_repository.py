from typing import List

from model_engine_server.domain.entities.llm_fine_tune_entity import LLMFineTuneEvent
from model_engine_server.domain.repositories.llm_fine_tune_events_repository import (
    LLMFineTuneEventsRepository,
)


class ABSFileLLMFineTuneEventsRepository(LLMFineTuneEventsRepository):
    def __init__(self):
        pass

    async def get_fine_tune_events(
        self, user_id: str, model_endpoint_name: str
    ) -> List[LLMFineTuneEvent]:
        raise NotImplementedError("ABS not supported yet")

    async def initialize_events(self, user_id: str, model_endpoint_name: str) -> None:
        raise NotImplementedError("ABS not supported yet")
