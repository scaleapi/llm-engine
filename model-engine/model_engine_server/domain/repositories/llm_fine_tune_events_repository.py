from abc import ABC, abstractmethod
from typing import List

from model_engine_server.domain.entities.llm_fine_tune_entity import LLMFineTuneEvent


class LLMFineTuneEventsRepository(ABC):
    @abstractmethod
    async def get_fine_tune_events(
        self, user_id: str, model_endpoint_name: str
    ) -> List[LLMFineTuneEvent]:
        pass

    @abstractmethod
    async def initialize_events(self, user_id: str, model_endpoint_name: str) -> None:
        pass
