from abc import ABC, abstractmethod
from typing import Optional

from model_engine_server.domain.entities.llm_fine_tune_entity import LLMFineTuneTemplate


class LLMFineTuneRepository(ABC):
    """
    Basically a store of model name + fine tuning method -> docker image batch job bundle ids

    """

    @abstractmethod
    async def get_job_template_for_model(
        self, model_name: str, fine_tuning_method: str
    ) -> Optional[LLMFineTuneTemplate]:
        pass

    @abstractmethod
    async def write_job_template_for_model(
        self, model_name: str, fine_tuning_method: str, job_template: LLMFineTuneTemplate
    ):
        pass
