from abc import ABC, abstractmethod
from typing import Optional

from llm_engine_server.domain.entities.llm_fine_tune_job_entity import LLMFineTuneJobTemplate


class LLMFineTuningJobRepository(ABC):
    """
    Basically a store of model name + fine tuning method -> docker image batch job bundle ids

    """

    @abstractmethod
    async def get_job_template_for_model(
        self, model_name: str, fine_tuning_method: str
    ) -> Optional[LLMFineTuneJobTemplate]:
        pass

    @abstractmethod
    async def write_job_template_for_model(
        self, model_name: str, fine_tuning_method: str, job_template: LLMFineTuneJobTemplate
    ):
        pass
