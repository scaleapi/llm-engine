from typing import Optional

from model_engine_server.domain.entities.llm_fine_tune_entity import LLMFineTuneTemplate
from model_engine_server.infra.repositories.llm_fine_tune_repository import LLMFineTuneRepository


class ABSFileLLMFineTuneRepository(LLMFineTuneRepository):
    def __init__(self, file_path: str):
        self.file_path = file_path

    async def get_job_template_for_model(
        self, model_name: str, fine_tuning_method: str
    ) -> Optional[LLMFineTuneTemplate]:
        raise NotImplementedError("ABS not supported yet")

    async def write_job_template_for_model(
        self, model_name: str, fine_tuning_method: str, job_template: LLMFineTuneTemplate
    ):
        raise NotImplementedError("ABS not supported yet")
