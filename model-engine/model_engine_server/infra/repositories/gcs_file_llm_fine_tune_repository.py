import json
from typing import IO, Dict, Optional

import smart_open
from model_engine_server.domain.entities.llm_fine_tune_entity import LLMFineTuneTemplate
from model_engine_server.infra.gateways.gcs_storage_client import get_gcs_sync_client
from model_engine_server.infra.repositories.llm_fine_tune_repository import LLMFineTuneRepository


class GCSFileLLMFineTuneRepository(LLMFineTuneRepository):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def _open(self, uri: str, mode: str = "rt", **kwargs) -> IO:
        client = get_gcs_sync_client()
        transport_params = {"client": client}
        return smart_open.open(uri, mode, transport_params=transport_params)

    @staticmethod
    def _get_key(model_name, fine_tuning_method):
        return f"{model_name}-{fine_tuning_method}"

    async def get_job_template_for_model(
        self, model_name: str, fine_tuning_method: str
    ) -> Optional[LLMFineTuneTemplate]:
        with self._open(self.file_path, "r") as f:
            data = json.load(f)
            key = self._get_key(model_name, fine_tuning_method)
            job_template_dict = data.get(key, None)
            if job_template_dict is None:
                return None
            return LLMFineTuneTemplate.parse_obj(job_template_dict)

    async def write_job_template_for_model(
        self, model_name: str, fine_tuning_method: str, job_template: LLMFineTuneTemplate
    ):
        with self._open(self.file_path, "r") as f:
            data: Dict = json.load(f)
        key = self._get_key(model_name, fine_tuning_method)
        data[key] = dict(job_template)
        with self._open(self.file_path, "w") as f:
            json.dump(data, f)

    async def initialize_data(self):
        with self._open(self.file_path, "w") as f:
            json.dump({}, f)
