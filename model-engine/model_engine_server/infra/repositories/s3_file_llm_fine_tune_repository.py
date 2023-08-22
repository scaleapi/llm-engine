import json
import os
from typing import IO, Dict, Optional

import boto3
import smart_open
from model_engine_server.domain.entities.llm_fine_tune_entity import LLMFineTuneTemplate
from model_engine_server.infra.repositories.llm_fine_tune_repository import LLMFineTuneRepository


class S3FileLLMFineTuneRepository(LLMFineTuneRepository):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def _get_s3_client(self, kwargs):
        profile_name = kwargs.get("aws_profile", os.getenv("AWS_PROFILE"))
        session = boto3.Session(profile_name=profile_name)
        client = session.client("s3")
        return client

    def _open(self, uri: str, mode: str = "rt", **kwargs) -> IO:
        # This follows the 5.1.0 smart_open API
        client = self._get_s3_client(kwargs)
        transport_params = {"client": client}
        return smart_open.open(uri, mode, transport_params=transport_params)

    @staticmethod
    def _get_key(model_name, fine_tuning_method):
        return f"{model_name}-{fine_tuning_method}"  # possible for collisions but we control these names

    async def get_job_template_for_model(
        self, model_name: str, fine_tuning_method: str
    ) -> Optional[LLMFineTuneTemplate]:
        # can hot reload the file lol
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
        # Use locally in script
        with self._open(self.file_path, "r") as f:
            data: Dict = json.load(f)
        key = self._get_key(model_name, fine_tuning_method)
        data[key] = dict(job_template)
        with self._open(self.file_path, "w") as f:
            json.dump(data, f)

    async def initialize_data(self):
        # Use locally in script
        with self._open(self.file_path, "w") as f:
            json.dump({}, f)
