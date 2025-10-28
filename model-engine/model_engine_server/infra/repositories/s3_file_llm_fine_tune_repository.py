import json
from typing import IO, Dict, Optional

import smart_open
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.entities.llm_fine_tune_entity import LLMFineTuneTemplate
from model_engine_server.infra.gateways.s3_utils import get_s3_client
from model_engine_server.infra.repositories.llm_fine_tune_repository import LLMFineTuneRepository

logger = make_logger(logger_name())


class S3FileLLMFineTuneRepository(LLMFineTuneRepository):
    def __init__(self, file_path: str):
        self.file_path = file_path
        logger.debug(f"Initialized S3FileLLMFineTuneRepository with path: {file_path}")

    def _open(self, uri: str, mode: str = "rt", **kwargs) -> IO:
        client = get_s3_client(kwargs)
        transport_params = {"client": client}
        return smart_open.open(uri, mode, transport_params=transport_params)

    @staticmethod
    def _get_key(model_name: str, fine_tuning_method: str) -> str:
        return f"{model_name}-{fine_tuning_method}" # possible for collisions but we control these names

    async def get_job_template_for_model(
        self, model_name: str, fine_tuning_method: str
    ) -> Optional[LLMFineTuneTemplate]:
        try:
            with self._open(self.file_path, "r") as f:
                data = json.load(f)
                key = self._get_key(model_name, fine_tuning_method)
                job_template_dict = data.get(key, None)
                if job_template_dict is None:
                    logger.debug(f"No template found for {key}")
                    return None
                logger.debug(f"Retrieved template for {key}")
                return LLMFineTuneTemplate.parse_obj(job_template_dict)
        except Exception as e:
            logger.error(f"Failed to get job template for {model_name}/{fine_tuning_method}: {e}")
            return None

    async def write_job_template_for_model(
        self, model_name: str, fine_tuning_method: str, job_template: LLMFineTuneTemplate
    ):
        with self._open(self.file_path, "r") as f:
            data: Dict = json.load(f)

        key = self._get_key(model_name, fine_tuning_method)
        data[key] = dict(job_template)

        with self._open(self.file_path, "w") as f:
            json.dump(data, f)

        logger.info(f"Wrote job template for {key}")

    async def initialize_data(self):
        with self._open(self.file_path, "w") as f:
            json.dump({}, f)
        logger.info(f"Initialized fine-tune repository at {self.file_path}")
