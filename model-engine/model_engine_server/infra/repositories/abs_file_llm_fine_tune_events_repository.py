import json
import os
from json.decoder import JSONDecodeError
from typing import IO, List

import smart_open
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from model_engine_server.domain.entities.llm_fine_tune_entity import LLMFineTuneEvent
from model_engine_server.domain.exceptions import ObjectNotFoundException
from model_engine_server.domain.repositories.llm_fine_tune_events_repository import (
    LLMFineTuneEventsRepository,
)

# Echoes llm/ia3_finetune/docker_image_fine_tuning_entrypoint.py
ABS_HF_USER_FINE_TUNED_WEIGHTS_PREFIX = f"https://{os.getenv('ABS_ACCOUNT_NAME')}.blob.core.windows.net/hosted-model-inference/fine_tuned_weights"


class ABSFileLLMFineTuneEventsRepository(LLMFineTuneEventsRepository):
    def __init__(self):
        pass

    def _open(self, uri: str, mode: str = "rt", **kwargs) -> IO:
        client = BlobServiceClient(
            f"https://{os.getenv('ABS_ACCOUNT_NAME')}.blob.core.windows.net",
            DefaultAzureCredential(),
        )
        transport_params = {"client": client}
        return smart_open.open(uri, mode, transport_params=transport_params)

    # echoes llm/ia3_finetune/docker_image_fine_tuning_entrypoint.py
    def _get_model_cache_directory_name(self, model_name: str):
        """How huggingface maps model names to directory names in their cache for model files.
        We adopt this when storing model cache files in s3.

        Args:
            model_name (str): Name of the huggingface model
        """
        name = "models--" + model_name.replace("/", "--")
        return name

    def _get_file_location(self, user_id: str, model_endpoint_name: str):
        model_cache_name = self._get_model_cache_directory_name(model_endpoint_name)
        abs_file_location = (
            f"{ABS_HF_USER_FINE_TUNED_WEIGHTS_PREFIX}/{user_id}/{model_cache_name}.jsonl"
        )
        return abs_file_location

    async def get_fine_tune_events(
        self, user_id: str, model_endpoint_name: str
    ) -> List[LLMFineTuneEvent]:
        abs_file_location = self._get_file_location(
            user_id=user_id, model_endpoint_name=model_endpoint_name
        )
        try:
            with self._open(abs_file_location, "r") as f:
                lines = f.readlines()
                final_events = []
                for line in lines:
                    try:
                        event_dict = json.loads(line)
                        event = LLMFineTuneEvent(
                            timestamp=event_dict["timestamp"],
                            message=str(event_dict["message"]),
                            level=event_dict.get("level", "info"),
                        )
                    except JSONDecodeError:
                        event = LLMFineTuneEvent(
                            message=line,
                            level="info",
                        )
                    final_events.append(event)
                return final_events
        except Exception as exc:  # TODO better exception
            raise ObjectNotFoundException from exc

    async def initialize_events(self, user_id: str, model_endpoint_name: str) -> None:
        abs_file_location = self._get_file_location(
            user_id=user_id, model_endpoint_name=model_endpoint_name
        )
        self._open(abs_file_location, "w")
