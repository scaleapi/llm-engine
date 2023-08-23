import json
import os
from json.decoder import JSONDecodeError
from typing import IO, List

import boto3
import smart_open
from model_engine_server.core.config import infra_config
from model_engine_server.core.domain_exceptions import ObjectNotFoundException
from model_engine_server.domain.entities.llm_fine_tune_entity import LLMFineTuneEvent
from model_engine_server.domain.repositories.llm_fine_tune_events_repository import (
    LLMFineTuneEventsRepository,
)

# Echoes llm/ia3_finetune/docker_image_fine_tuning_entrypoint.py
S3_HF_USER_FINE_TUNED_WEIGHTS_PREFIX = (
    f"s3://{infra_config().s3_bucket}/hosted-model-inference/fine_tuned_weights"
)


class S3FileLLMFineTuneEventsRepository(LLMFineTuneEventsRepository):
    def __init__(self):
        pass

    # _get_s3_client + _open copypasted from s3_file_llm_fine_tune_repo, in turn from s3_filesystem_gateway
    # sorry
    def _get_s3_client(self, kwargs):
        profile_name = kwargs.get("aws_profile", os.getenv("S3_WRITE_AWS_PROFILE"))
        session = boto3.Session(profile_name=profile_name)
        client = session.client("s3")
        return client

    def _open(self, uri: str, mode: str = "rt", **kwargs) -> IO:
        # This follows the 5.1.0 smart_open API
        client = self._get_s3_client(kwargs)
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
        s3_file_location = (
            f"{S3_HF_USER_FINE_TUNED_WEIGHTS_PREFIX}/{user_id}/{model_cache_name}.jsonl"
        )
        return s3_file_location

    async def get_fine_tune_events(
        self, user_id: str, model_endpoint_name: str
    ) -> List[LLMFineTuneEvent]:
        s3_file_location = self._get_file_location(
            user_id=user_id, model_endpoint_name=model_endpoint_name
        )
        try:
            with self._open(s3_file_location, "r") as f:
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
        s3_file_location = self._get_file_location(
            user_id=user_id, model_endpoint_name=model_endpoint_name
        )
        self._open(s3_file_location, "w")
