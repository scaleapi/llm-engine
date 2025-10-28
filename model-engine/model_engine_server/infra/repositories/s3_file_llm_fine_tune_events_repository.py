import json
from json.decoder import JSONDecodeError
from typing import IO, List

import smart_open
from model_engine_server.core.config import infra_config
from model_engine_server.core.loggers import logger_name, make_logger
from model_engine_server.domain.entities.llm_fine_tune_entity import LLMFineTuneEvent
from model_engine_server.domain.exceptions import ObjectNotFoundException
from model_engine_server.domain.repositories.llm_fine_tune_events_repository import (
    LLMFineTuneEventsRepository,
)
from model_engine_server.infra.gateways.s3_utils import get_s3_client

logger = make_logger(logger_name())

S3_HF_USER_FINE_TUNED_WEIGHTS_PREFIX = (
    f"s3://{infra_config().s3_bucket}/hosted-model-inference/fine_tuned_weights"
)


class S3FileLLMFineTuneEventsRepository(LLMFineTuneEventsRepository):
    def __init__(self):
        logger.debug("Initialized S3FileLLMFineTuneEventsRepository")

    def _open(self, uri: str, mode: str = "rt", **kwargs) -> IO:
        client = get_s3_client(kwargs)
        transport_params = {"client": client}
        return smart_open.open(uri, mode, transport_params=transport_params)

    def _get_model_cache_directory_name(self, model_name: str) -> str:
        name = "models--" + model_name.replace("/", "--")
        return name

    def _get_file_location(self, user_id: str, model_endpoint_name: str) -> str:
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
                logger.debug(
                    f"Retrieved {len(final_events)} events for {user_id}/{model_endpoint_name}"
                )
                return final_events
        except Exception as exc:
            logger.error(f"Failed to get fine-tune events from {s3_file_location}: {exc}")
            raise ObjectNotFoundException from exc

    async def initialize_events(self, user_id: str, model_endpoint_name: str) -> None:
        s3_file_location = self._get_file_location(
            user_id=user_id, model_endpoint_name=model_endpoint_name
        )
        with self._open(s3_file_location, "w"):
            pass
        logger.info(f"Initialized events file at {s3_file_location}")
