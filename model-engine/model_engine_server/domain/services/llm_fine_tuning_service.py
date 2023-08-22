from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from model_engine_server.domain.entities import FineTuneHparamValueType
from model_engine_server.domain.entities.batch_job_entity import DockerImageBatchJob


class LLMFineTuningService(ABC):
    @abstractmethod
    async def create_fine_tune(
        self,
        created_by: str,
        owner: str,
        model: str,
        training_file: str,
        validation_file: Optional[str],
        fine_tuning_method: str,
        hyperparameters: Dict[str, FineTuneHparamValueType],
        fine_tuned_model: str,
        wandb_config: Optional[Dict[str, Any]],
    ) -> str:
        pass

    @abstractmethod
    async def get_fine_tune(self, owner: str, fine_tune_id: str) -> Optional[DockerImageBatchJob]:
        pass

    @abstractmethod
    async def list_fine_tunes(self, owner: str) -> List[DockerImageBatchJob]:
        pass

    @abstractmethod
    async def cancel_fine_tune(self, owner: str, fine_tune_id: str) -> bool:
        pass

    @abstractmethod
    async def get_fine_tune_model_name_from_id(
        self, owner: str, fine_tune_id: str
    ) -> Optional[str]:
        pass
