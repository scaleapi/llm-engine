from abc import ABC, abstractmethod
from typing import Dict


class LLMFineTuningService(ABC):
    @abstractmethod
    async def create_fine_tune_job(
        self,
        created_by: str,
        owner: str,
        training_file: str,
        validation_file: str,
        model_name: str,
        base_model: str,
        fine_tuning_method: str,
        hyperparameters: Dict[str, str],
    ):
        pass

    @abstractmethod
    async def get_fine_tune_job(self, owner: str, fine_tune_id: str):
        pass

    @abstractmethod
    async def list_fine_tune_jobs(self, owner: str):
        pass

    @abstractmethod
    async def cancel_fine_tune_job(self, owner: str, fine_tune_id: str):
        pass
