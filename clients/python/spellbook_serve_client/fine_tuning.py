from typing import Dict

from spellbook_serve_client.api_engine import APIEngine, DEFAULT_TIMEOUT
from spellbook_serve_client.data_types import (
    CancelFineTuneJobResponse,
    CreateFineTuneJobRequest,
    CreateFineTuneJobResponse,
    GetFineTuneJobResponse,
    ListFineTuneJobResponse,
)


class FineTune(APIEngine):
    """
    FineTune API. This API is used to fine-tune models.
    """

    @classmethod
    def create(
        cls,
        training_file: str,
        validation_file: str,
        model_name: str,
        base_model: str,
        fine_tuning_method: str,
        hyperparameters: Dict[str, str],
    ) -> CreateFineTuneJobResponse:
        """
        Create a fine-tuning job

        Args:
            training_file (`str`):
                Path to file of training dataset
            validation_file (`str`):
                Path to file of validation dataset
            model_name (`str`):
                Name of the fine-tuned model
            base_model (`str`):
                Base model to train from
            fine_tuning_method (`str`):
                Fine-tuning method
            hyperparameters (`str`):
                Hyperparameters

        Returns:
            CreateFineTuneJobResponse: ID of the created fine-tuning job
        """
        request = CreateFineTuneJobRequest(
            training_file=training_file,
            validation_file=validation_file,
            model_name=model_name,
            base_model=base_model,
            fine_tuning_method=fine_tuning_method,
            hyperparameters=hyperparameters,
        )
        response = cls.post_sync(
            resource_name="v1/fine-tunes",
            data=request.dict(),
            timeout=DEFAULT_TIMEOUT,
        )
        return CreateFineTuneJobResponse.parse_obj(response)

    @classmethod
    def retrieve(
        cls,
        fine_tune_id: str,
    ) -> GetFineTuneJobResponse:
        """
        Get status of a fine-tuning job

        Args:
            fine_tune_id (`str`):
                ID of the fine-tuning job

        Returns:
            GetFineTuneJobResponse: ID and status of the requested job
        """
        response = cls.get(f"v1/fine-tunes/{fine_tune_id}", timeout=DEFAULT_TIMEOUT)
        return GetFineTuneJobResponse.parse_obj(response)

    @classmethod
    def list(cls) -> ListFineTuneJobResponse:
        """
        List fine-tuning jobs

        Returns:
            ListFineTuneJobResponse: list of all fine-tuning jobs and their statuses
        """
        response = cls.get("v1/fine-tunes", timeout=DEFAULT_TIMEOUT)
        return ListFineTuneJobResponse.parse_obj(response)

    @classmethod
    def cancel(cls, fine_tune_id: str) -> CancelFineTuneJobResponse:
        """
        Cancel a fine-tuning job

        Args:
            fine_tune_id (`str`):
                ID of the fine-tuning job

        Returns:
            CancelFineTuneJobResponse: whether the cancellation was successful
        """
        response = cls.put(
            f"v1/fine-tunes/{fine_tune_id}/cancel", data=None, timeout=DEFAULT_TIMEOUT
        )
        return CancelFineTuneJobResponse.parse_obj(response)
