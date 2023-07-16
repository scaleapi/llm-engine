from typing import Dict, Optional

from llmengine.api_engine import DEFAULT_TIMEOUT, APIEngine
from llmengine.data_types import (
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
        model: str,
        training_file: str,
        validation_file: Optional[str] = None,
        hyperparameters: Optional[Dict[str, str]] = None,
        suffix: Optional[str] = None,
    ) -> CreateFineTuneJobResponse:
        """
        Create a fine-tuning job.

        Example:
            ```python
            from llmengine import FineTune

            response = FineTune.create(
                training_file="s3://my-bucket/path/to/training-file.csv",
                validation_file="s3://my-bucket/path/to/validation-file.csv",
                model_name="llama-7b-ft-2023-07-18",
                base_model="llama-7b",
                fine_tuning_method="ia3",
                hyperparameters={},
            )

            print(response)
            ```

        JSON Response:
            ```json
            ```

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
            model=model,
            training_file=training_file,
            validation_file=validation_file,
            hyperparameters=hyperparameters,
            suffix=suffix,
        )
        response = cls.post_sync(
            resource_name="v1/llm/fine-tunes",
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

        Example:
            ```python
            from llmengine import FineTune

            response = FineTune.retrieve(
                fine_tune_id="ft_abc123...",
            )

            print(response)
            ```

        JSON Response:
            ```json
            ```


        Args:
            fine_tune_id (`str`):
                ID of the fine-tuning job

        Returns:
            GetFineTuneJobResponse: ID and status of the requested job
        """
        response = cls.get(f"v1/llm/fine-tunes/{fine_tune_id}", timeout=DEFAULT_TIMEOUT)
        return GetFineTuneJobResponse.parse_obj(response)

    @classmethod
    def list(cls) -> ListFineTuneJobResponse:
        """
        List fine-tuning jobs

        Example:
            ```python
            from llmengine import FineTune

            response = FineTune.list()
            print(response)
            ```

        JSON Response:
            ```json
            ```

        Returns:
            ListFineTuneJobResponse: list of all fine-tuning jobs and their statuses
        """
        response = cls.get("v1/llm/fine-tunes", timeout=DEFAULT_TIMEOUT)
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
            f"v1/llm/fine-tunes/{fine_tune_id}/cancel",
            data=None,
            timeout=DEFAULT_TIMEOUT,
        )
        return CancelFineTuneJobResponse.parse_obj(response)
