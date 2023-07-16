from typing import Dict, Optional

from llmengine.api_engine import DEFAULT_TIMEOUT, APIEngine
from llmengine.data_types import (
    CancelFineTuneResponse,
    CreateFineTuneRequest,
    CreateFineTuneResponse,
    GetFineTuneResponse,
    ListFineTunesResponse,
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
    ) -> CreateFineTuneResponse:
        """
        Creates a job that fine-tunes a specified model from a given dataset.

        Example:
            ```python
            from llmengine import FineTune

            response = FineTune.create(
                model="llama-7b",
                training_file="s3://my-bucket/path/to/training-file.csv",
            )

            print(response.json())
            ```

        JSON Response:
            ```json
            {
                "fine_tune_id": "ft_abc123"
            }
            ```

        Args:
            model (`str`):
                The name of the base model to fine-tune. See #model_zoo for the list of available models to fine-tune.

            training_file (`str`):
                Path to file of training dataset

            validation_file (`Optional[str]`):
                Path to file of validation dataset

            hyperparameters (`str`):
                Hyperparameters
                
            suffix (`Optional[str]`):
                A string that will be added to your fine-tuned model name.


        Returns:
            CreateFineTuneResponse: an object that contains the ID of the created fine-tuning job
        """
        request = CreateFineTuneRequest(
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
        return CreateFineTuneResponse.parse_obj(response)

    @classmethod
    def retrieve(
        cls,
        fine_tune_id: str,
    ) -> GetFineTuneResponse:
        """
        Get status of a fine-tuning job

        Example:
            ```python
            from llmengine import FineTune

            response = FineTune.retrieve(
                fine_tune_id="ft_abc123",
            )

            print(response.json())
            ```

        JSON Response:
            ```json
            {
                "fine_tune_id": "ft_abc123",
                "status": "RUNNING"
            }
            ```

        Args:
            fine_tune_id (`str`):
                ID of the fine-tuning job

        Returns:
            GetFineTuneResponse: an object that contains the ID and status of the requested job
        """
        response = cls.get(f"v1/llm/fine-tunes/{fine_tune_id}", timeout=DEFAULT_TIMEOUT)
        return GetFineTuneResponse.parse_obj(response)

    @classmethod
    def list(cls) -> ListFineTunesResponse:
        """
        List fine-tuning jobs

        Example:
            ```python
            from llmengine import FineTune

            response = FineTune.list()
            print(response.json())
            ```

        JSON Response:
            ```json
            [
                {
                    "fine_tune_id": "ft_abc123",
                    "status": "RUNNING"
                },
                {
                    "fine_tune_id": "ft_def456",
                    "status": "SUCCESS"
                }
            ]
            ```

        Returns:
            ListFineTunesResponse: an object that contains a list of all fine-tuning jobs and their statuses
        """
        response = cls.get("v1/llm/fine-tunes", timeout=DEFAULT_TIMEOUT)
        return ListFineTunesResponse.parse_obj(response)

    @classmethod
    def cancel(cls, fine_tune_id: str) -> CancelFineTuneResponse:
        """
        Cancel a fine-tuning job

        Example:
            ```python
            from llmengine import FineTune

            response = FineTune.cancel(fine_tune_id="ft_abc123")
            print(response.json())
            ```

        JSON Response:
            ```json
            {
                "success": "true"
            }
            ```


        Args:
            fine_tune_id (`str`):
                ID of the fine-tuning job

        Returns:
            CancelFineTuneResponse: an object that contains whether the cancellation was successful
        """
        response = cls.put(
            f"v1/llm/fine-tunes/{fine_tune_id}/cancel",
            data=None,
            timeout=DEFAULT_TIMEOUT,
        )
        return CancelFineTuneResponse.parse_obj(response)
