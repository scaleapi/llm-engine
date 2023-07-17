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

    Fine-tuning is a process where the LLM is further trained on a task-specific dataset, allowing the model to adjust its parameters to better align with the task at hand. Fine-tuning involves the supervised training phase, where prompt/response pairs are provided to optimize the performance of the LLM.

    Scale llm-engine provides APIs to create fine-tunes on a base-model with training & validation data-sets. APIs are also provided to list, cancel and retrieve fine-tuning jobs.
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

        Args:
            model (`str`):
                The name of the base model to fine-tune. See [Model Zoo](../../model_zoo) for the list of available models to fine-tune.

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

        The _model_ is the name of base model ([Model Zoo](../../model_zoo) for available models) to fine. The training
        file should consist of prompt and response pairs. Your data must be formatted as a CSV file
        that includes two columns: `prompt` and `response`. A maximum of 100,000 rows of data is
        currently supported. At least 200 rows of data is recommended to start to see benefits from
        fine-tuning.

        Here is an example script to create a 5-row CSV of properly formatted data for fine-tuning
        an airline question answering bot:

        ```python
        import csv

        # Define data
        data = [
          ("What is your policy on carry-on luggage?", "Our policy allows each passenger to bring one piece of carry-on luggage and one personal item such as a purse or briefcase. The maximum size for carry-on luggage is 22 x 14 x 9 inches."),
          ("How can I change my flight?", "You can change your flight through our website or mobile app. Go to 'Manage my booking' section, enter your booking reference and last name, then follow the prompts to change your flight."),
          ("What meals are available on my flight?", "We offer a variety of meals depending on the flight's duration and route. These can range from snacks and light refreshments to full-course meals on long-haul flights. Specific meal options can be viewed during the booking process."),
          ("How early should I arrive at the airport before my flight?", "We recommend arriving at least two hours before domestic flights and three hours before international flights."),
          "Can I select my seat in advance?", "Yes, you can select your seat during the booking process or afterwards via the 'Manage my booking' section on our website or mobile app."),
          ]

        # Write data to a CSV file
        with open('customer_service_data.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["prompt", "response"])
            writer.writerows(data)
        ```

        Example code for fine-tuning:
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

        Args:
            fine_tune_id (`str`):
                ID of the fine-tuning job

        Returns:
            GetFineTuneResponse: an object that contains the ID and status of the requested job

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

        """
        response = cls.get(f"v1/llm/fine-tunes/{fine_tune_id}", timeout=DEFAULT_TIMEOUT)
        return GetFineTuneResponse.parse_obj(response)

    @classmethod
    def list(cls) -> ListFineTunesResponse:
        """
        List fine-tuning jobs

        Returns:
            ListFineTunesResponse: an object that contains a list of all fine-tuning jobs and their statuses
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
        """
        response = cls.get("v1/llm/fine-tunes", timeout=DEFAULT_TIMEOUT)
        return ListFineTunesResponse.parse_obj(response)

    @classmethod
    def cancel(cls, fine_tune_id: str) -> CancelFineTuneResponse:
        """
        Cancel a fine-tuning job

        Args:
            fine_tune_id (`str`):
                ID of the fine-tuning job

        Returns:
            CancelFineTuneResponse: an object that contains whether the cancellation was successful

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

        """
        response = cls.put(
            f"v1/llm/fine-tunes/{fine_tune_id}/cancel",
            data=None,
            timeout=DEFAULT_TIMEOUT,
        )
        return CancelFineTuneResponse.parse_obj(response)
