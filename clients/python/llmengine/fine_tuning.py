from typing import Any, Dict, Optional, Union

from llmengine.api_engine import DEFAULT_TIMEOUT, APIEngine
from llmengine.data_types import (
    CancelFineTuneResponse,
    CreateFineTuneRequest,
    CreateFineTuneResponse,
    GetFineTuneEventsResponse,
    GetFineTuneResponse,
    ListFineTunesResponse,
)


class FineTune(APIEngine):
    """
    FineTune API. This API is used to fine-tune models.

    Fine-tuning is a process where the LLM is further trained on a task-specific dataset, allowing the model to adjust its parameters to better align with the task at hand. Fine-tuning is a supervised training phase, where prompt/response pairs are provided to optimize the performance of the LLM. LLM Engine currently uses [LoRA](https://arxiv.org/abs/2106.09685) for fine-tuning. Support for additional fine-tuning methods is upcoming.

    LLM Engine provides APIs to create fine-tunes on a base model with training & validation datasets. APIs are also provided to list, cancel and retrieve fine-tuning jobs.

    Creating a fine-tune will end with the creation of a Model, which you can view using `Model.get(model_name)` or delete using `Model.delete(model_name)`.
    """

    @classmethod
    def create(
        cls,
        model: str,
        training_file: str,
        validation_file: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Union[str, int, float]]] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        suffix: Optional[str] = None,
    ) -> CreateFineTuneResponse:
        """
        Creates a job that fine-tunes a specified model with a given dataset.

        This API can be used to fine-tune a model. The _model_ is the name of base model
        ([Model Zoo](../../model_zoo) for available models) to fine-tune. The training
        and validation files should consist of prompt and response pairs. `training_file`
        and `validation_file` must be either publicly accessible HTTP or HTTPS URLs, or
        file IDs of files uploaded to LLM Engine's [Files API](./#llmengine.File) (these
        will have the `file-` prefix). The referenced files must be CSV files that include
        two columns: `prompt` and `response`. A maximum of 100,000 rows of data is
        currently supported. At least 200 rows of data is recommended to start to see benefits from
        fine-tuning. For sequences longer than the native `max_seq_length` of the model, the sequences
        will be truncated.

        A fine-tuning job can take roughly 30 minutes for a small dataset (~200 rows)
        and several hours for larger ones.

        Args:
            model (`str`):
                The name of the base model to fine-tune. See [Model Zoo](../../model_zoo) for the list of available models to fine-tune.

            training_file (`str`):
                Publicly accessible URL or file ID referencing a CSV file for training. When no validation_file is provided, one will automatically be created using a 10% split of the training_file data.

            validation_file (`Optional[str]`):
                Publicly accessible URL or file ID referencing a CSV file for validation. The validation file is used to compute metrics which let LLM Engine pick the best fine-tuned checkpoint, which will be used for inference when fine-tuning is complete.

            hyperparameters (`Optional[Dict[str, Union[str, int, float, Dict[str, Any]]]]`):
                A dict of hyperparameters to customize fine-tuning behavior.

                Currently supported hyperparameters:

                * `lr`: Peak learning rate used during fine-tuning. It decays with a cosine schedule afterward. (Default: 2e-3)
                * `warmup_ratio`: Ratio of training steps used for learning rate warmup. (Default: 0.03)
                * `epochs`: Number of fine-tuning epochs. This should be less than 20. (Default: 5)
                * `weight_decay`: Regularization penalty applied to learned weights. (Default: 0.001)
                * `peft_config`: A dict of parameters for the PEFT algorithm. See [LoraConfig](https://huggingface.co/docs/peft/main/en/package_reference/tuners#peft.LoraConfig) for more information.

            wandb_config (`Optional[Dict[str, Any]]`):
                A dict of configuration parameters for Weights & Biases. See [Weights & Biases](https://docs.wandb.ai/ref/python/init) for more information.
                Set `hyperparameter["report_to"]` to `wandb` to enable automatic finetune metrics logging.
                Must include `api_key` field which is the wandb API key.
                Also supports setting `base_url` to use a custom Weights & Biases server.

            suffix (`Optional[str]`):
                A string that will be added to your fine-tuned model name. If present, the entire fine-tuned model name
                will be formatted like `"[model].[suffix].[YYMMDD-HHMMSS]"`. If absent, the
                fine-tuned model name will be formatted `"[model].[YYMMDD-HHMMSS]"`.
                For example, if `suffix` is `"my-experiment"`, the fine-tuned model name could be
                `"llama-2-7b.my-experiment.230717-230150"`.
                Note: `suffix` must be between 1 and 28 characters long, and can only contain alphanumeric characters and hyphens.

        Returns:
            CreateFineTuneResponse: an object that contains the ID of the created fine-tuning job

        Here is an example script to create a 5-row CSV of properly formatted data for fine-tuning
        an airline question answering bot:

        === "Formatting data in Python"
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

        Currently, data needs to be uploaded to either a publicly accessible web URL or to LLM Engine's
        private file server so that it can be read for fine-tuning. Publicly accessible HTTP and HTTPS
        URLs are currently supported.

        To privately share data with the LLM Engine API, use LLM Engine's [File.upload](../../api/python_client/#llmengine.File.upload)
        API. You can upload data in local file to LLM Engine's private file server and then use the
        returned file ID to reference your data in the FineTune API. The file ID is generally in the
        form of `file-<random_string>`, e.g. "file-7DLVeLdN2Ty4M2m".

        Example code for fine-tuning:
        === "Fine-tuning in Python"
            ```python
            from llmengine import FineTune

            response = FineTune.create(
                model="llama-2-7b",
                training_file="file-7DLVeLdN2Ty4M2m",
            )

            print(response.json())
            ```

        === "Response in JSON"
            ```json
            {
                "fine_tune_id": "ft-cir3eevt71r003ks6il0"
            }
            ```

        """
        request = CreateFineTuneRequest(
            model=model,
            training_file=training_file,
            validation_file=validation_file,
            hyperparameters=hyperparameters,
            wandb_config=wandb_config,
            suffix=suffix,
        )
        response = cls.post_sync(
            resource_name="v1/llm/fine-tunes",
            data=request.dict(),
            timeout=DEFAULT_TIMEOUT,
        )
        return CreateFineTuneResponse.parse_obj(response)

    @classmethod
    def get(
        cls,
        fine_tune_id: str,
    ) -> GetFineTuneResponse:
        """
        Get status of a fine-tuning job.

        This API can be used to get the status of an already running
        fine-tuning job. It takes as a single parameter the `fine_tune_id`
        and returns a
        [GetFineTuneResponse](../../api/data_types/#llmengine.GetFineTuneResponse)
        object with the id and status (`PENDING`, `STARTED`,
        `UNDEFINED`, `FAILURE` or `SUCCESS`).

        Args:
            fine_tune_id (`str`):
                ID of the fine-tuning job

        Returns:
            GetFineTuneResponse: an object that contains the ID and status of the requested job

        === "Getting status of fine-tuning in Python"
            ```python
            from llmengine import FineTune

            response = FineTune.get(
                fine_tune_id="ft-cir3eevt71r003ks6il0",
            )

            print(response.json())
            ```

        === "Response in JSON"
            ```json
            {
                "fine_tune_id": "ft-cir3eevt71r003ks6il0",
                "status": "STARTED"
            }
            ```
        """
        response = cls._get(f"v1/llm/fine-tunes/{fine_tune_id}", timeout=DEFAULT_TIMEOUT)
        return GetFineTuneResponse.parse_obj(response)

    @classmethod
    def list(cls) -> ListFineTunesResponse:
        """
        List fine-tuning jobs.

        This API can be used to list all the fine-tuning jobs.
        It returns a list of pairs of `fine_tune_id` and `status` for
        all existing jobs.

        Returns:
            ListFineTunesResponse: an object that contains a list of all fine-tuning jobs and their statuses

        === "Listing fine-tuning jobs in Python"
            ```python
            from llmengine import FineTune

            response = FineTune.list()
            print(response.json())
            ```

        === "Response in JSON"
            ```json
            {
                "jobs": [
                    {
                        "fine_tune_id": "ft-cir3eevt71r003ks6il0",
                        "status": "STARTED"
                    },
                    {
                        "fine_tune_id": "ft_def456",
                        "status": "SUCCESS"
                    }
                ]
            }
            ```
        """
        response = cls._get("v1/llm/fine-tunes", timeout=DEFAULT_TIMEOUT)
        return ListFineTunesResponse.parse_obj(response)

    @classmethod
    def cancel(cls, fine_tune_id: str) -> CancelFineTuneResponse:
        """
        Cancel a fine-tuning job.

        This API can be used to cancel an existing fine-tuning job if
        it's no longer required. It takes the `fine_tune_id` as a parameter
        and returns a response object which has a `success` field
        confirming if the cancellation was successful.

        Args:
            fine_tune_id (`str`):
                ID of the fine-tuning job

        Returns:
            CancelFineTuneResponse: an object that contains whether the cancellation was successful

        === "Cancelling fine-tuning job in Python"
            ```python
            from llmengine import FineTune

            response = FineTune.cancel(fine_tune_id="ft-cir3eevt71r003ks6il0")
            print(response.json())
            ```

        === "Response in JSON"
            ```json
            {
                "success": true
            }
            ```
        """
        response = cls.put(
            f"v1/llm/fine-tunes/{fine_tune_id}/cancel",
            data=None,
            timeout=DEFAULT_TIMEOUT,
        )
        return CancelFineTuneResponse.parse_obj(response)

    @classmethod
    def get_events(cls, fine_tune_id: str) -> GetFineTuneEventsResponse:
        """
        Get events of a fine-tuning job.

        This API can be used to get the list of detailed events for a fine-tuning job.
        It takes the `fine_tune_id` as a parameter and returns a response object
        which has a list of events that has happened for the fine-tuning job. Two events
        are logged periodically: an evaluation of the training loss, and an
        evaluation of the eval loss. This API will return all events for the fine-tuning job.

        Args:
            fine_tune_id (`str`):
                ID of the fine-tuning job

        Returns:
            GetFineTuneEventsResponse: an object that contains the list of events for the fine-tuning job

        === "Getting events for  fine-tuning jobs in Python"
            ```python
            from llmengine import FineTune

            response = FineTune.get_events(fine_tune_id="ft-cir3eevt71r003ks6il0")
            print(response.json())
            ```

        === "Response in JSON"
            ```json
            {
                "events":
                [
                    {
                        "timestamp": 1689665099.6704428,
                        "message": "{'loss': 2.108, 'learning_rate': 0.002, 'epoch': 0.7}",
                        "level": "info"
                    },
                    {
                        "timestamp": 1689665100.1966307,
                        "message": "{'eval_loss': 1.67730712890625, 'eval_runtime': 0.2023, 'eval_samples_per_second': 24.717, 'eval_steps_per_second': 4.943, 'epoch': 0.7}",
                        "level": "info"
                    },
                    {
                        "timestamp": 1689665105.6544185,
                        "message": "{'loss': 1.8961, 'learning_rate': 0.0017071067811865474, 'epoch': 1.39}",
                        "level": "info"
                    },
                    {
                        "timestamp": 1689665106.159139,
                        "message": "{'eval_loss': 1.513688564300537, 'eval_runtime': 0.2025, 'eval_samples_per_second': 24.696, 'eval_steps_per_second': 4.939, 'epoch': 1.39}",
                        "level": "info"
                    }
                ]
            }
            ```
        """
        response = cls._get(
            f"v1/llm/fine-tunes/{fine_tune_id}/events",
            timeout=DEFAULT_TIMEOUT,
        )
        return GetFineTuneEventsResponse.parse_obj(response)
