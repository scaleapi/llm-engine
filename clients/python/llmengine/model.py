from llmengine.api_engine import DEFAULT_TIMEOUT, APIEngine
from llmengine.data_types import (
    DeleteLLMEndpointResponse,
    GetLLMEndpointResponse,
    ListLLMEndpointsResponse,
)


class Model(APIEngine):
    """
    Model API. This API is used to get, list, and delete models. Models include both base
    models built into LLM Engine, and fine-tuned models that you create through the
    [FineTune.create()](./#llmengine.fine_tuning.FineTune.create) API.

    See [Model Zoo](../../model_zoo) for the list of publicly available base models.
    """

    @classmethod
    def get(
        cls,
        model: str,
    ) -> GetLLMEndpointResponse:
        """
        Get information about an LLM model.

        This API can be used to get information about a Model's source and inference framework.
        For self-hosted users, it returns additional information about number of shards, quantization, infra settings, etc.
        The function takes as a single parameter the name `model`
        and returns a
        [GetLLMEndpointResponse](../../api/data_types/#llmengine.GetLLMEndpointResponse)
        object.

        Args:
            model (`str`):
                Name of the model

        Returns:
            GetLLMEndpointResponse: object representing the LLM and configurations

        === "Accessing model in python"
            ```python
            from llmengine import Model

            response = Model.get("llama-7b.suffix.2023-07-18-12-00-00")

            print(response.json())
            ```

        === "Response in json"
            ```json
            {
                "id": null,
                "name": "llama-7b.suffix.2023-07-18-12-00-00",
                "model_name": null,
                "source": "hugging_face",
                "inference_framework": "text_generation_inference",
                "inference_framework_tag": null,
                "num_shards": null,
                "quantize": null,
                "spec": null
            }
            ```
        """
        response = cls._get(f"v1/llm/model-endpoints/{model}", timeout=DEFAULT_TIMEOUT)
        return GetLLMEndpointResponse.parse_obj(response)

    @classmethod
    def list(cls) -> ListLLMEndpointsResponse:
        """
        List LLM models available to call inference on.

        This API can be used to list all available models, including both publicly
        available models and user-created fine-tuned models.
        It returns a list of
        [GetLLMEndpointResponse](../../api/data_types/#llmengine.GetLLMEndpointResponse)
        objects for all models. The most important field is the model `name`.

        Returns:
            ListLLMEndpointsResponse: list of models

        === "Listing available modes in python"
            ```python
            from llmengine import Model

            response = Model.list()
            print(response.json())
            ```

        === "Response in json"
            ```json
            {
                "model_endpoints": [
                    {
                        "id": null,
                        "name": "llama-7b.suffix.2023-07-18-12-00-00",
                        "model_name": null,
                        "source": "hugging_face",
                        "inference_framework": "text_generation_inference",
                        "inference_framework_tag": null,
                        "num_shards": null,
                        "quantize": null,
                        "spec": null
                    },
                    {
                        "id": null,
                        "name": "llama-7b",
                        "model_name": null,
                        "source": "hugging_face",
                        "inference_framework": "text_generation_inference",
                        "inference_framework_tag": null,
                        "num_shards": null,
                        "quantize": null,
                        "spec": null
                    },
                    {
                        "id": null,
                        "name": "llama-13b-deepspeed-sync",
                        "model_name": null,
                        "source": "hugging_face",
                        "inference_framework": "deepspeed",
                        "inference_framework_tag": null,
                        "num_shards": null,
                        "quantize": null,
                        "spec": null
                    },
                    {
                        "id": null,
                        "name": "falcon-40b",
                        "model_name": null,
                        "source": "hugging_face",
                        "inference_framework": "text_generation_inference",
                        "inference_framework_tag": null,
                        "num_shards": null,
                        "quantize": null,
                        "spec": null
                    }
                ]
            }
            ```
        """
        response = cls._get("v1/llm/model-endpoints", timeout=DEFAULT_TIMEOUT)
        return ListLLMEndpointsResponse.parse_obj(response)

    @classmethod
    def delete(cls, model: str) -> DeleteLLMEndpointResponse:
        """
        Deletes an LLM model.

        This API can be used to delete a fine-tuned model. It takes
        as parameter the name of the `model` and returns a response
        object which has a `deleted` field confirming if the deletion
        was successful. If called on a base model included with LLM
        Engine, an error will be thrown.

        Args:
            model (`str`):
                Name of the model

        Returns:
            response: whether the model was successfully deleted

        === "Deleting model in python"
            ```python
            from llmengine import Model

            response = Model.delete("llama-7b.suffix.2023-07-18-12-00-00")
            print(response.json())
            ```

        === "Response in json"
            ```json
            {
                "deleted": true
            }
            ```
        """
        response = cls._delete(f"v1/llm/model-endpoints/{model}", timeout=DEFAULT_TIMEOUT)
        return DeleteLLMEndpointResponse.parse_obj(response)
