from spellbook_serve_client.api_engine import APIEngine, DEFAULT_TIMEOUT, assert_self_hosted
from spellbook_serve_client.data_types import (
    ListLLMModelEndpointsV1Response, GetLLMModelEndpointV1Response, CreateLLMModelEndpointV1Response, CreateLLMModelEndpointV1Request
)


class Model(APIEngine):
    """
    Model API. This API is used to retrieve, list, and create models.

    Note:
        This feature is only available for self-hosted users.

    Example:
        ```python
        from spellbook_serve_client import Model

        response = Model.list()
        print(response)
        ```
    """
    @classmethod
    @assert_self_hosted
    def create(
        cls,
        model_name: str,
    ) -> CreateLLMModelEndpointV1Response:
        """
        Create a fine-tuning job

        Args:
            model_name (`str`):
                Name of the model

        Returns:
            response: ID of the created fine-tuning job
        """
        request = CreateLLMModelEndpointV1Request(
            model_name=model_name,
        )
        response = cls.post_sync(
            resource_name="v1/model-endpoints",
            data=request.dict(),
            timeout=DEFAULT_TIMEOUT,
        )
        return CreateLLMModelEndpointV1Response.parse_obj(response)

    @classmethod
    @assert_self_hosted
    def retrieve(
        cls,
        model_name: str,
    ) -> GetLLMModelEndpointV1Response:
        """
        Get an LLM model endpoint

        Args:
            model_name (`str`):
                Name of the model

        Returns:
            response: object representing the LLM endpoint and configurations
        """
        response = cls.get(f"v1/model-endpoints/{model_name}", timeout=DEFAULT_TIMEOUT)
        return GetLLMModelEndpointV1Response.parse_obj(response)

    @classmethod
    @assert_self_hosted
    def list(cls) -> ListLLMModelEndpointsV1Response:
        """
        List model endpoints

        Returns:
            response: list of model endpoints
        """
        response = cls.get("v1/model-endpoints", timeout=DEFAULT_TIMEOUT)
        return ListLLMModelEndpointsV1Response.parse_obj(response)
