from typing import AsyncIterable, Iterator, Union

from spellbook_serve_client.api_engine import APIEngine
from spellbook_serve_client.data_types import (
    CompletionStreamV1Request,
    CompletionStreamV1Response,
    CompletionSyncV1Request,
    CompletionSyncV1Response,
)


class Completion(APIEngine):
    """
    Completion API. This API is used to generate text completions.

    Example:
        ```python
        from spellbook_serve_client import Completion

        response = Completion.create(
            model_name="llama-7b",
            prompt="Hello, my name is",
            max_new_tokens=10,
            temperature=0.2,
        )
        print(response.outputs[0].text)
        ```

    Example:
        ```python
        from spellbook_serve_client import Completion

        response_stream = Completion.create(
            model_name="llama-7b",
            prompt="Hello, my name is",
            max_new_tokens=10,
            temperature=0.2,
            stream=True,
        )
        for response in response_stream:
            print(response.output.text)
        ```

    Example:
        ```python
        from spellbook_serve_client import Completion

        async def main():
            response_stream = await Completion.acreate(
                model_name="llama-7b",
                prompt="Hello, my name is",
                max_new_tokens=10,
                temperature=0.2,
                stream=True,
            )
            async for response in response_stream:
                print(response.output.text)
        ```
    """

    @classmethod
    async def acreate(
        cls,
        model_name: str,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.2,
        timeout: int = 10,
        stream: bool = False,
    ) -> Union[CompletionSyncV1Response, AsyncIterable[CompletionStreamV1Response]]:
        """
        Create a completion task.

        Args:
            model_name (str):
                Model name to use for inference
            prompt (str):
                Input text
            max_new_tokens (int):
                Maximum number of generated tokens
            temperature (float):
                The value used to module the logits distribution.
            timeout (int):
                Timeout in seconds
            stream (bool):
                Whether to stream the response. If true, the return type is an
                `Iterator[CompletionStreamV1Response]`.

        Returns:
            response (CompletionStreamV1Response): generated response or iterator of response chunks
        """
        if stream:

            async def _acreate_stream(**kwargs) -> AsyncIterable[CompletionStreamV1Response]:
                data = CompletionStreamV1Request(**kwargs).dict()
                response = cls.apost_stream(
                    resource_name=f"v1/llm/completions-stream?model_endpoint_name={model_name}",
                    data=data,
                    timeout=timeout,
                )
                async for chunk in response:
                    yield CompletionStreamV1Response.parse_obj(chunk)

            return _acreate_stream(
                model_name=model_name,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                timeout=timeout,
            )

        else:

            async def _acreate_sync(**kwargs) -> CompletionSyncV1Response:
                data = CompletionSyncV1Request(**kwargs).dict()
                response = await cls.apost_sync(
                    resource_name=f"v1/llm/completions-sync?model_endpoint_name={model_name}",
                    data=data,
                    timeout=timeout,
                )
                return CompletionSyncV1Response.parse_obj(response)

            return await _acreate_sync(
                prompts=[prompt], max_new_tokens=max_new_tokens, temperature=temperature
            )

    @classmethod
    def create(
        cls,
        model_name: str,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.2,
        timeout: int = 10,
        stream: bool = False,
    ) -> Union[CompletionSyncV1Response, Iterator[CompletionStreamV1Response]]:
        """
        Create a completion task.

        Args:
            model_name (str):
                Model name to use for inference
            prompt (str):
                Input text
            max_new_tokens (int):
                Maximum number of generated tokens
            temperature (float):
                The value used to module the logits distribution.
            timeout (int):
                Timeout in seconds
            stream (bool):
                Whether to stream the response. If true, the return type is an `Iterator`.

        Returns:
            response (CompletionStreamV1Response): generated response or iterator of response chunks
        """
        if stream:

            def _create_stream(**kwargs):
                data_stream = CompletionStreamV1Request(**kwargs).dict()
                response_stream = cls.post_stream(
                    resource_name=f"v1/llm/completions-stream?model_endpoint_name={model_name}",
                    data=data_stream,
                    timeout=timeout,
                )
                for chunk in response_stream:
                    yield CompletionStreamV1Response.parse_obj(chunk)

            return _create_stream(
                prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature
            )

        else:
            data = CompletionSyncV1Request(
                prompts=[prompt], max_new_tokens=max_new_tokens, temperature=temperature
            ).dict()
            response = cls.post_sync(
                resource_name=f"v1/llm/completions-sync?model_endpoint_name={model_name}",
                data=data,
                timeout=timeout,
            )
            return CompletionSyncV1Response.parse_obj(response)
