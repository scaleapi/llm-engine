from typing import AsyncIterable, Iterator, List, Optional, Union

from llmengine.api_engine import APIEngine
from llmengine.data_types import (
    CompletionStreamResponse,
    CompletionStreamV1Request,
    CompletionSyncResponse,
    CompletionSyncV1Request,
)

COMPLETION_TIMEOUT = 300


class Completion(APIEngine):
    """
    Completion API. This API is used to generate text completions.

    Language models are trained to understand natural language and predict text outputs as a response to
    their inputs. The inputs are called _prompts_ and the outputs are referred to as _completions_.
    LLMs take the input prompts and chunk them into smaller units called _tokens_ to process and generate
    language. Tokens may include trailing spaces and even sub-words; this process is language dependent.

    The Completion API can be run either synchronous or asynchronously (via Python `asyncio`).
    For each of these modes, you can also choose whether to stream token responses or not.
    """

    @classmethod
    async def acreate(
        cls,
        model: str,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.2,
        stop_sequences: Optional[List[str]] = None,
        return_token_log_probs: Optional[bool] = False,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        timeout: int = COMPLETION_TIMEOUT,
        stream: bool = False,
    ) -> Union[CompletionSyncResponse, AsyncIterable[CompletionStreamResponse]]:
        """
        Creates a completion for the provided prompt and parameters asynchronously (with `asyncio`).

        This API can be used to get the LLM to generate a completion *asynchronously*.
        It takes as parameters the `model` ([see Model Zoo](../../model_zoo)) and the `prompt`.
        Optionally it takes `max_new_tokens`, `temperature`, `timeout` and `stream`.
        It returns a
        [CompletionSyncResponse](../../api/data_types/#llmengine.CompletionSyncResponse)
        if `stream=False` or an async iterator of
        [CompletionStreamResponse](../../api/data_types/#llmengine.CompletionStreamResponse)
        with `request_id` and `outputs` fields.

        Args:
            model (str):
                Name of the model to use. See [Model Zoo](../../model_zoo) for a list of Models that are supported.
            prompt (str):
                The prompt to generate completions for, encoded as a string.

            max_new_tokens (int):
                The maximum number of tokens to generate in the completion.

                The token count of your prompt plus `max_new_tokens` cannot exceed the model's context length. See
                [Model Zoo](../../model_zoo) for information on each supported model's context length.

            temperature (float):
                What sampling temperature to use, in the range `[0, 1]`. Higher values like 0.8 will make the output
                more random, while lower values like 0.2 will make it more focused and deterministic.
                When temperature is 0 [greedy search](https://huggingface.co/docs/transformers/generation_strategies#greedy-search) is used.

            stop_sequences (Optional[List[str]]):
                One or more sequences where the API will stop generating tokens for the current completion.

            return_token_log_probs (Optional[bool]):
                Whether to return the log probabilities of generated tokens.
                When True, the response will include a list of tokens and their log probabilities.

            presence_penalty (Optional[float]):
                Only supported in vllm, lightllm
                Penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
                https://platform.openai.com/docs/guides/gpt/parameter-details
                Range: [0.0, 2.0]. Higher values encourage the model to use new tokens.

            frequency_penalty (Optional[float]):
                Only supported in vllm, lightllm
                Penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
                https://platform.openai.com/docs/guides/gpt/parameter-details
                Range: [0.0, 2.0]. Higher values encourage the model to use new tokens.

            top_k (Optional[int]):
                Integer that controls the number of top tokens to consider.
                Range: [1, infinity). -1 means consider all tokens.

            top_p (Optional[float]):
                Float that controls the cumulative probability of the top tokens to consider.
                Range: (0.0, 1.0]. 1.0 means consider all tokens.

            timeout (int):
                Timeout in seconds. This is the maximum amount of time you are willing to wait for a response.

            stream (bool):
                Whether to stream the response. If true, the return type is an
                `Iterator[CompletionStreamResponse]`. Otherwise, the return type is a `CompletionSyncResponse`.
                When streaming, tokens will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format).

        Returns:
            response (Union[CompletionSyncResponse, AsyncIterable[CompletionStreamResponse]]): The generated response (if `stream=False`) or iterator of response chunks (if `stream=True`)

        === "Asynchronous completion without token streaming in Python"
            ```python
            import asyncio
            from llmengine import Completion

            async def main():
                response = await Completion.acreate(
                    model="llama-2-7b",
                    prompt="Hello, my name is",
                    max_new_tokens=10,
                    temperature=0.2,
                )
                print(response.json())

            asyncio.run(main())
            ```

        === "Response in JSON"
            ```json
            {
                "request_id": "9cfe4d5a-f86f-4094-a935-87f871d90ec0",
                "output": {
                    "text": "_______ and I am a _______",
                    "num_completion_tokens": 10
                }
            }
            ```

        Token streaming can be used to reduce _perceived_ latency for applications. Here is how applications can use streaming:

        === "Asynchronous completion with token streaming in Python"
            ```python
            import asyncio
            from llmengine import Completion

            async def main():
                stream = await Completion.acreate(
                    model="llama-2-7b",
                    prompt="why is the sky blue?",
                    max_new_tokens=5,
                    temperature=0.2,
                    stream=True,
                )

                async for response in stream:
                    if response.output:
                        print(response.json())

            asyncio.run(main())
            ```

        === "Response in JSON"
            ```json
            {"request_id": "9cfe4d5a-f86f-4094-a935-87f871d90ec0", "output": {"text": "\\n", "finished": false, "num_completion_tokens": 1}}
            {"request_id": "9cfe4d5a-f86f-4094-a935-87f871d90ec0", "output": {"text": "I", "finished": false, "num_completion_tokens": 2}}
            {"request_id": "9cfe4d5a-f86f-4094-a935-87f871d90ec0", "output": {"text": " think", "finished": false, "num_completion_tokens": 3}}
            {"request_id": "9cfe4d5a-f86f-4094-a935-87f871d90ec0", "output": {"text": " the", "finished": false, "num_completion_tokens": 4}}
            {"request_id": "9cfe4d5a-f86f-4094-a935-87f871d90ec0", "output": {"text": " sky", "finished": true, "num_completion_tokens": 5}}
            ```
        """
        if stream:

            async def _acreate_stream(
                **kwargs,
            ) -> AsyncIterable[CompletionStreamResponse]:
                data = CompletionStreamV1Request(**kwargs).dict()
                response = cls.apost_stream(
                    resource_name=f"v1/llm/completions-stream?model_endpoint_name={model}",
                    data=data,
                    timeout=timeout,
                )
                async for chunk in response:
                    yield CompletionStreamResponse.parse_obj(chunk)

            return _acreate_stream(
                model=model,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences,
                return_token_log_probs=return_token_log_probs,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                top_k=top_k,
                top_p=top_p,
                timeout=timeout,
            )

        else:

            async def _acreate_sync(**kwargs) -> CompletionSyncResponse:
                data = CompletionSyncV1Request(**kwargs).dict()
                response = await cls.apost_sync(
                    resource_name=f"v1/llm/completions-sync?model_endpoint_name={model}",
                    data=data,
                    timeout=timeout,
                )
                return CompletionSyncResponse.parse_obj(response)

            return await _acreate_sync(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences,
                return_token_log_probs=return_token_log_probs,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                top_k=top_k,
                top_p=top_p,
            )

    @classmethod
    def create(
        cls,
        model: str,
        prompt: str,
        max_new_tokens: int = 20,
        temperature: float = 0.2,
        stop_sequences: Optional[List[str]] = None,
        return_token_log_probs: Optional[bool] = False,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        timeout: int = COMPLETION_TIMEOUT,
        stream: bool = False,
    ) -> Union[CompletionSyncResponse, Iterator[CompletionStreamResponse]]:
        """
        Creates a completion for the provided prompt and parameters synchronously.

        This API can be used to get the LLM to generate a completion *synchronously*.
        It takes as parameters the `model` ([see Model Zoo](../../model_zoo)) and the `prompt`.
        Optionally it takes `max_new_tokens`, `temperature`, `timeout` and `stream`.
        It returns a
        [CompletionSyncResponse](../../api/data_types/#llmengine.CompletionSyncResponse)
        if `stream=False` or an async iterator of
        [CompletionStreamResponse](../../api/data_types/#llmengine.CompletionStreamResponse)
        with `request_id` and `outputs` fields.

        Args:
            model (str):
                Name of the model to use. See [Model Zoo](../../model_zoo) for a list of Models that are supported.

            prompt (str):
                The prompt to generate completions for, encoded as a string.

            max_new_tokens (int):
                The maximum number of tokens to generate in the completion.

                The token count of your prompt plus `max_new_tokens` cannot exceed the model's context length. See
                [Model Zoo](../../model_zoo) for information on each supported model's context length.

            temperature (float):
                What sampling temperature to use, in the range `[0, 1]`. Higher values like 0.8 will make the output
                more random, while lower values like 0.2 will make it more focused and deterministic.
                When temperature is 0 [greedy search](https://huggingface.co/docs/transformers/generation_strategies#greedy-search) is used.

            stop_sequences (Optional[List[str]]):
                One or more sequences where the API will stop generating tokens for the current completion.

            return_token_log_probs (Optional[bool]):
                Whether to return the log probabilities of generated tokens.
                When True, the response will include a list of tokens and their log probabilities.

            presence_penalty (Optional[float]):
                Only supported in vllm, lightllm
                Penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
                https://platform.openai.com/docs/guides/gpt/parameter-details
                Range: [0.0, 2.0]. Higher values encourage the model to use new tokens.

            frequency_penalty (Optional[float]):
                Only supported in vllm, lightllm
                Penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
                https://platform.openai.com/docs/guides/gpt/parameter-details
                Range: [0.0, 2.0]. Higher values encourage the model to use new tokens.

            top_k (Optional[int]):
                Integer that controls the number of top tokens to consider.
                Range: [1, infinity). -1 means consider all tokens.

            top_p (Optional[float]):
                Float that controls the cumulative probability of the top tokens to consider.
                Range: (0.0, 1.0]. 1.0 means consider all tokens.

            timeout (int):
                Timeout in seconds. This is the maximum amount of time you are willing to wait for a response.

            stream (bool):
                Whether to stream the response. If true, the return type is an
                `Iterator[CompletionStreamResponse]`. Otherwise, the return type is a `CompletionSyncResponse`.
                When streaming, tokens will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format).


        Returns:
            response (Union[CompletionSyncResponse, AsyncIterable[CompletionStreamResponse]]): The generated response (if `stream=False`) or iterator of response chunks (if `stream=True`)

        === "Synchronous completion without token streaming in Python"
            ```python
            from llmengine import Completion

            response = Completion.create(
                model="llama-2-7b",
                prompt="Hello, my name is",
                max_new_tokens=10,
                temperature=0.2,
            )
            print(response.json())
            ```

        === "Response in JSON"
            ```json
            {
                "request_id": "8bbd0e83-f94c-465b-a12b-aabad45750a9",
                "output": {
                    "text": "_______ and I am a _______",
                    "num_completion_tokens": 10
                }
            }
            ```

        Token streaming can be used to reduce _perceived_ latency for applications. Here is how applications can use streaming:

        === "Synchronous completion with token streaming in Python"
            ```python
            from llmengine import Completion

            stream = Completion.create(
                model="llama-2-7b",
                prompt="why is the sky blue?",
                max_new_tokens=5,
                temperature=0.2,
                stream=True,
            )

            for response in stream:
                if response.output:
                    print(response.json())
            ```

        === "Response in JSON"
            ```json
            {"request_id": "ebbde00c-8c31-4c03-8306-24f37cd25fa2", "output": {"text": "\\n", "finished": false, "num_completion_tokens": 1 } }
            {"request_id": "ebbde00c-8c31-4c03-8306-24f37cd25fa2", "output": {"text": "I", "finished": false, "num_completion_tokens": 2 } }
            {"request_id": "ebbde00c-8c31-4c03-8306-24f37cd25fa2", "output": {"text": " don", "finished": false, "num_completion_tokens": 3 } }
            {"request_id": "ebbde00c-8c31-4c03-8306-24f37cd25fa2", "output": {"text": "’", "finished": false, "num_completion_tokens": 4 } }
            {"request_id": "ebbde00c-8c31-4c03-8306-24f37cd25fa2", "output": {"text": "t", "finished": true, "num_completion_tokens": 5 } }
            ```
        """
        if stream:

            def _create_stream(**kwargs):
                data_stream = CompletionStreamV1Request(**kwargs).dict()
                response_stream = cls.post_stream(
                    resource_name=f"v1/llm/completions-stream?model_endpoint_name={model}",
                    data=data_stream,
                    timeout=timeout,
                )
                for chunk in response_stream:
                    yield CompletionStreamResponse.parse_obj(chunk)

            return _create_stream(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences,
                return_token_log_probs=return_token_log_probs,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                top_k=top_k,
                top_p=top_p,
            )

        else:
            data = CompletionSyncV1Request(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences,
                return_token_log_probs=return_token_log_probs,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                top_k=top_k,
                top_p=top_p,
            ).dict()
            response = cls.post_sync(
                resource_name=f"v1/llm/completions-sync?model_endpoint_name={model}",
                data=data,
                timeout=timeout,
            )
            return CompletionSyncResponse.parse_obj(response)
