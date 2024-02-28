Language Models are trained to predict natural language and provide text outputs as a response
to their inputs. The inputs are called _prompts_ and outputs are referred to as _completions_.
LLMs take the input _prompts_ and chunk them into smaller units called _tokens_ to process and
generate language. Tokens may include trailing spaces and even sub-words. This process is
language dependent.

Scale's LLM Engine provides access to open source language models (see [Model Zoo](../../model_zoo))
that can be used for producing completions to prompts.

## Completion API call

An example API call looks as follows:

=== "Completion call in Python"
```python
from llmengine import Completion

response = Completion.create(
    model="llama-2-7b",
    prompt="Hello, my name is",
    max_new_tokens=10,
    temperature=0.2,
)

print(response.json())
# '{"request_id": "c4bf0732-08e0-48a8-8b44-dfe8d4702fb0", "output": {"text": "________ and I am a ________", "num_completion_tokens": 10}}'

print(response.output.text)
# ________ and I am a ________
```

- **model:** The LLM you want to use (see [Model Zoo](../../model_zoo)).
- **prompt:** The main input for the LLM to respond to.
- **max_new_tokens:** The maximum number of tokens to generate in the chat completion.
- **temperature:** The sampling temperature to use. Higher values make the output more random,
  while lower values will make it more focused and deterministic.
  When temperature is 0 [greedy search](https://huggingface.co/docs/transformers/generation_strategies#greedy-search) is used.

See the full [Completion API reference documentation](../../api/python_client/#llmengine.Completion) to learn more.

## Completion API response

An example Completion API response looks as follows:

=== "Response in JSON"
    ```python
        >>> print(response.json())
        {
          "request_id": "c4bf0732-08e0-48a8-8b44-dfe8d4702fb0",
          "output": {
            "text": "_______ and I am a _______",
            "num_completion_tokens": 10
          }
        }
    ```
=== "Response in Python"
    ```python
        >>> print(response.output.text)
        _______ and I am a _______
    ```

## Token streaming

The Completions API supports token streaming to reduce _perceived_ latency for certain
applications. When streaming, tokens will be sent as data-only
[server-side events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format).

To enable token streaming, pass `stream=True` to either [Completion.create](../../api/python_client/#llmengine.completion.Completion.create) or [Completion.acreate](../../api/python_client/#llmengine.completion.Completion.acreate).

Note that errors from streaming calls are returned back to the user as plain-text messages and currently need to be handled by the client.

An example of token streaming using the synchronous Completions API looks as follows:

=== "Token streaming with synchronous API in python"

```python
import sys

from llmengine import Completion

stream = Completion.create(
    model="llama-2-7b",
    prompt="Give me a 200 word summary on the current economic events in the US.",
    max_new_tokens=1000,
    temperature=0.2,
    stream=True,
)

for response in stream:
    if response.output:
        print(response.output.text, end="")
        sys.stdout.flush()
    else: # an error occurred
        print(response.error) # print the error message out 
        break
```

## Async requests

The Python client supports `asyncio` for creating Completions. Use [Completion.acreate](../../api/python_client/#llmengine.completion.Completion.acreate) instead of [Completion.create](../../api/python_client/#llmengine.completion.Completion.create)
to utilize async processing. The function signatures are otherwise identical.

An example of async Completions looks as follows:

=== "Completions with asynchronous API in python"

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

## Batch completions

The Python client also supports batch completins. Batch completions supports distributing data to multiple workers to accelerate inference. It also tries to maximize throughput so the completions should finish quite a bit faster than hitting models through HTTP. Use [Completion.batch_complete](../../api/python_client/#llmengine.completion.Completion.batch_complete) to utilize batch completions.

Some examples of batch completions:

=== "Batch completions with prompts in the request"
```python
from llmengine import Completion
from llmengine.data_types import CreateBatchCompletionsModelConfig, CreateBatchCompletionsRequestContent

content = CreateBatchCompletionsRequestContent(
    prompts=["What is deep learning", "What is a neural network"],
    max_new_tokens=10,
    temperature=0.0
)

response = Completion.batch_create(
    output_data_path="s3://my-path",
    model_config=CreateBatchCompletionsModelConfig(
        model="llama-2-7b",
        checkpoint_path="s3://checkpoint-path",
        labels={"team":"my-team", "product":"my-product"}
    ),
    content=content
)
print(response.job_id)
```

=== "Batch completions with prompts in a file and with 2 parallel jobs"
```python
from llmengine import Completion
from llmengine.data_types import CreateBatchCompletionsModelConfig, CreateBatchCompletionsRequestContent

# Store CreateBatchCompletionsRequestContent data into input file "s3://my-input-path"

response = Completion.batch_create(
    input_data_path="s3://my-input-path",
    output_data_path="s3://my-output-path",
    model_config=CreateBatchCompletionsModelConfig(
        model="llama-2-7b",
        checkpoint_path="s3://checkpoint-path",
        labels={"team":"my-team", "product":"my-product"}
    ),
    data_parallelism=2
)
print(response.job_id)
```

## Which model should I use?

See the [Model Zoo](../../model_zoo) for more information on best practices for which model to use for Completions.
