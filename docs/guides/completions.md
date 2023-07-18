Language Models are trained to understand natural language and provide text outputs as a response 
to their inputs. The inputs are called _prompts_ and outputs are referred to as _completions_. 
LLMs take the input _prompts_ and chunk them into smaller units called _tokens_ to process and 
generate language. Tokens may include trailing spaces and even sub-words. This process is 
language dependent.

Scale LLM Engine provides access to open source language models (see [Model Zoo](../../model_zoo)) 
that can be used for producing completions to prompts.

## Completion API call

An example API call looks as follows:

```python
from llmengine import Completion

response = Completion.create(
    model="llama-7b",
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

See the full [API reference documentation](../../api/python_client/#llmengine.Completion) to learn more.

## Completion API response

An example Completion API response looks as follows:

=== "Response in json"
    ```python
    >>> print(response.json())
    ```
    Example output:
    ```json
    {
      "request_id": "c4bf0732-08e0-48a8-8b44-dfe8d4702fb0",
      "outputs": [
        {
          "text": "_______ and I am a _______",
          "num_completion_tokens": 10
        }
      ]
    }
    ```
=== "Response in python"
    ```python
    >>> print(response.output.text)
    ```
    Example output:
    ```python
    _______ and I am a _______
    ```

## Token streaming

The Completions API supports token streaming to reduce _perceived_ latency for certain 
applications. When streaming, tokens will be sent as data-only 
[server-side events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format).

To enable token streaming, pass `stream=True` to either `Completion.create` or `Completion.acreate`.

An example of token streaming using the synchronous Completions API looks as follows

=== "Token streaming with synchronous API in python"
```python
import sys

from llmengine import Completion

stream = Completion.create(
    model="falcon-7b-instruct",
    prompt="Give me a 200 word summary on the current economic events in the US.",
    max_new_tokens=1000,
    temperature=0.2,
    stream=True,
)

for response in stream:
    if response.output:
        print(response.output.text, end="")
        sys.stdout.flush()
```

## Async requests

The Python client supports `asyncio` for creating Completions. Use `Completion.acreate` instead of `Completion.create`
to utilize async processing. The function signatures are otherwise identical.

An example of async Completions looks as follows

=== "Completions with asynchronous API in python"
```python
import asyncio
from llmengine import Completion

async def main():
    response = await Completion.acreate(
        model="llama-7b",
        prompt="Hello, my name is",
        max_new_tokens=10,
        temperature=0.2,
    )
    print(response.json())

asyncio.run(main())
```

## Which model should I use?

See the [Model Zoo](../../model_zoo) for more information on best practices for which model to use for Completions.
