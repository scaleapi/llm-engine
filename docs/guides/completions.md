Language Models are trained to understand natural language and provide text outputs as a response to their inputs. The inputs are called _prompts_ and outputs are referred to as _completions_. LLMs take the input _prompts_ and chunk them smaller units called _tokens_ to process and generate language. Tokens may include trailing spaces and even sub-words, this process is language dependent.

Scale llm-engine provides access to open source language models (see [Model Zoo](../../model_zoo)) that can be used for producing completions to prompts.

## Completion API call

An example API call looks as follows:

```python
from llmengine import Completion

response = Completion.create(
    model_name="llama-7b",
    prompt="Hello, my name is",
    max_new_tokens=10,
    temperature=0.2,
)
```

The _model_name_ is the LLM to be used (see [Model Zoo](../../model_zoo)).
The _prompt_ is the main input for the LLM to respond to.
The _max_new_tokens_ parameter is the maximum number of tokens to generate in the chat completion.
The _temperature_ is the sampling temperature to use. Higher values make the output more random, while lower values will make it more focussed and deterministic.

See the full [API reference documentation](../../api/python_client/#llmengine.Completion) to learn more.

## Completion API response

An example Completion API response looks as follows:

```json
{
  "outputs": [
    {
      "text": "_______ and I am a _______",
      "num_completion_tokens": 10
    }
  ]
}
```

In Python, the response is of type [CompletionSyncV1Response](../../api/python_client/#llmengine.CompletionSyncV1Response), which maps to the above JSON structure.

```python
print( response.outputs[0].text )
```

## Token streaming

The Completions API support token streaming to reduce _perceived_ latency for certain applications. When streaming,
tokens will be sent as data-only [server-side events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format).

To enable token streaming, pass `stream=True` to either `Completion.create` or `Completion.acreate`.

An example of token streaming using the synchronous Completions API looks as follows:

```python
from llmengine import Completion

stream = Completion.create(
    model_name="llama-7b",
    prompt="why is the sky blue?",
    max_new_tokens=5,
    temperature=0.2,
    stream=True,
)

for response in stream:
    if response.output:
        print(response.json())
```

## Async requests

The Python client supports `asyncio` for creating Completions. Use `Completion.acreate` instead of `Completion.create`
to utilize async processing. The function signatures are otherwise identical.

An example of async Completions looks as follows:

```python
import asyncio
from llmengine import Completion

async def main():
    response = await Completion.acreate(
        model_name="llama-7b",
        prompt="Hello, my name is",
        max_new_tokens=10,
        temperature=0.2,
    )
    print(response.json())

asyncio.run(main())
```

## Which model should I use?

See the [Model Zoo](../../model_zoo) for more information on best practices for which model to use for Completions.
