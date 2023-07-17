LLM Engine provides a list of open source language models (see [Model Zoo](/model_zoo)) that can be used for producing 
Completions.  An example API call looks as follows:
```python
from llmengine import Completion

response = Completion.create(
    model_name="llama-7b",
    prompt="Hello, my name is",
    max_new_tokens=10,
    temperature=0.2,
)
```

See the full [API reference documentation](/api/python_client/#llmengine.Completion) to learn more.

## Completions response format

An example completions API response looks as follows:

```json
{
    "outputs":
    [
        {
            "text": "_______ and I am a _______",
            "num_completion_tokens": 10
        }
    ],
}
```

In Python, the response is of type [CompletionSyncV1Response](/api/python_client/#llmengine.CompletionSyncV1Response), 
which maps to the above JSON structure.

## Token streaming

The Completions API support token streaming to reduce perceived latency for certain applications. When streaming, 
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

See the [Model Zoo](/model_zoo) for more information on best practices for which model to use for Completions.
