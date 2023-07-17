Large Language Models are trained to understand natural language and provide text outputs as a response to
their inputs. The inputs are called _prompts_ and outputs are referred to as _completions_.
LLMs take the input _prompts_ and chunk them smaller units called _tokens_ to process and generate
language. Tokens may include trailing spaces and even sub-words; this process is language dependent.

The LLMEngine provides access to open source language models (see [Model Zoo](../../model_zoo)) that can be used for producing completions to prompts.

The repository contains a `Completion` API that is used to send a prompt to an LLM and get back a completion.

A full reference of the `Completion` API is [here](/api/python_client/#llmengine.Completion)

There are two primary entry points to the API - `Completion.create` for synchronous calls and `Completion.acreate` for asynchronous calls via python `asyncio`.

Both methods in the API are given the name of the model and the initial prompt as mandatory input parameters.

Optional parameters can be specified for max_new_tokens, temperature, timeout in seconds, and whether responses from the model should be streamed or returned when completed.

## Sync API

In the [Getting Started](../../getting_started) guide, we provided a very simple way of using the Completion API to make a call to a model:

=== "Basic example"

```py
from llmengine import Completion

response = Completion.create(
  model_name="llama-7b",
  prompt="Suggest a name for an icecream shop"
  max_new_tokens=10,
  temperature=0.2)
print(response.output.text)
```

The _model_name_ is the LLM to be used (see [Model Zoo](../../model_zoo)).
The _prompt_ is the main input for the LLM to respond to.
The _max_new_tokens_ parameter is the maximum number of tokens to generate in the chat completion.
The _temperature_ is the sampling temperature to use. Higher values make the output more random, while lower values will make it more focussed and deterministic.

Different models from the [Model Zoo](../../model_zoo) can be referenced in a similar manner:

=== "Using different LLMs"

```py
from llmengine import Completion

# Iterate over a few models in the Model Zoo
for model_name in ["llama-7b", "falcon-7b", "falcon-7b-instruct"]:
  response = Completion.create(
    model_name=model_name,
    prompt="Suggest a name for an icecream shop",
    max_new_tokens=100)
  print(f"Model: {model_name}\nResponse: {response.output.text}")
```

Similarly, we can vary the maximum number of output tokens, the temperature, and the timeouts for these calls:

=== "Using different LLMs and temperatures"

```py
from llmengine import Completion

# Iterate over a few models in the Model Zoo
for model_name in ["llama-7b", "falcon-7b", "falcon-7b-instruct"]:
  for temperature in [0.1, 0.5, 0.9]:
    response = Completion.create(
      model_name=model_name,
      prompt="Suggest a name for an icecream shop",
      max_new_tokens=100,
      temperature=temperature,
      timeout=100)
    print(f"Model: {model_name} Temperature: {temperature}\nResponse: {response.output.text}")
```

For synchronous calls like the ones above, the response is of type [CompletionSyncV1Response](../../api/python_client/#llmengine.CompletionSyncV1Response), which maps to the above JSON structure.

The full reference of that class is [here](/api/python_client/#llmengine.CompletionSyncV1Response).

The `output` field contains a list of elements of type [`CompletionOutput`](/api/python_client/#llmengine.CompletionOutput), each element of which contains a `text` string with the completion, and optional elements for the number of tokens in the prompt and in the completion.

=== "Response Format"

```py
from llmengine import Completion

response = Completion.create(
  model_name="falcon-7b-instruct",
  prompt="Suggest a name for an icecream shop")
print(response.output)

# The output of this command is
# [CompletionOutput(text='\nIcy Creamery', num_completion_tokens=6)]
```

### Sync API with streaming responses

It is also possible to set the `stream` argument in `Completion.create` to `True` in order to get streaming responses back from the LLM. The API supports token streaming to reduce _perceived_ latency for certain applications. When streaming, tokens will be sent as data-only [server-side events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format).

=== "Streaming responses"

```python
from llmengine import Completion

Expand All
	@@ -54,30 +99,63 @@ stream = Completion.create(
for response in stream:
    if response.output:
        print(response.json())

# JSON responses here are of this form:
# {"output": {"text": "\\n", "finished": false, "num_completion_tokens": 1 }}
# {"output": {"text": "I", "finished": false, "num_completion_tokens": 2 }}
# {"output": {"text": " don", "finished": false, "num_completion_tokens": 3 }}
# {"output": {"text": "’", "finished": false, "num_completion_tokens": 4 }}
# {"output": {"text": "t", "finished": true, "num_completion_tokens": 5 }}
```

## Async API

Python's `asyncio` module can be used to make Completion calls asynchronously via `Completion.acreate`. The function signature is otherwise identical to `Completion.create`.

=== "Using python `async`"

```python
import asyncio
from llmengine import Completion

async def main():
  response = await Completion.acreate(
    model_name="falcon-7b-instruct",
    prompt="Suggest a name for an icecream shop")
  print(response.json())

asyncio.run(main())

# JSON response here is:
# {"output": {"text": "\nI scream, you scream, we all scream for ice cream!", "num_completion_tokens": 15}}
```

### Async API with streaming responses

It is also possible to set the `stream` argument in `Completion.acreate` to `True` in order to get streaming responses back from the LLM:

=== "Using python `async` and streaming responses"

```python
import asyncio
from llmengine import Completion

async def main():
  stream = await Completion.acreate(
    model_name="falcon-7b-instruct",
    prompt="Suggest a name for an icecream shop",
    stream=True)
  async for response in stream:
    if response.output:
      print(response.json())

asyncio.run(main())

# JSON responses:
# {"output": {"text": "\n", "finished": false, "num_completion_tokens": 1}}
# {"output": {"text": "I", "finished": false, "num_completion_tokens": 2}}
# {"output": {"text": "gl", "finished": false, "num_completion_tokens": 3}}
# {"output": {"text": "oo", "finished": false, "num_completion_tokens": 4}}
# {"output": {"text": " Cream", "finished": false, "num_completion_tokens": 5}}
# {"output": {"text": "ery", "finished": false, "num_completion_tokens": 6}}
# {"output": {"text": "<|endoftext|>", "finished": true, "num_completion_tokens": 7}}
```
