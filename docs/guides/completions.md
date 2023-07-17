An LLM is a Machine Learning model that is given an initial text prompt and then generates a completion for that prompt. LLMs generate completions in different ways - foundation models pick the most likely completion based on their training over a large corpus of documents, instruction-tuned models complete text by trying to follow instructions provided in the prompt, and other models can be fine-tuned to respond in distinct ways.

The LLMEngine repository contains a `Completion` API that is used to get a response from an LLM.

A full reference of the Completions API is [here](/api/python_client/#llmengine.Completion)

There are two primary entry points to the API - `Completion.create` for synchronous calls and `Completion.acreate` for asynchronous calls via python `asyncio`.

Both methods in the API are given the name of the model and the initial prompt as mandatory input parameters.

Optional parameters can be specified for max_new_tokens, temperature, timeout in seconds, and whether responses from the model should be streamed or returned when completed.

## Sync API

In the [Getting Started](./getting_started) guide, we provided a very simple way of using the Completion API to make a call to a model:

=== "Basic example"
```py
from llmengine import Completion

response = Completion.create(
  model_name="llama-7b",
  prompt="Suggest a name for an icecream shop")
print(response.outputs[0].text)
```

Different models from the [Model Zoo](./model_zoo) can be referenced in a similar manner:

=== "Using different LLMs"
```py
from llmengine import Completion

# Iterate over a few models in the Model Zoo
for model_name in ["llama-7b", "falcon-7b", "falcon-7b-instruct"]:
  response = Completion.create(
    model_name=model_name,
    prompt="Suggest a name for an icecream shop",
    max_new_tokens=100)
  print(f"Model: {model_name}\nResponse: {response.outputs[0].text}")
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
    print(f"Model: {model_name} Temperature: {temperature}\nResponse: {response.outputs[0].text}")
```

For synchronous calls like the ones above, the response is of type `CompletionSyncV1Response`.

The full reference of that class is [here](/api/python_client/#llmengine.CompletionSyncV1Response).

The `outputs` field contains a list of elements of type [`CompletionOutput`](/api/python_client/#llmengine.CompletionOutput), each element of which contains a `text` string with the completion, and optional elements for the number of tokens in the prompt and in the completion.

=== "Response Format"
```py
from llmengine import Completion

response = Completion.create(
  model_name="falcon-7b-instruct",
  prompt="Suggest a name for an icecream shop")
print(response.outputs)

# The output of this command is
# [CompletionOutput(text='\nIcy Creamery', num_prompt_tokens=None, num_completion_tokens=6)]
```

### Sync API with streaming responses

It is also possible to set the `stream` argument in `Completion.create` to `True` in order to get streaming responses back from the LLM. The API supports token streaming to reduce perceived latency for certain applications. When streaming, tokens will be sent as data-only [server-side events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format).

=== "Streaming responses"
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

# JSON responses here are of this form:
# {"output": {"text": "\\n", "finished": false, "num_prompt_tokens": null, "num_completion_tokens": 1 }, "traceback": null }
# {"output": {"text": "I", "finished": false, "num_prompt_tokens": null, "num_completion_tokens": 2 }, "traceback": null }
# {"output": {"text": " don", "finished": false, "num_prompt_tokens": null, "num_completion_tokens": 3 }, "traceback": null }
# {"output": {"text": "’", "finished": false, "num_prompt_tokens": null, "num_completion_tokens": 4 }, "traceback": null }
# {"output": {"text": "t", "finished": true, "num_prompt_tokens": null, "num_completion_tokens": 5 }, "traceback": null }
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
# {"outputs": [{"text": "\nI scream, you scream, we all scream for ice cream!", "num_prompt_tokens": null, "num_completion_tokens": 15}], "traceback": null}
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
# {"output": {"text": "\n", "finished": false, "num_prompt_tokens": null, "num_completion_tokens": 1}, "traceback": null}
# {"output": {"text": "I", "finished": false, "num_prompt_tokens": null, "num_completion_tokens": 2}, "traceback": null}
# {"output": {"text": "gl", "finished": false, "num_prompt_tokens": null, "num_completion_tokens": 3}, "traceback": null}
# {"output": {"text": "oo", "finished": false, "num_prompt_tokens": null, "num_completion_tokens": 4}, "traceback": null}
# {"output": {"text": " Cream", "finished": false, "num_prompt_tokens": null, "num_completion_tokens": 5}, "traceback": null}
# {"output": {"text": "ery", "finished": false, "num_prompt_tokens": null, "num_completion_tokens": 6}, "traceback": null}
# {"output": {"text": "<|endoftext|>", "finished": true, "num_prompt_tokens": null, "num_completion_tokens": 7}, "traceback": null}
```

