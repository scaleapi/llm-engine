# LLM Engine

The LLM Engine Python library provides a convenient way of interfacing with a
`llmengine` endpoint running on
[LLM Engine](https://scaleapi.github.io/llm-engine/) or on your own infrastructure.

## Get Started

### Install

```shell
pip install scale-llm-engine
```

### Usage

If you are using LLM Engine, you can get your API key from
[https://spellbook.scale.com/settings](https://spellbook.scale.com/settings).
Set the `SCALE_API_KEY` environment variable to your API key.

If you are using your own infrastructure, you can set the
`LLM_ENGINE_BASE_PATH` environment variable to the base URL of your
self-hosted `llmengine` endpoint.

```python
from llmengine import Completion

response = Completion.create(
    model="llama-2-7b",
    prompt="Hello, my name is",
    max_new_tokens=10,
    temperature=0.2,
)
print(response.outputs[0].text)
```

## Documentation

Documentation is available at
[https://scaleapi.github.io/llm-engine/](https://scaleapi.github.io/llm-engine/).
