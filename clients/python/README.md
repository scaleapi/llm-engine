# Spellbook Serve

The Spellbook Serve Python library provides a convenient way of interfacing with a
`spellbook-serve` endpoint running on
[Scale Spellbook Serve](https://scaleapi.github.io/spellbook-serve/) or on your own infrastructure.

## Get Started

### Install

```shell
pip install spellbook-serve-client
```

### Usage

If you are using Scale Spellbook Serve, you can get your API key from
[https://spellbook.scale.com/settings](https://spellbook.scale.com/settings). 
Set the `SCALE_API_KEY` environment variable to your API key.

If you are using your own infrastructure, you can set the
`SPELLBOOK_SERVE_BASE_PATH` environment variable to the base URL of your
self-hosted `spellbook-serve` endpoint.

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

## Documentation

Documentation is available at
[https://scaleapi.github.io/spellbook-serve/](https://scaleapi.github.io/spellbook-serve/).
