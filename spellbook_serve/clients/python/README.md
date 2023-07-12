# Spellbook Serve

The Spellbook Serve Python library provides a convenient way of interfacing with a
`spellbook-serve` endpoint running on
[Scale Spellbook Serve](https://spellbook.readme.io/docs/) or on your own infrastructure.

## Get Started

### Install

```shell
pip install spellbook-serve-client
```

### Usage

```python
from spellbook_serve_client import Client

client = Client("https://api.spellbook.scale.com", "flan-t5-xxl-deepspeed-sync")
client.generate("Why is the sky blue?").outputs[0].text
# ' Rayleigh scattering'

result = ""
for response in client.generate_stream("Why is the sky blue?"):
    if response.output:
        result += response.output.text
result
# ' Rayleigh scattering'
```

or with the asynchronous client:

```python
from spellbook_serve_client import AsyncClient

client = AsyncClient("https://api.spellbook.scale.com", "flan-t5-xxl-deepspeed-sync")
response = await client.generate("Why is the sky blue?")
print(response.outputs[0].text)
# ' Rayleigh scattering'

# Token Streaming
text = ""
async for response in client.generate_stream("Why is the sky blue?"):
    if not response.token.special:
        text += response.token.text

print(text)
# ' Rayleigh scattering'
```
