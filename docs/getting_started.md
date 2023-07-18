# Getting Started

To start using LLM Engine's public inference and fine-tuning APIs:

=== "pip"
    ```commandline
    pip install scale-llm-engine
    ```
=== "conda"
    ```commandline
    conda install scale-llm-engine -c conda-forge
    ```

## Scale API Keys

To leverage Scale's hosted versions of these models, you will need a Scale Spellbook API key.

### Retrieving your API Key

To retrieve your API key, head to [Scale Spellbook](https://spellbook.scale.com) where
you will get a Scale API key on the [settings](https://spellbook.scale.com/settings) page.

!!! note "Different API Keys for different Scale Products"

    If you have leveraged Scale's platform for annotation work in the past, please note that your Spellbook API key will be different than the Scale Annotation API key. You will want to create a Spellbook API key before getting started.

### Set your API Key

LLM Engine leverages environment variables to access your API key.
Set this API key as the `SCALE_API_KEY` environment variable by adding the
following line to your `.zshrc` or `.bash_profile`, or by running it in the terminal before you run your python application.


```
export SCALE_API_KEY="[Your API key]"
```

## Example Code

### Sample Completion

With your API key set, you can now send LLM Engine requests using the Python client:


```py
from llmengine import Completion

response = Completion.create(
    model="falcon-7b-instruct",
    prompt="I'm opening a pancake restaurant that specializes in unique pancake shapes, colors, and flavors. List 3 quirky names I could name my restaurant.",
    max_new_tokens=100,
    temperature=0.2,
)

print(response.outputs[0].text)
```

### With Streaming


```py
from llmengine import Completion
import sys

stream = Completion.create(
    model="falcon-7b-instruct",
    prompt="Give me a 200 word summary on the current economic events in the US.",
    max_new_tokens=1000,
    temperature=0.2,
    stream=True
)

for response in stream:
    if response.output:
        print(response.output.text, end="")
        sys.stdout.flush()
```
