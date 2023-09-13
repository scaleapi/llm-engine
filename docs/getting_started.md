# Getting Started

The fastest way to get started with LLM Engine is to use the Python client in this repository to 
run inference and fine-tuning on Scale's infrastructure. This path does not require you to install 
anything on your infrastructure, and Scale's free research preview gives you access to experimentation using open source LLMs.

To start, install LLM Engine via pip:

=== "pip"
    ```commandline
    pip install scale-llm-engine
    ```

## Scale API Keys

Next, you need a Scale Spellbook API key.

### Retrieving your API Key

To retrieve your API key, head to [Scale Spellbook](https://spellbook.scale.com) where
you will get an API key on the [settings](https://spellbook.scale.com/settings) page.

!!! note "Different API Keys for different Scale Products"

    If you have leveraged Scale's platform for annotation work in the past, please note that your Spellbook API key will be different than the Scale Annotation API key. You will want to create a Spellbook API key before getting started.

### Set your API Key

LLM Engine uses environment variables to access your API key.

Set this API key as the `SCALE_API_KEY` environment variable by running the following command in your terminal before you run your python application.


```
export SCALE_API_KEY="[Your API key]"
```

You can also add in the line above to your `.zshrc` or `.bash_profile` so it's automatically set for future sessions.

Alternatively, you can also set your API key using either of the following patterns:
```
llmengine.api_engine.api_key = "abc"
llmengine.api_engine.set_api_key("abc")
```
These patterns are useful for Jupyter Notebook users to set API keys without the need for using `os.environ`.

## Example Code

### Sample Completion

With your API key set, you can now send LLM Engine requests using the Python client:


```py
from llmengine import Completion

response = Completion.create(
    model="llama-2-7b",
    prompt="I'm opening a pancake restaurant that specializes in unique pancake shapes, colors, and flavors. List 3 quirky names I could name my restaurant.",
    max_new_tokens=100,
    temperature=0.2,
)

print(response.output.text)
```

### With Streaming


```py
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
    try:
        if response.output:
            print(response.output.text, end="")
            sys.stdout.flush()
    except: # an error occurred
        print(stream.text) # print the error message out 
        break
```
