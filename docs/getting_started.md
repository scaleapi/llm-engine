# Getting Started

**Note: As of October 31st 2024, LLM Engine's public demo service is sunsetted. We have thus removed the documentation 
pieces relating to calling the demo service, procuring a Spellbook API key, etc. Please view our Self Hosting Guide instead. 
We will however leave behind the Example Code snippets for posterity, and as a reference for self-hosted and Scale internal users.**

To start, install LLM Engine via pip:

=== "pip"
    ```commandline
    pip install scale-llm-engine
    ```

## Scale user ID

Next, you need a Scale user ID. Recall that this is only applicable to Scale internal users for now, and we are just leaving 
this note to serve as internal documentation.


### Set your API Key

LLM Engine uses environment variables to access your API key.

Set the `SCALE_API_KEY` environment variable to your Scale user ID by running the following command in your terminal before you run your python application.


```
export SCALE_API_KEY="[Your Scale user ID]"
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
    if response.output:
        print(response.output.text, end="")
        sys.stdout.flush()
    else: # an error occurred
        print(response.error) # print the error message out 
        break
```
