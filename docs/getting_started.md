# 🚀 Getting Started

The fastest way to get started with LLMEngine is to use the python client in this repository to run inference and fine-tuning on Scale's infrastructure. This path does not require you to install anything on your infrastructure, and Scale's free tier gives you access to experimentation using open source LLMs.

To start with, install LLMEngine via pip or conda:

=== "Install using pip"
`commandline
    pip install scale-llm-engine
    `
=== "Install using conda"
`commandline
    conda install scale-llm-engine -c conda-forge
    `

Next, navigate to [https://spellbook.scale.com](https://spellbook.scale.com) where
you will get a Scale API key on the [settings](https://spellbook.scale.com/settings) page.
Set this API key as the `SCALE_API_KEY` environment variable:

=== "Set API key"
`commandline
    export SCALE_API_KEY = "[Your API key]"
    `

With your API key set, you can now send LLMEngine requests using the Python client.

Here is an example of inference using the llama-7b model:

=== "Using the Python Client"

```py
from llmengine import Completion

response = Completion.create(
  model_name="llama-7b",
  prompt="Suggest a name for a new icecream shop")
print(response.output.text)
```
