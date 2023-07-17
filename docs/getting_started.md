# 🚀 Getting Started

The fastest way to get started with LLM Engine is to use the python client in this repository to run inference and fine-tuning on Scale's infrastructure. This path does not require you to install anything on your infrastructure, and Scale's free tier gives you access to experimentation using open source LLMs.

To start with, install LLM Engine via pip or conda:

=== "Install using pip"
    ```commandline
    pip install scale-llm-engine
    ```
=== "Install using conda"
    ```commandline
    conda install scale-llm-engine -c conda-forge
    ```

## Scale API Keys

Next, you need a Scale Spellbook API key.

### Retrieving your API Key

To retrieve your API key, head to [Scale Spellbook](https://spellbook.scale.com) where
you will get an API key on the [settings](https://spellbook.scale.com/settings) page.

!!! note "Different API Keys for different Scale Products"

    If you have leveraged Scale's platform for annotation work in the past, please note that your Spellbook API key will be different than the Scale Annotation API key. You will want to create a Spellbook API key before getting started.

### Using your API Key

LLM Engine uses environment variables to access your API key.

Set this API key as the `SCALE_API_KEY` environment variable by running the following command in your terminal before you run your python application.

=== "Set API key"
    ```commandline
    export SCALE_API_KEY="[Your API key]"
    ```

You can also add in the line above to your `.zshrc` or `.bash_profile` so it's automatically set for future sessions.

## Example Code

### Sample Completion

With your API key set, you can now send LLM Engine requests using the Python client:

=== "Using the Python Client"
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

=== "Using the Python Client"
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
