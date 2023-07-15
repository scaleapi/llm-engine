# 🚀 Getting Started

To start using LLM Engine's public inference and fine-tuning APIs, simply run the following:

=== "Install using pip"
    ```commandline
    pip install llm-engine
    ```
=== "Install using conda"
    ```commandline
    conda install llm-engine -c conda-forge
    ```

Navigate to [https://spellbook.scale.com](https://spellbook.scale.com) where
you will get a Scale API key on the [settings](https://spellbook.scale.com/settings) page.
Set this API key as the `SCALE_API_KEY` environment variable by adding the
following line to your `.zshrc` or `.bash_profile`:

=== "Set API key"
    ```commandline
    export SCALE_API_KEY = "[Your API key]"

With your API key set, you can now send requests to the public LLM Engine
APIs using the Python client:

=== "Using the Python Client"
    ```py
    from llmengine import Completion

    response = Completion.create(
        model_name="llama-7b",
        prompt="Hello, my name is",
        max_new_tokens=10,
        temperature=0.2,
    )
    print(response.outputs[0].text)
    ```
 
