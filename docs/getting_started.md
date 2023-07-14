# 🚀 Getting Started

To start using Spellbook Serve with public inference APIs, simply run the following:

=== "Install using pip"
    ```commandline
    pip install spellbook-serve-client
    ```
=== "Install using conda"
    ```commandline
    conda install spellbook-serve-client -c conda-forge
    ```

Navigate to [https://spellbook.scale.com](https://spellbook.scale.com) where
you will get a Scale API key. Set this API key as the `SCALE_API_KEY`
environment variable.

With the API key, you can now send requests to Spellbook Serve public inference
APIs using Python client:

=== "Using the Python Client"
    ```py
    from spellbook_serve_client import Completion

    response = Completion.create(
        model_name="llama-7b",
        prompt="Hello, my name is",
        max_new_tokens=10,
        temperature=0.2,
    )
    print(response.outputs[0].text)
    ```
 
