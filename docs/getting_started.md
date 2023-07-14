# 🚀 Getting Started

To start using Spellbook Serve with public inference APIs, simply run the following:

=== "Install using pip"
    ```commandline
    pip install spellbook-serve
    ```
=== "Install using conda"
    ```commandline
    conda install spellbook-serve -c conda-forge
    ```

Navigate to [https://spellbook.scale.com](https://spellbook.scale.com) where
you will get a Scale API key.

With the API key, you can now send requests to Spellbook Serve public inference
APIs using the CLI or Python client:

=== "Using the CLI"
    ```commandline
    spellbook-serve generate flan-t5-xxl \
        --prompt "Hello, my name is"
        --temperature 0.5
        --max-tokens 20

    # Expected output:
    #
    # Hello, my name is Flan.
    ```
=== "Using the Python Client"
    ```py
    from spellbook_serve_client import Completion

    response = Completion.create(
        model_name="llama-7b-text-generation-inference",
        prompt="Hello, my name is",
        max_new_tokens=10,
        temperature=0.2,
    )
    print(response.outputs[0].text)
    ```

## 💻 Installation on Kubernetes

To install Spellbook Serve on your infrastructure in Kubernetes, you can use the
Helm chart:

```commandline
helm repo add spellbook https://spellbook.github.io/helm-charts
helm repo update
helm install spellbook-serve spellbook/spellbook-serve
```
