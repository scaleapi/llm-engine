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

You can generate an API key using the CLI command

```commandline
spellbook-serve generate-api-key
```

With the API key, you can now send requests to Spellbook Serve public inference
APIs using the CLI or Python client:

=== "Using the CLI"
    ```commandline
    spellbook-serve endpoints list

    # Expected output:
    #
    # public-flan-t5-xxl
    # public-stablelm-2
    # public-dolly-2

    spellbook-serve endpoints send public-flan-t5-xxl \
        --input "Hello, my name is"
        --temperature 0.5
        --top_p 0.9

    # Expected output:
    #
    # Hello, my name is Flan.
    ```
=== "Using the Python Client"
    ```py
    import spellbook_serve as ss

    client = ss.Client()
    request = ss.EndpointRequest(
        input="Hello, my name is",
        temperature=0.5,
        top_p=0.9,
    )
    endpoint = client.get_model_endpoint("public-flan-t5-xxl")
    future = endpoint.send(request)
    response = future.get()
    print(response)
    ```

## 💻 Installation on Kubernetes

To install Spellbook Serve on Kubernetes, you can use the Helm chart:

```commandline
helm repo add spellbook https://spellbook.github.io/helm-charts
helm repo update
helm install spellbook-serve spellbook/spellbook-serve
```
