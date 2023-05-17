# ⚡ Spellbook Serve ⚡

The fastest, cheapest, and easiest way to scale foundation models on your own
infrastructure.

## 💻 Quick Install

```commandline
pip install spellbook-serve
conda install spellbook-serve -c conda-forge
```

## 🤔 About

Foundation models are emerging as the building blocks of AI. However, deploying
these models to the cloud still requires infrastructure expertise, and can be
expensive.

Spellbook Serve is a Python library, CLI, and Helm chart that provides
everything you need to deploy your foundation models to the cloud using
Kubernetes. Key features include:

🤗 **Open-Source Integrations**: Deploy any [Huggingface](https://huggingface.co/)
model with a single command. Integrate seamlessly with
[Langchain](https://github.com/hwchase17/langchain) chat applications.

* [Huggingface Documentation](./docs/integrations/huggingface.md)
* [Huggingface Example](./docs/examples/huggingface.md)
* [Langchain Documentation](./docs/integrations/langchain.md)
* [Langchain Example](./docs/examples/langchain.md)

❄ **Fast Cold-Start Times**: To prevent GPUs from idling, Spellbook Serve
automatically scales your model to zero when it's not in use and scales up
within seconds, even for large foundation models.

* [Documentation](./docs/cold-start.md)
* [Benchmarks](./docs/benchmarks/cold-start-times.md)

💸 **Cost-Optimized**: Deploy AI models up to 7x cheaper than OpenAI APIs,
including cold-start and warm-down times.

* [Benchmarks](./docs/benchmarks/cost.md)

🐳 **Deploying from any docker image**: Turn any Docker image into an
auto-scaling deployment with simple APIs.

* [Documentation](./docs/concepts/bundles.md)
* [Example](./docs/examples/custom-endpoints.md)

🎙️ **Language-Model Specific Features**: Spellbook Serve provides APIs for
streaming responses and dynamically batching inputs for higher throughput and
lower latency.

* [Streaming Documentation](./docs/concepts/streaming.md)
* [Streaming Example](./docs/examples/streaming.md)
* [Dynamic Batching Documentation](./docs/concepts/dynamic-batching.md)
* [Dynamic Batching Example](./docs/examples/dynamic-batching.md)

## 📖 Documentation

Please see [here](./docs) for full documentation on:

* Getting started (installation, setting up the environment)
* How-To examples (demos, integrations, helper functions)
* Reference (full API docs)
