<img src="https://static.remotasks.com/uploads/602b25a6e0984c00343d3b26/scale-1.png"/>

# LLM Engine

**The open source engine for inference and fine-tuning of Large Language Models**.

LLM Engine is the easiest way to customize and serve LLMs.

LLMs can be accessed via Scale's hosted version or by using the helm charts in this repository to run model inference and fine-tuning in your own infrastructure.

## Quick Install

=== "Install the python package"
```commandline
pip install scale-llm-engine
```

## About

Foundation models are emerging as the building blocks of AI. However,
deploying these models to the cloud and fine-tuning them are expensive
operations that require infrastructure and ML expertise. It is also difficult
to maintain over time as new models are released and new techniques for both
inference and fine-tuning are made available.

LLM Engine is a Python library, CLI, and Helm chart that provides
everything you need to serve and fine-tune foundation models, whether you use
Scale's hosted infrastructure or do it in your own cloud infrastructure using
Kubernetes.

### Key Features

**Ready-to-use APIs for your favorite models**: Deploy and serve
open-source foundation models - including LLaMA, MPT and Falcon.
Use Scale-hosted models or deploy to your own infrastructure.

**Fine-tune your favorite models**: Fine-tune open-source foundation
models like LLaMA, MPT etc. with your own data for optimized performance.

**Optimized Inference**: LLM Engine provides inference APIs
for streaming responses and dynamically batching inputs for higher throughput
and lower latency.

**Open-Source Integrations**: Deploy any [Hugging Face](https://huggingface.co/)
model with a single command.

**Deploying from any docker image**: Turn any Docker image into an
auto-scaling deployment with simple APIs.

### Features Coming Soon

**k8s Installation Documentation**: We are working hard to document installation and
maintenance of inference and fine-tuning functionality on your own infrastructure.
For now, our documentation covers using our client libraries to access Scale's
hosted infrastructure.

**Fast Cold-Start Times**: To prevent GPUs from idling, LLM Engine
automatically scales your model to zero when it's not in use and scales up
within seconds, even for large foundation models.

**Cost Optimization**: Deploy AI models cheaper than commercial ones,
including cold-start and warm-down times.
