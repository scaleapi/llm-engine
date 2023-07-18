<img src="https://static.remotasks.com/uploads/602b25a6e0984c00343d3b26/scale-1.png"/>

# LLM Engine

**The open source engine for fine-tuning large language models**. LLM Engine is the easiest way to customize and serve LLMs.
Use Scale's hosted version or run it in your own cloud.

## Quick Install

```
    pip install scale-llm-engine
```

## About

Foundation models are emerging as the building blocks of AI. However,
fine-tuning these models and deploying them to the cloud are expensive
operations that require infrastructure and ML expertise.

LLM Engine is a Python library, CLI, and Helm chart that provides
everything you need to fine-tune and serve foundation models in the cloud
using Kubernetes. Key features include:

**Ready-to-use APIs for your favorite models**:
Fine-tune and serve open-source foundation models, including MPT, Falcon,
and LLaMA. Use Scale-hosted endpoints or deploy to your own infrastructure.

**Deploying from any docker image**: Turn any Docker image into an
auto-scaling deployment with simple APIs.

**Optimized Inference**: LLM Engine provides inference APIs
for streaming responses and dynamically batching inputs for higher throughput
and lower latency.

**Open-Source Integrations**: Deploy any [Hugging Face](https://huggingface.co/)
model with a single command.

### Features Coming Soon

**Fast Cold-Start Times**: To prevent GPUs from idling, LLM Engine
automatically scales your model to zero when it's not in use and scales up
within seconds, even for large foundation models.

**Cost Optimization**: Deploy AI models cheaper than commercial ones,
including cold-start and warm-down times.
