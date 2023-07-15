# ⚡ LLM Engine ⚡

The easiest way to fine-tune and serve custom foundation models.

## 💻 Quick Install

=== "Install using pip"
    ```commandline
    pip install llm-engine
    ```

## 🤔 About

Foundation models are emerging as the building blocks of AI. However, deploying
these models to the cloud and fine-tuning them still requires infrastructure and
ML expertise, and can be expensive.

LLM Engine is a Python library, CLI, and Helm chart that provides
everything you need to fine-tune and serve foundation models in the cloud
using Kubernetes. Key features include:

🚀 **Ready-to-use Fine-Tuning and Inference APIs for your favorite models**:
LLM Engine comes with ready-to-use APIs for your favorite
open-source models, including MPT, Falcon, and LLaMA. Use Scale-hosted endpoints
or deploy to your own infrastructure.

🐳 **Deploying from any docker image**: Turn any Docker image into an
auto-scaling deployment with simple APIs.

🎙️**Optimized Inference**: LLM Engine provides inference APIs
for streaming responses and dynamically batching inputs for higher throughput
and lower latency.

🤗 **Open-Source Integrations**: Deploy any [Huggingface](https://huggingface.co/)
model with a single command.

### 🔥 Features Coming Soon

❄ **Fast Cold-Start Times**: To prevent GPUs from idling, LLM Engine
automatically scales your model to zero when it's not in use and scales up
within seconds, even for large foundation models.

💸 **Cost-Optimized**: Deploy AI models cheaper than commercial ones,
including cold-start and warm-down times.
