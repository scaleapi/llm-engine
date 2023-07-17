# ⚡ LLMEngine ⚡

**The k8s based open source engine for inference and fine-tuning of Large Language Models**.

LLMEngine is the easiest way to customize and serve LLMs.

LLMs can be accessed via Scale's hosted version or by using the helm charts in this repository to run model inference and fine-tuning in your own infrastructure.

## 🤔 About

Foundation models are emerging as the building blocks of AI. However, deploying
these models to the cloud and fine-tuning them still requires infrastructure and
ML expertise. It is also difficult to maintain over time as new models are released
and new techniques for both inference and fine-tuning are made available.

LLMEngine is a Python library, CLI, and Helm chart that provides everything you need to
fine-tune and serve open-source foundation models in the cloud using k8s.

If you get a [Scale API key](./getting_started), you can use LLMEngine's python client to access inference and fine-tuning APIs for all supported [models](./model_zoo)

If you instead want to self-host, you can use LLMEngine's helm charts to install LLMEngine server-side code on a k8s cluster of your choice. Note that the current documentation does not cover this use-case, but we will be documenting this path over the next few days.

Key features include:

🚀 **Ready-to-use Fine-Tuning and Inference APIs for your favorite models**:
LLMEngine comes with ready-to-use APIs for your favorite
open-source [models](./model_zoo), including MPT, Falcon, and LLaMA. Use Scale-hosted endpoints
or deploy to your own infrastructure.

🐳 **Deploying from any docker image**: Turn any Docker image into an
auto-scaling deployment with simple APIs.

🎙️**Optimized Inference**: LLMEngine provides inference APIs
for streaming responses and dynamically batching inputs for higher throughput
and lower latency.

🤗 **Open-Source Integrations**: Deploy any [Huggingface](https://huggingface.co/)
model with a single command.

### 🔥 Features Coming Soon

❄ **Fast Cold-Start Times**: To prevent GPUs from idling, LLMEngine
automatically scales your active models to zero when they are not in use and scales up
within seconds, even for large foundation models.

💸 **Cost-Optimized**: Use cold-start and warm-down times to Deploy AI models in a cost-optimized manner.
