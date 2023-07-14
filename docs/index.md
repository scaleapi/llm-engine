# ⚡ Spellbook Serve ⚡

The fastest, cheapest, and easiest way to deploy and scale your custom foundation models.

## 💻 Quick Install

=== "Install using pip"
    ```commandline
    pip install spellbook-serve-client
    ```
 
## 🤔 About

Foundation models are emerging as the building blocks of AI. However, deploying
these models to the cloud still requires infrastructure expertise, and can be
expensive.

Spellbook Serve is a Python library, CLI, and Helm chart that provides
everything you need to deploy your foundation models to the cloud using
Kubernetes. Key features include:

🚀 **Ready-to-use Inference APIs for your favorite models**: Spellbook Serve
comes with ready-to-use [inference APIs](/model_zoo/) for your favorite
open-source models, including MPT, Falcon, and LLaMA. Use scale-hosted endpoints
or deploy to your own infrastructure.

🐳 **Deploying from any docker image**: Turn any Docker image into an
auto-scaling deployment with simple APIs.

🎙️**Language-Model Specific Features**: Spellbook Serve provides APIs for
streaming responses and dynamically batching inputs for higher throughput and
lower latency.

🤗 **Open-Source Integrations**: Deploy any [Huggingface](https://huggingface.co/)
model with a single command. Integrate seamlessly with
[Langchain](https://github.com/hwchase17/langchain) chat applications.

### 🔥 Features Coming Soon

❄ **Fast Cold-Start Times**: To prevent GPUs from idling, Spellbook Serve
automatically scales your model to zero when it's not in use and scales up
within seconds, even for large foundation models.

💸 **Cost-Optimized**: Deploy AI models an order of magnitude cheaper than
OpenAI APIs, including cold-start and warm-down times.

