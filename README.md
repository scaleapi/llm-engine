# ⚡ LLM Engine ⚡

**The open source engine for fine-tuning large language models**. LLM Engine is the easiest way to customize and serve LLMs.
Use Scale's hosted version or run it in your own cloud.

## 💻 Quick Install

```commandline
pip install scale-llm-engine
```

## 🤔 About

Foundation models are emerging as the building blocks of AI. However, deploying
these models to the cloud and fine-tuning them still requires infrastructure and
ML expertise, and can be expensive.

LLM Engine is a Python library, CLI, and Helm chart that provides
everything you need to fine-tune and serve foundation models in the cloud
using Kubernetes. Key features include:

🎁 **Ready-to-use Fine-Tuning and Inference APIs for your favorite models**:
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

## 🚀 Getting Started

Navigate to [https://spellbook.scale.com](https://spellbook.scale.com) where
you will get a Scale API key on the [settings](https://spellbook.scale.com/settings) page.
Set this API key as the `SCALE_API_KEY` environment variable by adding the
following line to your `.zshrc` or `.bash_profile`:

```commandline
export SCALE_API_KEY = "[Your API key]"
```

With your API key set, you can now send LLM Engine requests using the Python client:

```py
from llmengine import Completion

response = Completion.create(
    model="llama-7b",
    prompt="Hello, my name is",
    max_new_tokens=10,
    temperature=0.2,
)
print(response.outputs[0].text)
```

See the [documentation site](https://scaleapi.github.io/llm-engine/) for more details.