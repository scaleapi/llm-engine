# ⚡ LLM Engine ⚡

**The open source engine for fine-tuning large language models**. LLM Engine is the easiest way to customize and serve LLMs.
Use Scale's hosted version or run it in your own cloud.

## 💻 Quick Install

```commandline
pip install scale-llm-engine
```

## 🤔 About

Foundation models are emerging as the building blocks of AI. However, 
fine-tuning these models and deploying them to the cloud are expensive 
operations that require infrastructure and ML expertise.

LLM Engine is a Python library, CLI, and Helm chart that provides
everything you need to fine-tune and serve foundation models in the cloud
using Kubernetes. Key features include:

🎁 **Ready-to-use APIs for your favorite models**:
Fine-tune and serve open-source foundation models, including MPT, Falcon,
and LLaMA. Use Scale-hosted endpoints or deploy to your own infrastructure.

🐳 **Deploying from any docker image**: Turn any Docker image into an
auto-scaling deployment with simple APIs.

🎙️**Optimized Inference**: LLM Engine provides inference APIs
for streaming responses and dynamically batching inputs for higher throughput
and lower latency.

🤗 **Open-Source Integrations**: Deploy any [Hugging Face](https://huggingface.co/)
model with a single command.

### 🔥 Features Coming Soon

❄ **Fast Cold-Start Times**: To prevent GPUs from idling, LLM Engine
automatically scales your model to zero when it's not in use and scales up
within seconds, even for large foundation models.

💸 **Cost Optimization**: Deploy AI models cheaper than commercial ones,
including cold-start and warm-down times.

## 🚀 Getting Started

Navigate to [Scale Spellbook](https://spellbook.scale.com/) to first create 
an account, and then grab your API key on the [Settings](https://spellbook.scale.com/settings) 
page. Set this API key as the `SCALE_API_KEY` environment variable by adding the
following line to your `.zshrc` or `.bash_profile`:

```commandline
export SCALE_API_KEY="[Your API key]"
```

You may need to run the `. ~/.zshrc` command to re-read your updated `.zshrc`.


With your API key set, you can now send LLM Engine requests using the Python client. 
Try out this starter code:

```py
from llmengine import Completion

response = Completion.create(
    model_name="falcon-7b-instruct",
    prompt="I'm opening a pancake restaurant that specializes in unique pancake shapes, colors, and flavors. List 3 quirky names I could name my restaurant.",
    max_new_tokens=100,
    temperature=0.2,
)

print(response.outputs[0].text)
```

You should see a successful completion of your given prompt!

Next, visit our [documentation site](https://scaleapi.github.io/llm-engine/) for more on
the `Completion` and `FineTune` APIs and how to use them.