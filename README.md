# ⚡ LLM Engine ⚡

**The open source engine for fine-tuning large language models**. 

Scale's LLM Engine is the easiest way to customize and serve LLMs. In LLM Engine, models can be accessed via Scale's hosted version or by using the Helm charts in this repository to run model inference and fine-tuning in your own infrastructure.

## 💻 Quick Install

```commandline
pip install scale-llm-engine
```

## 🤔 About

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

🎁 **Ready-to-use APIs for your favorite models**: Deploy and serve
open-source foundation models — including LLaMA, MPT and Falcon.
Use Scale-hosted models or deploy to your own infrastructure.

🔧 **Fine-tune foundation models**: Fine-tune open-source foundation
models on your own data for optimized performance.

🎙️ **Optimized Inference**: LLM Engine provides inference APIs
for streaming responses and dynamically batching inputs for higher throughput
and lower latency.

🤗 **Open-Source Integrations**: Deploy any [Hugging Face](https://huggingface.co/)
model with a single command.

### Features Coming Soon

🐳 **K8s Installation Documentation**: We are working hard to document installation and
maintenance of inference and fine-tuning functionality on your own infrastructure.
For now, our documentation covers using our client libraries to access Scale's
hosted infrastructure.

❄ **Fast Cold-Start Times**: To prevent GPUs from idling, LLM Engine
automatically scales your model to zero when it's not in use and scales up
within seconds, even for large foundation models.

💸 **Cost Optimization**: Deploy AI models cheaper than commercial ones,
including cold-start and warm-down times.

## 🚀 Quick Start

Navigate to [Scale Spellbook](https://spellbook.scale.com/) to first create 
an account, and then grab your API key on the [Settings](https://spellbook.scale.com/settings) 
page. Set this API key as the `SCALE_API_KEY` environment variable by adding the
following line to your `.zshrc` or `.bash_profile`:

```commandline
export SCALE_API_KEY="[Your API key]"
```

If you run into an "Invalid API Key" error, you may need to run the `. ~/.zshrc` command to 
re-read your updated `.zshrc`.


With your API key set, you can now send LLM Engine requests using the Python client. 
Try out this starter code:

```py
from llmengine import Completion

response = Completion.create(
    model="falcon-7b-instruct",
    prompt="I'm opening a pancake restaurant that specializes in unique pancake shapes, colors, and flavors. List 3 quirky names I could name my restaurant.",
    max_new_tokens=100,
    temperature=0.2,
)

print(response.output.text)
```

You should see a successful completion of your given prompt!

_What's next?_ Visit the [LLM Engine documentation pages](https://scaleapi.github.io/llm-engine/) for more on
the `Completion` and `FineTune` APIs and how to use them.
