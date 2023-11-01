# Public Model Zoo

Scale hosts the following models in the LLM Engine Model Zoo:

| Model Name            | Inference APIs Available | Fine-tuning APIs Available | Inference Frameworks Available |
| --------------------- | ------------------------ | -------------------------- | ------------------------------ |
| `llama-7b`            | ✅                       | ✅                         | deepspeed, text-generation-inference |
| `llama-2-7b`          | ✅                       | ✅                         | text-generation-inference, vllm |
| `llama-2-7b-chat`     | ✅                       |                            | text-generation-inference, vllm |
| `llama-2-13b`         | ✅                       |                            | text-generation-inference, vllm |
| `llama-2-13b-chat`    | ✅                       |                            | text-generation-inference, vllm |
| `llama-2-70b`         | ✅                       | ✅                         | text-generation-inference, vllm |
| `llama-2-70b-chat`    | ✅                       |                            | text-generation-inference, vllm |
| `falcon-7b`           | ✅                       |                            | text-generation-inference, vllm |
| `falcon-7b-instruct`  | ✅                       |                            | text-generation-inference, vllm | 
| `falcon-40b`          | ✅                       |                            | text-generation-inference, vllm |
| `falcon-40b-instruct` | ✅                       |                            | text-generation-inference, vllm |
| `mpt-7b`              | ✅                       |                            | deepspeed, text-generation-inference, vllm |
| `mpt-7b-instruct`     | ✅                       | ✅                         | deepspeed, text-generation-inference, vllm |
| `flan-t5-xxl`         | ✅                       |                            | deepspeed, text-generation-inference |
| `mistral-7b`         | ✅                       |   ✅                         | vllm | 
| `mistral-7b-instruct`         | ✅                       |    ✅                        | vllm |
| `codellama-7b`         | ✅                       | ✅                           | text-generation-inference, vllm |
| `codellama-7b-instruct`         | ✅                       | ✅                           | text-generation-inference, vllm |
| `codellama-13b`         | ✅                       |                            | text-generation-inference, vllm |
| `codellama-34b`         | ✅                       |                            | text-generation-inference, vllm |

## Usage

Each of these models can be used with the
[Completion](../api/python_client/#llmengine.Completion) API.

The specified models can be fine-tuned with the
[FineTune](../api/python_client/#llmengine.FineTune) API.

More information about the models can be found using the
[Model](../api/python_client/#llmengine.Model) API.
