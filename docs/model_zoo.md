# Public Model Zoo

Scale hosts the following models in the LLM Engine Model Zoo:

| Model Name                        | Inference APIs Available | Fine-tuning APIs Available | Inference Frameworks Available             | Inference max total tokens (prompt + response) |
| --------------------------------- | ------------------------ | -------------------------- | ------------------------------------------ | ---------------------------------------------- |
| `llama-7b`                        | ✅                       | ✅                         | deepspeed, text-generation-inference       | 2048                                           |
| `llama-2-7b`                      | ✅                       | ✅                         | text-generation-inference, vllm            | 4096                                           |
| `llama-2-7b-chat`                 | ✅                       |                            | text-generation-inference, vllm            | 4096                                           |
| `llama-2-13b`                     | ✅                       |                            | text-generation-inference, vllm            | 4096                                           |
| `llama-2-13b-chat`                | ✅                       |                            | text-generation-inference, vllm            | 4096                                           |
| `llama-2-70b`                     | ✅                       | ✅                         | text-generation-inference, vllm            | 4096                                           |
| `llama-2-70b-chat`                | ✅                       |                            | text-generation-inference, vllm            | 4096                                           |
| `llama-3-8b`                      | ✅                       |                            | vllm                                       | 8192                                           |
| `llama-3-8b-instruct`             | ✅                       |                            | vllm                                       | 8192                                           |
| `llama-3-70b`                     | ✅                       |                            | vllm                                       | 8192                                           |
| `llama-3-70b-instruct`            | ✅                       |                            | vllm                                       | 8192                                           |
| `llama-3-1-8b`                    | ✅                       |                            | vllm                                       | 131072                                         |
| `llama-3-1-8b-instruct`           | ✅                       |                            | vllm                                       | 131072                                         |
| `llama-3-1-70b`                   | ✅                       |                            | vllm                                       | 131072                                         |
| `llama-3-1-70b-instruct`          | ✅                       |                            | vllm                                       | 131072                                         |
| `falcon-7b`                       | ✅                       |                            | text-generation-inference, vllm            | 2048                                           |
| `falcon-7b-instruct`              | ✅                       |                            | text-generation-inference, vllm            | 2048                                           |
| `falcon-40b`                      | ✅                       |                            | text-generation-inference, vllm            | 2048                                           |
| `falcon-40b-instruct`             | ✅                       |                            | text-generation-inference, vllm            | 2048                                           |
| `mpt-7b`                          | ✅                       |                            | deepspeed, text-generation-inference, vllm | 2048                                           |
| `mpt-7b-instruct`                 | ✅                       | ✅                         | deepspeed, text-generation-inference, vllm | 2048                                           |
| `flan-t5-xxl`                     | ✅                       |                            | deepspeed, text-generation-inference       | 2048                                           |
| `mistral-7b`                      | ✅                       | ✅                         | vllm                                       | 8000                                           |
| `mistral-7b-instruct`             | ✅                       | ✅                         | vllm                                       | 8000                                           |
| `mixtral-8x7b`                    | ✅                       |                            | vllm                                       | 32768                                          |
| `mixtral-8x7b-instruct`           | ✅                       |                            | vllm                                       | 32768                                          |
| `mixtral-8x22b`                   | ✅                       |                            | vllm                                       | 65536                                          |
| `mixtral-8x22b-instruct`          | ✅                       |                            | vllm                                       | 65536                                          |
| `codellama-7b`                    | ✅                       | ✅                         | text-generation-inference, vllm            | 16384                                          |
| `codellama-7b-instruct`           | ✅                       | ✅                         | text-generation-inference, vllm            | 16384                                          |
| `codellama-13b`                   | ✅                       | ✅                         | text-generation-inference, vllm            | 16384                                          |
| `codellama-13b-instruct`          | ✅                       | ✅                         | text-generation-inference, vllm            | 16384                                          |
| `codellama-34b`                   | ✅                       | ✅                         | text-generation-inference, vllm            | 16384                                          |
| `codellama-34b-instruct`          | ✅                       | ✅                         | text-generation-inference, vllm            | 16384                                          |
| `codellama-70b`                   | ✅                       |                            | vllm                                       | 16384                                          |
| `codellama-70b-instruct`          | ✅                       |                            | vllm                                       | 4096                                           |
| `zephyr-7b-alpha`                 | ✅                       |                            | text-generation-inference, vllm            | 32768                                          |
| `zephyr-7b-beta`                  | ✅                       |                            | text-generation-inference, vllm            | 32768                                          |
| `gemma-2b`                        | ✅                       |                            | vllm                                       | 8192                                           |
| `gemma-2b-instruct`               | ✅                       |                            | vllm                                       | 8192                                           |
| `gemma-7b`                        | ✅                       |                            | vllm                                       | 8192                                           |
| `gemma-7b-instruct`               | ✅                       |                            | vllm                                       | 8192                                           |
| `phi-3-mini-4k-instruct`          | ✅                       |                            | vllm                                       | 4096                                           |
| `deepseek-coder-v2`               | ✅                       |                            | vllm                                       | 131072                                         |
| `deepseek-coder-v2-instruct`      | ✅                       |                            | vllm                                       | 131072                                         |
| `deepseek-coder-v2-lite`          | ✅                       |                            | vllm                                       | 131072                                         |
| `deepseek-coder-v2-lite-instruct` | ✅                       |                            | vllm                                       | 131072                                         |
| `qwen2-72b-instruct`              | ✅                       |                            | vllm                                       | 32768                                          |
 

## Usage

Each of these models can be used with the
[Completion](../api/python_client/#llmengine.Completion) API.

The specified models can be fine-tuned with the
[FineTune](../api/python_client/#llmengine.FineTune) API.

More information about the models can be found using the
[Model](../api/python_client/#llmengine.Model) API.
