# Integrations

## Weights & Biases

LLM Engine integrates with Weights & Biases to track metrics during fine tuning. To enable:

```python
from llmengine import FineTune

response = FineTune.create(
    model="llama-2-7b",
    training_file="s3://my-bucket/path/to/training-file.csv",
    validation_file="s3://my-bucket/path/to/validation-file.csv",
    hyperparameters={"report_to": "wandb"},
    wandb_config={"api_key":"key", "project":"fine-tune project"}
)
```

Configs to specify:

| Field             | Subfield                                                             | Note                                                      |
| ----------------- | -------------------------------------------------------------------- | --------------------------------------------------------- |
| `hyperparameters` | `report_to`                                                          | Set to `wandb` to enables automatic metrics tracking      |
| `wandb_config`    | `api_key`                                                            | The API key, must specify                                 |
| `wandb_config`    | `base_url`                                                           | (Optional) base URL of a custom Weights & Biases server   |
| `wandb_config`    | keys from from [wandb.init()](https://docs.wandb.ai/ref/python/init) | See [wandb.init()](https://docs.wandb.ai/ref/python/init) |
