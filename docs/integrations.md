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

- (Required) Set `hyperparameters.report_to` to `wandb` to enables automatic metrics tracking.
- (Required) Set `wandb_config.api_key` to the API key.
- (Optional) Set `wandb_config.base_url` to use a custom Weights & Biases server.
- `wandb_config` also accepts keys from [wandb.init()](https://docs.wandb.ai/ref/python/init).
