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

Setting `report_to` in `hyperparameters` to `wandb` enables automatic metrics tracking.


`wandb_config` can contain any parameters from [wandb.init()](https://docs.wandb.ai/ref/python/init).
`api_key` which is the API key must be specified. Can also specify `base_url` to use a custom Weights & Biases server.
