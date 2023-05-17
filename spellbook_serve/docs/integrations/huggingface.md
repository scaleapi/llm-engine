# 🤗 Using Spellbook Serve with Huggingface Transformers

To deploy huggingface models with Spellbook Serve, you need to install the
`huggingface` extra:

```commandline
pip install "spellbook-serve[transformers]"
```

To create a Model Bundle from a model on Huggingface transformers, you can use

```py
import spellbook_serve as ss

client = ss.Client()
client.create_model_endpoint_from_huggingface(
    model_name="bert-base-uncased",
    model_class="BertForSequenceClassification",
    tokenizer_class="BertTokenizer",
    tokenizer_name="bert-base-uncased",
    task="text-classification",
    num_labels=2,
    max_length=128,
    batch_size=32,
    framework="pt",
    model_endpoint_type="async",
)
```

The input arguments to `create_model_endpoint_from_huggingface` are the same as
the input arguments to the associated Huggingface pipeline.
