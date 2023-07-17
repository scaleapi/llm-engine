# Fine Tuning API
The Fine Tuning API allows you to fine tune various open-source LLMs on your own data, then make inference calls to the resulting LLM.

## Preparing Data
Your data must be formatted as a CSV file that includes two columns: `prompt` and `response`. The data needs to be uploaded to somewhere publicly accessible, so that we can read the data to fine tune on it.

## Launching the Fine Tune
Once you have uploaded your data, you can use our API to launch a Fine Tune. You will need to provide the base model to train off of, the locations of the training and validation files, an optional set of hyperparameters to override, and an optional suffix to append to the name of the fine tune. 

Currently, we support the following base models:

`mpt-7b-instruct`

`llama-7b`

Once the fine tune is launched, you can also get the status of your fine tune.

TODO

## Making inference calls to your fine tune

TODO