# Preparing the model weights/tokenizers

Our TensorRT-LLM docker image expects weights to live in s3/other blob store with the following directory structure:

root/
  model_tokenizer/
    <everything in a HF directory other than the weights themselves>
  model_weights/
    config.json
    rank<i>.engine

You can obtain `model_weights` by building a TRT-LLM engine via the directions found on Nvidia's site (e.g. https://github.com/NVIDIA/TensorRT-LLM/blob/main/README.md#installation, https://github.com/NVIDIA/TensorRT-LLM/blob/v0.8.0/examples/llama/convert_checkpoint.py)

The inference image is built via the Dockerfile in the same directory as this readme.