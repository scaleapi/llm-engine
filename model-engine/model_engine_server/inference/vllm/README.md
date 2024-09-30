# VLLM

## Building container

There are three build targets for vLLM. 
1. vLLM endpoint
2. vLLM batch job v1
3. vLLM batch job v2

```bash
VLLM_VERSION=0.5.4 bash build_and_upload_image.sh $ACCOUNT_ID $IMAGE_TAG {BUILD_TARGET=vllm|vllm_batch|vllm_batch_v2}
```

## Running locally

### Endpoint

1. Download model weights to `model_files`
2. Run docker locally
```bash
IMAGE=${ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com/vllm:${IMAGE_TAG}
docker kill vllm; docker rm vllm;
docker run \
    --runtime nvidia \
    --shm-size=16gb \
    --gpus '"device=0"' \
    -v $MODEL_PATH:/workspace/model_files:ro \
    -p 5005:5005 \
    --name vllm \
    ${IMAGE} \
    python -m vllm_server --model model_files --tensor-parallel-size 1 --port 5005 --disable-log-requests
```

3. Send curl requests
```bash
curl -X POST localhost:5005/v1/chat/completions \
 -H "Content-Type: application/json" \
 -d '{"messages":[{"role": "user", "content": "Hey, whats the temperature in Paris right now?"}],"model":"model_files","max_tokens":100,"temperature":0.2,"guided_regex":"Sean.*"}'
```

### Batch job v2
```bash
IMAGE_BATCH=${ACCOUNT_ID}.dkr.ecr.us-west-2.amazonaws.com/llm-engine/batch-infer-vllm:${IMAGE_TAG}

export MODEL=gemma-2-2b-it && export MODEL_PATH=/data/model_files/$MODEL
docker kill vllm_batch; docker rm vllm_batch;
docker run \
    --runtime nvidia \
    --shm-size=16gb \
    --gpus '"device=6,7"' \
    -v $MODEL_PATH:/workspace/model_files:ro \
    -v ${REPO_PATH}/llm-engine/model-engine/model_engine_server/inference/vllm/examples:/workspace/examples \
    -v ${REPO_PATH}/llm-engine/model-engine/model_engine_server/inference/vllm/vllm_batch.py:/workspace/vllm_batch.py \
    -p 5005:5005 \
    -e CONFIG_FILE=/workspace/examples/v2/gemma/config.json \
    -e MODEL_WEIGHTS_FOLDER=/workspace/model_files \
    --name vllm_batch \
    ${IMAGE_BATCH} \
    python vllm_batch.py   

```