# quick commands

```
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