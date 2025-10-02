# Model Engine

The Model Engine is an API server that allows users to create, deploy, edit,
and delete machine learning endpoints. It consists of two main architectural
components:

- The [gateway](./model_engine_server/entrypoints/start_fastapi_server.py)
  provides a REST API for users to interact with. The routes of the REST API are
  defined in [`model_engine_server.api`](./model_engine_server/api).
- The [`model_engine_server.service_builder`](./model_engine_server/service_builder)
  package is the part of the code that creates the inference pods. It is the
  endpoint builder. When we do a `POST` request to `/endpoints`, this gets run.
  It gets run when users create or edit endpoints with `[POST,PUT] /v1/model-endpoints`

There are two other microservices:

- The [kubernetes cache](./model_engine_server/entrypoints/k8s_cache.py)
  stores endpoint metadata on Redis so that Model Engine does not overload the API
  server.
- The celery autoscaler (link TBD) automatically scales
  the number of inference pods based on the number of requests for async endpoints.

## Getting started

Be sure to install the global `../requirements-dev.txt` first prior
to any installations of requirements in this directory
(`pip install -r ../requirements-dev.txt`), as well as the pre-commit hooks
(`pre-commit install` in the `llm-engine` root folder). Then, install the
requirements files and this folder as editable

```bash
pip install -r requirements.txt && \
    pip install -r requirements-test.txt && \
    pip install -r requirements_override.txt && \
    pip install -e .
```

Run `mypy . --install-types` to set up mypy.

## Testing

Most of the business logic in Model Engine should contain unit tests, located in
[`tests/unit`](./tests/unit). To run the tests, run `pytest`.

### Testing the http_forwarder

First have some endpoint running on port 5005
```sh
(llm-engine-vllm) ➜  vllm git:(dmchoi/vllm_batch_upgrade) ✗ export IMAGE=692474966980.dkr.ecr.us-west-2.amazonaws.com/vllm:0.10.1.1-rc2
(llm-engine-vllm) ➜  vllm git:(dmchoi/vllm_batch_upgrade) ✗ export MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct && export MODEL_PATH=/data/model_files/$MODEL
(llm-engine-vllm) ➜  vllm git:(dmchoi/vllm_batch_upgrade) ✗ export REPO_PATH=/mnt/home/dmchoi/repos/scale
(llm-engine-vllm) ➜  vllm git:(dmchoi/vllm_batch_upgrade) ✗ docker kill vlll; docker rm vllm; docker run \
    --runtime nvidia \
    --shm-size=16gb \
    --gpus '"device=0,1,2,3"' \
    -v $MODEL_PATH:/workspace/model_files:ro \
    -v ${REPO_PATH}/llm-engine/model-engine/model_engine_server/inference/vllm/vllm_server.py:/workspace/vllm_server.py \
    -p 5005:5005 \
    --name vllm \
    ${IMAGE} \
    python -m vllm_server --model model_files  --port 5005 --disable-log-requests --max-model-len 4096 --max-num-seqs 16 --enforce-eager
```

Then you can run the forwarder locally like this
```sh
GIT_TAG=test python model_engine_server/inference/forwarding/http_forwarder.py \
    --config model_engine_server/inference/configs/service--http_forwarder.yaml \
    --num-workers 1 \
    --set "forwarder.sync.extra_routes=['/v1/chat/completions','/v1/completions']" \
    --set "forwarder.stream.extra_routes=['/v1/chat/completions','/v1/completions']" \
    --set "forwarder.sync.healthcheck_route=/health" \
    --set "forwarder.stream.healthcheck_route=/health"
```

Then you can hit the forwarder like this
```sh
 curl -X POST localhost:5000/v1/chat/completions  -H "Content-Type: application/json" -d "{\"args\": {\"model\":\"$MODEL\", \"messages\":[{\"role\": \"systemr\", \"content\": \"Hey, what's the temperature in Paris right now?\"}],\"max_tokens\":100,\"temperature\":0.2,\"guided_regex\":\"Sean.*\"}}"
```

## Generating OpenAI types
We've decided to make our V2 APIs OpenAI compatible. We generate the
corresponding Pydantic models:
1. Fetch the OpenAPI spec from https://github.com/openai/openai-openapi/blob/master/openapi.yaml
2. Run scripts/generate-openai-types.sh
