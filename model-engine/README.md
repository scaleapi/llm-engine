# Model Engine

The Model Engine is an API server that allows users to create, deploy, edit,
and delete machine learning endpoints.

## Architecture

### Core Components

- **[Gateway](./model_engine_server/entrypoints/start_fastapi_server.py)** - REST API server. Routes are defined in [`model_engine_server.api`](./model_engine_server/api).
- **[Service Builder](./model_engine_server/service_builder)** - Creates inference pods when endpoints are created/edited via `[POST,PUT] /v1/model-endpoints`.

### Supporting Services

- **[Kubernetes Cache](./model_engine_server/entrypoints/k8s_cache.py)** - Stores endpoint metadata in Redis to reduce API server load.
- **Celery Autoscaler** - Automatically scales inference pods based on request volume for async endpoints.

## Getting Started

### Prerequisites

Install global dev requirements and pre-commit hooks from the `llm-engine` root:

```bash
pip install -r ../requirements-dev.txt
pre-commit install
```

### Installation

```bash
pip install -r requirements.txt
pip install -r requirements-test.txt
pip install -r requirements_override.txt
pip install -e .
```

Set up mypy:

```bash
mypy . --install-types
```

### Running Tests

```bash
pytest
```

Unit tests are in [`tests/unit`](./tests/unit).

## OpenAPI Schema Generation

Model Engine is the **source of truth** for the Launch API schema. We generate OpenAPI schemas that are consumed by client libraries (e.g., [launch-python-client](https://github.com/scaleapi/launch-python-client)).

### Why Two Schema Versions?

FastAPI with Pydantic v2 generates **OpenAPI 3.1** schemas. However, code generators like OpenAPI Generator 6.x have incomplete 3.1 support. We provide two versions:

| File | Version | Use Case |
|------|---------|----------|
| `openapi.json` | OpenAPI 3.1 | Native FastAPI output, documentation |
| `openapi-3.0.json` | OpenAPI 3.0 | Code generation (OpenAPI Generator 6.x) |

### Generating Schemas

```bash
python scripts/generate_openapi_schemas.py [output_dir]
```

This generates:
- `openapi.json` - Native 3.1 schema
- `openapi-3.0.json` - Processed 3.0-compatible schema
- `metadata.json` - Generation timestamp and git tag

### What the 3.0 Processing Does

The `get_openapi_schema(openapi_30_compatible=True)` function converts:

1. **Nullable types**: `anyOf: [{type: string}, {type: null}]` → `{type: string, nullable: true}`
2. **Const removal**: Removes `const` when `enum` is present (3.1-only feature)
3. **Schema renaming**: Converts auto-generated discriminated union names to clean names (e.g., `RootModel_Annotated_Union_...` → `CreateLLMModelEndpointV1Request`)

### Client Library Workflow

```
┌─────────────┐     generate      ┌──────────────────┐
│ Model Engine│ ─────────────────▶│ openapi-3.0.json │
│   (FastAPI) │                   └────────┬─────────┘
└─────────────┘                            │
                                           │ copy to client repos
                              ┌────────────┴────────────┐
                              ▼                         ▼
                    ┌─────────────────┐       ┌─────────────────┐
                    │ launch-python-  │       │ other clients   │
                    │ client          │       │                 │
                    └────────┬────────┘       └────────┬────────┘
                             │                         │
                             ▼                         ▼
                    ┌─────────────────┐       ┌─────────────────┐
                    │ OpenAPI Generator│       │ OpenAPI Generator│
                    │ (python, 6.4.0) │       │ (any language)  │
                    └─────────────────┘       └─────────────────┘
```

### Updating Client Libraries

When the API changes:

1. Generate new schemas in model-engine:
   ```bash
   python scripts/generate_openapi_schemas.py specs/
   ```

2. Copy `specs/openapi-3.0.json` to client repos as `openapi.json`

3. Run the client's code generator (see client repo for specific commands)

4. Test and commit

## Other Scripts

### Generating OpenAI Types

For OpenAI-compatible V2 APIs, we generate Pydantic models from OpenAI's spec:

1. Fetch spec from https://github.com/openai/openai-openapi/blob/master/openapi.yaml
2. Run `scripts/generate-openai-types.sh`

## Local Development

### Testing the HTTP Forwarder

Start an endpoint on port 5005:

```bash
export IMAGE=692474966980.dkr.ecr.us-west-2.amazonaws.com/vllm:0.10.1.1-rc2
export MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
export MODEL_PATH=/data/model_files/$MODEL
export REPO_PATH=/mnt/home/dmchoi/repos/scale

docker run \
    --runtime nvidia \
    --shm-size=16gb \
    --gpus '"device=0,1,2,3"' \
    -v $MODEL_PATH:/workspace/model_files:ro \
    -v ${REPO_PATH}/llm-engine/model-engine/model_engine_server/inference/vllm/vllm_server.py:/workspace/vllm_server.py \
    -p 5005:5005 \
    --name vllm \
    ${IMAGE} \
    python -m vllm_server --model model_files --port 5005 --disable-log-requests --max-model-len 4096 --max-num-seqs 16 --enforce-eager
```

Run the forwarder:

```bash
GIT_TAG=test python model_engine_server/inference/forwarding/http_forwarder.py \
    --config model_engine_server/inference/configs/service--http_forwarder.yaml \
    --num-workers 1 \
    --set "forwarder.sync.extra_routes=['/v1/chat/completions','/v1/completions']" \
    --set "forwarder.stream.extra_routes=['/v1/chat/completions','/v1/completions']" \
    --set "forwarder.sync.healthcheck_route=/health" \
    --set "forwarder.stream.healthcheck_route=/health"
```

Test it:

```bash
curl -X POST localhost:5000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"args": {"model":"meta-llama/Meta-Llama-3.1-8B-Instruct", "messages":[{"role": "system", "content": "Hello"}], "max_tokens":100}}'
```
