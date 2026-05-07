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

### Control Plane Local Setup

The control plane (Gateway API server, Service Builder, K8s Cache) can be run entirely
locally without GPU hardware or cloud credentials. Endpoint creation calls succeed
against a fake k8s/SQS/ECR backend, letting you iterate on control plane code quickly.

**Prerequisites:** Python 3.10+, Docker

#### One-time setup

```bash
cd model-engine/

# Install Python dependencies
make install

# Start Postgres + Redis
make dev-up

# Apply database migrations
make dev-migrate
```

#### Run the API server

```bash
make dev-server
```

The gateway starts at http://localhost:5000 with auto-reload on file changes.
Authentication is skipped automatically (`SKIP_AUTH=true`) so any token works.

#### Make API calls

```bash
# List model endpoints
curl http://localhost:5000/v1/model-endpoints \
  -H "Authorization: Bearer test-user"

# Create an LLM endpoint (uses fake k8s — no real infra needed)
curl -X POST http://localhost:5000/v1/llm/model-endpoints \
  -H "Authorization: Bearer test-user" \
  -H "Content-Type: application/json" \
  -d '{"name":"local-test","model_name":"meta-llama/Meta-Llama-3.1-8B-Instruct","inference_framework":"vllm","min_workers":0,"max_workers":1,"gpus":1,"gpu_type":"nvidia-ampere-a10","endpoint_type":"sync"}'
```

#### Stop backing services

```bash
make dev-down
```

#### What `LOCAL=true` does

Running with `LOCAL=true` (set automatically by `make dev-server` and `make dev-migrate`):

- Skips the `GIT_TAG` env var requirement
- Uses a **fake queue delegate** (no SQS/Azure Service Bus needed)
- Uses a **fake Docker repository** (no ECR/ACR/GAR needed)
- Auth is skipped when `identity_service_url` is absent from config (default)
- Postgres and Redis are real local services (via docker-compose)

This means you can create/update/delete endpoints via the API and see them reflected
in Postgres, without any Kubernetes cluster or cloud account.

#### Running individual components manually

If you prefer to set env vars yourself rather than use `make`:

```bash
export LOCAL=true
export GIT_TAG=local
export ML_INFRA_DATABASE_URL=postgresql://postgres:password@localhost:5432/llm_engine
export DEPLOY_SERVICE_CONFIG_PATH=$(pwd)/service_configs/service_config_local.yaml
export ML_INFRA_SERVICES_CONFIG_PATH=$(pwd)/model_engine_server/core/configs/default.yaml

# Gateway
start-fastapi-server --port 5000 --num-workers 1 --debug

# Database migration
bash model_engine_server/db/migrations/run_database_migration.sh
```

### Full End-to-End Local Flow (control plane + real inference pod)

This setup uses [kind](https://kind.sigs.k8s.io/) (Kubernetes in Docker) to run a real
local k8s cluster. The Service Builder creates actual Deployments in kind; the K8s Cacher
polls kind and updates Redis. No GPU required — we use the built-in echo server as the
inference container.

**Prerequisites:** Python 3.10+, Docker, [`kind`](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)

#### One-time cluster + image setup

```bash
cd model-engine/

# Start Postgres + Redis (if not already running)
make dev-up

# Apply DB migrations (if not already done)
make dev-migrate

# Create kind cluster and the model-engine namespace
make kind-up

# Build model-engine:local and load it into kind
make kind-image        # takes ~2-3 min on first build
```

#### Run the full stack (4 terminals)

```bash
# Terminal 1 — Gateway
make dev-server-full

# Terminal 2 — Service Builder (picks up endpoint creation tasks from Redis)
make dev-service-builder

# Terminal 3 — K8s Cacher (polls kind, writes endpoint status to Redis)
make dev-k8s-cacher
```

#### Create a test endpoint and watch it spin up

```bash
# Terminal 4 — create a sync CPU endpoint using the echo server
curl -X POST http://localhost:5000/v1/model-endpoints \
  -H "Authorization: Bearer test-user" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "local-echo",
    "bundle_name": "echo-bundle",
    "endpoint_type": "sync",
    "cpus": 0.25,
    "memory": "256Mi",
    "min_workers": 1,
    "max_workers": 1,
    "per_worker": 1,
    "model_bundle": {
      "name": "echo-bundle",
      "metadata": {},
      "flavor": {
        "flavor": "runnable_image",
        "repository": "model-engine",
        "tag": "local",
        "command": [
          "python", "-m",
          "model_engine_server.inference.forwarding.echo_server",
          "--port", "5005"
        ],
        "predict_route": "/predict",
        "healthcheck_route": "/healthz",
        "readiness_initial_delay_seconds": 15
      }
    }
  }'

# Poll status — transitions PENDING → UPDATE_PENDING → READY (30-60 s)
curl http://localhost:5000/v1/model-endpoints/<endpoint-id> \
  -H "Authorization: Bearer test-user"

# Watch the pod come up in kind
kubectl --context kind-llm-engine get pods -n model-engine -w
```

#### Tear down

```bash
make kind-down          # delete kind cluster
make dev-down           # stop Postgres + Redis
```

#### How the full flow works

| Component | Mode | What it does locally |
|---|---|---|
| Gateway (`dev-server-full`) | `cloud_provider=onprem` + `LOCAL=true` | Real Redis queue, fake Docker registry |
| Service Builder | `cloud_provider=onprem` + Redis broker | Creates real k8s Deployments in kind |
| K8s Cacher | `cloud_provider=onprem` | Polls kind, writes status to Redis |
| Inference pod | `model-engine:local` in kind | Runs echo server on port 5005 |
| Forwarder sidecar | `model-engine:local` in kind | HTTP forwarder proxies requests |

> **Note:** LLM endpoints (vLLM, TGI) require GPU hardware and pulling large images — use the generic sync endpoint with the echo server for local flow testing.

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
