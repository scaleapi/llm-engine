# Model Engine Architecture

**Audience:** Service owners and deployment engineers installing, operating, or debugging model engine in a customer environment.

**Scope:** This document covers system structure, lifecycle flows, cross-cutting concerns, and component deep-dives. Configuration reference is in `helm-values.md`. Per-cloud behavior differences are in `cloud-matrix.md`.

---

## 1. System Structure

### 1.1 Architecture Overview

Model engine consists of five core pods and a set of external dependencies. The control plane (Gateway, Service Builder, K8s Cacher) runs in the model engine namespace. Inference pods run in a separate endpoint namespace, typically `llm-engine`.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  Control Plane Namespace (e.g. model-engine)                                     │
│                                                                                  │
│  ┌─────────────┐   REST    ┌─────────────────┐                                  │
│  │   Gateway   │──────────▶│  Service Builder │                                  │
│  │  (FastAPI)  │           │  (Celery worker) │                                  │
│  └──────┬──────┘           └────────┬─────────┘                                 │
│         │  read                     │ write K8s                                  │
│         │  endpoint                 │ resources                                  │
│         ▼  status                   ▼                                            │
│  ┌─────────────┐           ┌─────────────────┐                                  │
│  │  K8s Cacher │──────────▶│     Redis       │                                  │
│  │ (Deployment)│  write     │  (cache store)  │                                  │
│  └─────────────┘  TTL 60s  └─────────────────┘                                 │
│                                                                                  │
│  ┌──────────────────┐                                                            │
│  │ Celery Autoscaler│  (scales async endpoint workers by queue depth)            │
│  │  (StatefulSet)   │                                                            │
│  └──────────────────┘                                                            │
│                                                                                  │
│  ┌──────────────┐                                                                │
│  │ Balloon Pods │  (low-priority GPU placeholder pods, one Deployment per GPU)  │
│  └──────────────┘                                                                │
└──────────────────────────────────────────────────────────────────────────────────┘

External Dependencies
┌────────────────┐  ┌──────────────┐  ┌──────────────────────┐  ┌────────────────┐
│   PostgreSQL   │  │    Redis     │  │   Message Broker     │  │ Object Storage │
│  (endpoint DB) │  │ (K8s cache)  │  │ SQS / ASB / Redis    │  │ S3 / GCS / ABS │
└────────────────┘  └──────────────┘  └──────────────────────┘  └────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────┐
│  Endpoint Namespace (e.g. llm-engine)                                            │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────┐                        │
│  │  Sync/Streaming Endpoint (Deployment)               │                        │
│  │  ┌──────────────┐  ┌──────────────────────────────┐ │                        │
│  │  │ HTTP Forwarder│  │  vLLM / inference process   │ │                        │
│  │  └──────────────┘  └──────────────────────────────┘ │                        │
│  └─────────────────────────────────────────────────────┘                        │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────┐                        │
│  │  Async Endpoint (Deployment)                        │                        │
│  │  ┌──────────────────────────────────────────────┐   │                        │
│  │  │  Celery worker (reads from SQS/ASB/Redis)    │   │                        │
│  │  └──────────────────────────────────────────────┘   │                        │
│  └─────────────────────────────────────────────────────┘                        │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────┐                        │
│  │  Multi-node Endpoint (LeaderWorkerSet / LWS)        │                        │
│  │  ┌──────────────┐  ┌──────────────┐                 │                        │
│  │  │  Leader pod  │  │  Worker pods │  (no HPA/KEDA)  │                        │
│  │  └──────────────┘  └──────────────┘                 │                        │
│  └─────────────────────────────────────────────────────┘                        │
└──────────────────────────────────────────────────────────────────────────────────┘
```

**Data flow summary:**

- **Endpoint creation:** Client → Gateway REST → broker queue → Service Builder Celery worker → K8s API
- **Sync inference:** Client → Gateway → HTTP forward to inference pod → response
- **Async inference:** Client → Gateway → broker queue → Celery worker in inference pod → result stored → Client polls
- **Streaming inference:** Client → Gateway → SSE stream from inference pod
- **Status reads:** Gateway → Redis (written by K8s Cacher, not K8s API directly)

### 1.2 Kubernetes Resource Inventory

Resources created and managed by the helm chart (control plane):

| Resource | Kind | Notes |
|---|---|---|
| `model-engine-gateway` | Deployment | FastAPI server; replicas configured via `replicaCount.gateway` |
| `model-engine-builder` | Deployment | Celery worker for endpoint creation; replicas via `replicaCount.builder` |
| `model-engine-cacher` | Deployment | K8s cache loop; typically 1 replica (`replicaCount.cacher`) |
| `model-engine-celery-autoscaler` | StatefulSet | Scales async endpoint workers; shards via `celery_autoscaler.num_shards` |
| `model-engine-gateway` | HPA | Autoscales gateway replicas based on concurrency |
| `model-engine-config` | ConfigMap | Runtime config mounted into all control plane pods |
| `model-engine` | ServiceAccount | Used by control plane pods |
| `model-engine` | ClusterRole + ClusterRoleBinding | K8s API access for Service Builder and Cacher |
| Balloon Deployments | Deployment (one per GPU type) | Low-priority placeholder pods; see `balloons` in values |

Resources created per inference endpoint (in endpoint namespace):

| Resource | Kind | Condition |
|---|---|---|
| Inference Deployment | Deployment | All non-LWS endpoints |
| LeaderWorkerSet | LeaderWorkerSet (CRD) | Multi-node endpoints only |
| K8s Service | Service | Sync and streaming endpoints |
| HPA | HorizontalPodAutoscaler | Sync/streaming, `min_workers > 0` |
| KEDA ScaledObject | ScaledObject (CRD) | Sync/streaming, `min_workers == 0` |
| PodDisruptionBudget | PodDisruptionBudget | All endpoints (configurable) |
| Istio VirtualService | VirtualService | Sync/streaming, `istio_enabled: true` |
| Istio DestinationRule | DestinationRule | Sync/streaming, `istio_enabled: true` |
| Istio ServiceEntry | ServiceEntry | Multi-node + `istio_enabled: true` |
| SQS Queue / ASB Topic | Cloud resource | Async endpoints and all endpoints on async clouds |

!!! note "HPA and KEDA are mutually exclusive"
    The Service Builder enforces this: when creating or updating an endpoint, it deletes the KEDA ScaledObject before creating an HPA (if `min_workers > 0`), or deletes the HPA before creating a KEDA ScaledObject (if `min_workers == 0`). Both never coexist on the same endpoint.

### 1.3 External Dependencies and Prerequisites

The following must exist and be reachable from the cluster before `helm install`:

| Dependency | Required For | Notes |
|---|---|---|
| PostgreSQL | All operations | Endpoint metadata, bundle records, batch job records |
| Redis | Gateway routing, cacher, async metrics | Two logical roles: K8s cache and inference autoscaling metrics |
| Message broker (SQS / ASB / Redis) | Async endpoints; endpoint creation queue | Cloud-dependent; see §3.3 |
| Object storage (S3 / GCS / ABS) | LLM artifacts, fine-tune repos, batch job progress | Cloud-dependent |
| Image registry (ECR / ACR / GAR) | All image pulls | Must be mirrored from `public.ecr.aws/b2z8n5q1/` in customer envs |
| Prometheus | KEDA scale-to-zero | Required if any sync endpoint uses `min_workers == 0`; see §3.1 |
| KEDA | Scale-to-zero | Must be installed in cluster if any endpoint uses `min_workers == 0` |
| Istio | VirtualService routing, mTLS | Optional but strongly recommended; set `istio_enabled: true/false` to match actual state |
| NVIDIA GPU Operator | GPU inference | Required for GPU workloads; nodes must be labeled and driver-ready |

!!! warning "Image registry mirroring"
    In customer environments, all model engine images must be mirrored from the public ECR source (`public.ecr.aws/b2z8n5q1/`) to the customer registry before installation. The `vllm_repository` value defaults to a relative path that resolves to Scale's internal ECR in many deployment configurations and **must be overridden**. Failing to mirror is the most common silent deployment failure: endpoint creation returns HTTP 200 but the endpoint stays `INITIALIZING` indefinitely.

---

## 2. Lifecycle Flows

### 2.1 Generic Endpoint Creation Flow

The endpoint creation path is identical for all endpoint types (sync, async, streaming, multi-node). The LLM API layer (§2.3) is a higher-level wrapper that feeds into the same flow.

```
Client
  │
  │  POST /v1/model-endpoints
  ▼
Gateway (FastAPI)
  │  Validates request, writes endpoint record to PostgreSQL (status: PENDING)
  │  Enqueues Celery task to endpoint creation queue (SQS / ASB / Redis)
  │
  ▼
Message Broker
  │  Task sits in queue (SQS queue / ASB topic / Redis queue)
  │
  ▼
Service Builder (Celery worker)
  │  Dequeues task
  │  Calls K8s API to create/update:
  │    - Deployment or LeaderWorkerSet
  │    - HPA or KEDA ScaledObject (sync/streaming, non-LWS only)
  │    - K8s Service (sync/streaming only)
  │    - Istio VirtualService + DestinationRule (if istio_enabled, non-LWS)
  │    - Istio ServiceEntry (if istio_enabled, LWS only)
  │    - PodDisruptionBudget
  │  Updates endpoint record in PostgreSQL (status: INITIALIZING → READY)
  │
  ▼
K8s Cacher (background loop, every 15s)
  │  Reads endpoint state from K8s API
  │  Writes to Redis with 60s TTL
  │
  ▼
Gateway
  │  Reads endpoint status from Redis (not K8s API directly)
  │  Returns status to client via GET /v1/model-endpoints/{id}
```

**Timing constraints:**

- The Celery task has a **30-minute hard timeout**. Endpoint creation that exceeds this ceiling (e.g., very large image pulls on cold nodes) will fail with no retry, and the endpoint will be stuck `INITIALIZING`.
- The K8s Cacher runs on a **15-second poll cycle**. After the Service Builder marks an endpoint `READY` in PostgreSQL, there is a brief window (up to 15s) before the Gateway's Redis cache reflects the new state. During this window, status reads may lag.

!!! warning "Celery task timeout is a hard ceiling"
    The 30-minute Celery task timeout applies to the entire endpoint creation operation, including image pull time. For large model images on cold nodes, image pull alone can approach this limit. Plan capacity accordingly and ensure balloon pods keep GPU nodes warm so image pulls start quickly.

### 2.2 Inference Flows

#### Synchronous Inference

```
Client
  │  POST /v1/model-endpoints/{id}/predict
  ▼
Gateway
  │  Looks up endpoint URL from Redis cache
  │  HTTP POST directly to inference pod's HTTP forwarder
  ▼
Inference Pod (HTTP Forwarder + vLLM / model process)
  │  Processes request, returns response
  ▼
Gateway → Client  (response forwarded synchronously)
```

The Gateway does not queue synchronous requests. The inference pod must be reachable at the time of the request. If the pod is not yet ready or has been evicted, the client receives an error immediately.

#### Asynchronous Inference

```
Client
  │  POST /v1/model-endpoints/{id}/predict  (async endpoint)
  ▼
Gateway
  │  Enqueues Celery task to inference queue
  │  (per-endpoint SQS queue / ASB topic / Redis queue)
  │  Returns task_id immediately (HTTP 200)
  ▼
Message Broker (per-endpoint queue)
  │
  ▼
Celery Worker (inside inference pod)
  │  Dequeues task
  │  Runs inference
  │  Stores result in Celery result backend (Redis / SQS)
  ▼
Client polls GET /v1/tasks/{task_id}
  │
  ▼
Gateway
  │  Reads task result from Celery result backend
  │  Returns status: PENDING / SUCCESS / FAILURE
```

Each async endpoint has its own dedicated queue: one SQS queue per endpoint on AWS, one ASB topic per endpoint on Azure. The Celery Autoscaler monitors queue depth and scales the Deployment's replica count accordingly (see §3.1).

#### Streaming Inference

Streaming follows the same routing path as synchronous inference. The Gateway establishes a Server-Sent Events (SSE) connection to the inference pod and streams response chunks back to the client as they arrive. The inference pod must support streaming — vLLM does natively via its `/v1/chat/completions` and `/v1/completions` endpoints with `stream=true`.

### 2.3 LLM API Layer

Model engine exposes two API surfaces for LLM inference:

| API Surface | Routes | Description |
|---|---|---|
| Generic endpoint API | `GET/POST /v1/model-endpoints`, `/v1/model-endpoints/{id}/predict` | Low-level; caller specifies image, resources, and all parameters explicitly |
| LLM endpoint API v1 | `/v1/llms/...` | Higher-level; opinionated defaults, auto-selects vLLM image and hardware |
| LLM endpoint API v2 | `/v2/...` | OpenAI-compatible; same infrastructure as v1 LLM API |

**v1 vs v2:**

- **v1** (`/v1/llms/...`): Model engine's native LLM API. Returns model engine response format.
- **v2** (`/v2/...`): OpenAI-compatible API. Accepts and returns the same request/response format as OpenAI's API, including `stream=true` for SSE streaming. Pydantic models are generated from OpenAI's official OpenAPI spec. Endpoints: `POST /v2/chat/completions`, `POST /v2/completions`.

**How LLM endpoints use Service Builder:**

The LLM endpoint API (`LiveLLMModelEndpointService`) is a thin wrapper over the generic `LiveModelEndpointService`. When a client calls `POST /v1/llms` to create an LLM endpoint, the service translates a `CreateLLMModelEndpointV1Request` into a `CreateModelEndpointV1Request` with opinionated defaults — vLLM image from `vllm_repository`, resource sizing from `recommendedHardware`, GPU type selection — and then delegates to the same Service Builder queue path described in §2.1. There is no separate infrastructure for LLM endpoints. They are regular model endpoints with a curated configuration. All failure modes from §2.1 apply equally.

**`recommendedHardware` auto-selection:**

The `recommendedHardware` helm value contains a lookup table keyed by GPU memory requirement (`byGpuMemoryGb`) and by model name (`byModelName`). When an LLM endpoint is created without explicit resource specifications, the service queries this table to select GPU type, GPU count, CPU, memory, storage, and `nodes_per_worker`. When `nodes_per_worker > 1`, the service creates a multi-node (LWS) endpoint instead of a regular Deployment. See §3.4 for details.

---

## 3. Cross-cutting Concerns

### 3.1 Autoscaling

Model engine uses three distinct autoscaling mechanisms depending on endpoint type and configuration. They are not interchangeable, and only one mechanism applies to any given endpoint at a time.

#### Sync and Streaming Endpoints: HPA (`min_workers > 0`)

When `min_workers > 0`, the Service Builder creates a `HorizontalPodAutoscaler` targeting the endpoint's Deployment. The HPA scales based on CPU and memory metrics. The autoscaling API version is selected based on cluster version: `autoscaling/v2` for Kubernetes >= 1.26, `autoscaling/v2beta2` for Kubernetes 1.23–1.25.

```
min_workers > 0  →  KEDA ScaledObject deleted (if exists)  →  HPA created
```

#### Sync and Streaming Endpoints: KEDA (`min_workers == 0`)

When `min_workers == 0`, the Service Builder creates a KEDA `ScaledObject` instead of an HPA. KEDA uses request concurrency metrics sourced from Prometheus to decide when to scale the endpoint from 0 replicas to 1 replica.

```
min_workers == 0  →  HPA deleted (if exists)  →  KEDA ScaledObject created
```

!!! warning "KEDA requires `prometheus_server_address`"
    KEDA-based scale-to-zero **requires** `config.values.infra.prometheus_server_address` to be set in helm values. Without it, the `can_scale_http_endpoint_from_zero_flag` is `False` and scale-to-zero will silently not work. This is enforced in `dependencies.py`:

    ```python
    can_scale_http_endpoint_from_zero_flag=infra_config().prometheus_server_address is not None
    ```

    This is one of the most non-obvious configuration dependencies in the system. The endpoint creation will succeed and the KEDA ScaledObject will be created, but scaling will not function.

!!! warning "Known limitation: KEDA only scales 0→1, not 1→N"
    As of the current codebase, KEDA ScaledObjects only support scaling a sync endpoint from 0 replicas to 1 replica. Scaling from 1 to N is not implemented. This is a documented TODO in `k8s_endpoint_resource_delegate.py`:

    ```python
    # Right now, keda only will support scaling from 0 to 1
    # TODO support keda scaling from 1 to N as well
    if request.build_endpoint_request.min_workers > 0:
        # ... create HPA
    else:  # min workers == 0, use keda
        # ... create KEDA ScaledObject
    ```

    For endpoints that need to scale beyond 1 replica, use `min_workers >= 1` (which triggers HPA instead of KEDA).

#### Async Endpoints: Celery Autoscaler

Async endpoints are scaled by the Celery Autoscaler StatefulSet, not by HPA or KEDA. The Celery Autoscaler monitors the depth of each endpoint's message queue (SQS queue on AWS, ASB topic on Azure, Redis queue on GCP/on-prem) and adjusts the Deployment's replica count by patching the K8s API directly.

The number of autoscaler shards is configured via `celery_autoscaler.num_shards`. Multiple shards distribute the monitoring load across many concurrent endpoints. The Celery Autoscaler is enabled via `celery_autoscaler.enabled: true`.

#### Multi-node (LWS) Endpoints: No Autoscaling

LeaderWorkerSet endpoints do not support autoscaling. `min_workers` must equal `max_workers`. No HPA or KEDA ScaledObject is created. Capacity changes require deleting and recreating the endpoint.

#### Autoscaling Summary

| Endpoint Type | `min_workers` | Scaler | Metric Source |
|---|---|---|---|
| Sync / Streaming | `> 0` | HPA | CPU / memory |
| Sync / Streaming | `== 0` | KEDA ScaledObject | Prometheus (request concurrency) |
| Async | any | Celery Autoscaler StatefulSet | Queue depth (SQS / ASB / Redis) |
| Multi-node (LWS) | must equal `max_workers` | None | — |

### 3.2 Observability

**Structured logging:**
All control plane components emit structured JSON logs. Log verbosity is controlled via `debug_mode` in helm values.

**Datadog APM (optional):**
Enabled by setting `dd_trace_enabled: true` in `config.values.launch` and installing the Datadog agent in the cluster. When enabled, the `DatadogMonitoringMetricsGateway` is used instead of `FakeMonitoringMetricsGateway`. This gates distributed tracing and APM metrics. The top-level `datadog.enabled` helm value controls Datadog agent sidecar injection.

```python
# from dependencies.py
if hmi_config.dd_trace_enabled:
    monitoring_metrics_gateway = DatadogMonitoringMetricsGateway()
else:
    monitoring_metrics_gateway = FakeMonitoringMetricsGateway()
```

**Prometheus metrics:**
Request concurrency metrics are exposed and consumed by KEDA for scale-to-zero. The Prometheus server must be reachable at the address configured in `prometheus_server_address`. See §3.1 for the dependency.

**OpenTelemetry tracing:**
An OTel-based telemetry design is in progress and not yet in production. Current tracing is provided via the `TracingGateway` abstraction, with Datadog as the primary production implementation.

**K8s Cacher readiness probe:**
The K8s Cacher writes a readiness file (`READYZ_FPATH`) after its first successful loop iteration. This gates the cacher pod's `readinessProbe`, ensuring the Redis cache has at least one warm cycle before the pod is considered ready.

### 3.3 Cloud Backend Abstraction

The `config.values.infra.cloud_provider` value is the single switch that drives selection of broker, storage, registry, and auth implementations at runtime. This selection happens in `dependencies.py` and `k8s_cache.py` on startup. Changing this value without corresponding infrastructure changes will cause runtime failures.

#### Broker (message queue) selection

| `cloud_provider` | Endpoint creation queue | Async inference queue | Queue delegate |
|---|---|---|---|
| `aws` (default) | SQS | SQS | `SQSQueueEndpointResourceDelegate` |
| `azure` | Azure Service Bus | Azure Service Bus | `ASBQueueEndpointResourceDelegate` |
| `gcp` | Redis (Memorystore) | Redis (Memorystore) | `RedisQueueEndpointResourceDelegate` |
| `onprem` | Redis | Redis | `OnPremQueueEndpointResourceDelegate` |

!!! note "Redis broker is the legacy path"
    Redis was the original broker for all clouds. SQS (AWS) and Azure Service Bus (Azure) replaced it due to reliability and scale limitations. GCP and on-prem still use Redis as the broker. Redis-as-broker has known reliability limitations compared to SQS and ASB.

!!! warning "Azure Service Bus idle connection drops"
    Azure Service Bus drops idle AMQP connections after approximately 300 seconds. This manifests as random 503 errors on async inference with no obvious configuration cause. The fix is `broker_pool_limit=0` (disables connection pooling, forcing reconnection on each use). This was resolved in a recent commit — verify your deployment includes the fix before deploying to Azure.

#### Storage selection

| `cloud_provider` | Filesystem gateway | LLM artifact gateway | File storage gateway |
|---|---|---|---|
| `aws` / `onprem` | `S3FilesystemGateway` | `S3LLMArtifactGateway` | `S3FileStorageGateway` |
| `azure` | `ABSFilesystemGateway` | `ABSLLMArtifactGateway` | `ABSFileStorageGateway` |
| `gcp` | `GCSFilesystemGateway` | `GCSLLMArtifactGateway` | `GCSFileStorageGateway` |

On-prem uses S3-compatible storage (MinIO or equivalent) via the same S3 gateways as AWS.

#### Registry selection

| `cloud_provider` | Docker repository class |
|---|---|
| `aws` (default) | `ECRDockerRepository` |
| `azure` | `ACRDockerRepository` |
| `gcp` | `GARDockerRepository` |
| `onprem` | `OnPremDockerRepository` |

#### Inference autoscaling metrics gateway selection

| `cloud_provider` | Autoscaling metrics gateway |
|---|---|
| `azure` | `ASBInferenceAutoscalingMetricsGateway` |
| all others | `RedisInferenceAutoscalingMetricsGateway` |

#### Fine-tune repository selection

| `cloud_provider` | Fine-tune repository | Fine-tune events repository |
|---|---|---|
| `aws` / `onprem` | `S3FileLLMFineTuneRepository` | `S3FileLLMFineTuneEventsRepository` |
| `azure` | `ABSFileLLMFineTuneRepository` | `ABSFileLLMFineTuneEventsRepository` |
| `gcp` | `GCSFileLLMFineTuneRepository` | `GCSFileLLMFineTuneEventsRepository` |

### 3.4 GPU and Hardware Configuration

#### Node selectors and GPU labels

Inference pods are scheduled to GPU nodes using the `k8s.amazonaws.com/accelerator` node label. This label must be present on GPU nodes before endpoints can be created. The GPU types referenced across model engine configuration:

| Label value | GPU |
|---|---|
| `nvidia-ampere-a10` | NVIDIA A10 |
| `nvidia-ampere-a100` | NVIDIA A100 |
| `nvidia-tesla-t4` | NVIDIA T4 |
| `nvidia-hopper-h100` | NVIDIA H100 (full) |
| `nvidia-hopper-h100-1g20gb` | NVIDIA H100 (MIG 1g.20gb) |
| `nvidia-hopper-h100-3g40gb` | NVIDIA H100 (MIG 3g.40gb) |

GPU nodes must have the `nvidia.com/gpu: NoSchedule` taint that GPU inference pods tolerate. The NVIDIA GPU Operator must be installed and the driver must be functional on every GPU node (`nvidia-smi` must succeed).

#### Balloon pods and GPU node warming

The `balloons` helm value creates one low-priority Deployment per accelerator type. Each balloon Deployment occupies a configurable number of replicas (`replicaCount`) on the corresponding node type, requesting GPU resources to prevent the cluster autoscaler from scaling down GPU nodes between inference workloads.

The `balloonConfig.reserveHighPriority: true` flag restricts eviction to only high-priority pods. When a real inference pod is scheduled, it evicts balloon pods to claim GPU resources. Setting `replicaCount: 0` for a GPU type disables warming for that node type.

```yaml
# Example: keep 2 H100 nodes and 1 A10 node warm
balloonConfig:
  reserveHighPriority: true

balloons:
  - acceleratorName: nvidia-hopper-h100
    replicaCount: 2
    gpuCount: 4
  - acceleratorName: nvidia-ampere-a10
    replicaCount: 1
  - acceleratorName: cpu
    replicaCount: 0   # disabled
```

#### `recommendedHardware` auto-selection

The `recommendedHardware` helm value provides two lookup tables used by the LLM endpoint service:

- `byGpuMemoryGb`: Matches on `gpu_memory_le` (less-than-or-equal GB of model GPU memory). Selects GPU type, GPU count, CPU, memory, storage, and `nodes_per_worker`.
- `byModelName`: Named overrides that take precedence over the `byGpuMemoryGb` table for specific models.

```yaml
recommendedHardware:
  byGpuMemoryGb:
    - gpu_memory_le: 24
      cpus: 10
      gpus: 1
      memory: 24Gi
      storage: 80Gi
      gpu_type: nvidia-ampere-a10
      nodes_per_worker: 1
    - gpu_memory_le: 180
      cpus: 20
      gpus: 2
      memory: 160Gi
      storage: 160Gi
      gpu_type: nvidia-hopper-h100
      nodes_per_worker: 1
    - gpu_memory_le: 640
      cpus: 80
      gpus: 8
      memory: 800Gi
      storage: 640Gi
      gpu_type: nvidia-hopper-h100
      nodes_per_worker: 2       # triggers LWS creation
  byModelName:
    - name: deepseek-coder-v2
      cpus: 160
      gpus: 8
      memory: 800Gi
      storage: 640Gi
      gpu_type: nvidia-hopper-h100
      nodes_per_worker: 1
```

When `nodes_per_worker > 1`, the LLM endpoint service creates a multi-node (LWS) endpoint instead of a regular Deployment. This is the mechanism by which large models are automatically placed on multi-node configurations without requiring the caller to specify resource details.

#### `imageCache`

The `imageCache` helm value defines per-node-type image pre-pulling configuration. Each entry specifies a `nodeSelector` and optional tolerations matching a GPU node pool. Pre-pulling model images onto nodes reduces inference pod startup time. This is distinct from balloon pods: balloon pods keep nodes allocated; `imageCache` keeps images warm on those nodes.

---

## 4. Component Reference

### 4.1 K8s Cacher

**What it does:**
The K8s Cacher is a standalone Deployment (typically 1 replica) that runs a continuous polling loop. Every `sleep_interval_seconds` (default: **15 seconds**), it:

1. Reads the current state of all model endpoint Deployments and LeaderWorkerSets from the K8s API
2. Writes endpoint status records to Redis with a TTL of `ttl_seconds` (default: **60 seconds**)
3. Updates the image cache state (for the `imageCache` feature)

**Why it exists:**
Direct K8s API calls from Gateway pods were unreliable at scale — requests would time out under load. The Cacher decouples Gateway reads from K8s API polling, with Redis as the intermediary. The Gateway reads exclusively from Redis for endpoint status; it never calls the K8s API for status lookups at request time.

**Code path:**
```
k8s_cache.py (main loop, --sleep-interval-seconds)
  └─ ModelEndpointCacheWriteService.execute()
       ├─ LiveEndpointResourceGateway  →  K8s API (reads Deployments / LWS)
       └─ RedisModelEndpointCacheRepository.write(ttl=60s)
```

**Startup behavior:**
The cacher calls `load_incluster_config()` first (for in-cluster operation), falling back to `load_kube_config()` for local development. It writes a readiness file after the first successful loop iteration to gate its `readinessProbe` — the pod is not considered ready until at least one cache cycle has completed successfully.

!!! danger "Failure mode: Redis auth broken → endpoint status `unknown`"
    If the cacher cannot write to Redis — due to misconfigured Redis auth, network partition, or expired credentials — it fails silently from the Gateway's perspective. The Gateway reads stale or absent Redis entries and returns endpoint status as `"unknown"`, not an error and not `INITIALIZING`.

    **This is the most deceptive failure mode in model engine.** An endpoint may be fully `READY` and serving traffic, but the status API returns `"unknown"` indefinitely because the cacher-to-Redis path is broken.

    How to diagnose: check cacher pod logs for Redis connection errors. Verify Redis auth credentials and network reachability from the cacher pod. In smoke tests, the signature is: Service Builder logs show the endpoint reached `READY`, but `GET /v1/model-endpoints/{id}` returns `"unknown"` without ever transitioning.

**Parameters (configurable via CLI args, set in helm Deployment spec):**

| Parameter | Default | Description |
|---|---|---|
| `--ttl-seconds` | `60` | Redis TTL for cache entries |
| `--sleep-interval-seconds` | `15` | Poll interval between K8s API reads |
| `--redis-url-override` | None | Override the Redis URL from `hmi_config.cache_redis_url` |

!!! warning "TTL must be greater than sleep interval"
    If `ttl_seconds < sleep_interval_seconds`, cache entries expire between writes, causing cache misses on every Gateway status request. The cacher logs a warning if this condition is detected, but does not fail or exit. The default values (60s TTL, 15s interval) satisfy this requirement with a 4x margin.

### 4.2 Balloon Pods

**What they do:**
Balloon pods are low-priority Deployments that run an `ubuntu` container with an infinite sleep command. One Deployment exists per GPU type, configured via the `balloons` helm value. They request GPU resources, causing the cluster autoscaler to provision GPU nodes and keep them allocated even when no inference pods are running.

**Why they exist:**
GPU nodes are expensive to run continuously but slow to provision (5–15 minutes for a new node to join and be ready). Without balloon pods, the cluster autoscaler scales GPU nodes down during idle periods. When a new endpoint is created, the cluster must provision a fresh GPU node, and the 30-minute Celery task timeout (§2.1) starts counting during this wait. Balloon pods eliminate this cold-start delay.

**How eviction works:**
Balloon pods are created with a low PriorityClass. When a real inference pod needs to be scheduled on a node occupied by a balloon pod, Kubernetes evicts the balloon pod (preemption). The `balloonConfig.reserveHighPriority: true` setting restricts preemption to only high-priority pods, preventing lower-priority workloads from accidentally evicting balloons and defeating the warming strategy.

**Configuration:**
```yaml
balloonConfig:
  reserveHighPriority: true

balloons:
  - acceleratorName: nvidia-ampere-a10
    replicaCount: 1
  - acceleratorName: nvidia-ampere-a100
    replicaCount: 0       # disabled — no A100 node warming
  - acceleratorName: nvidia-hopper-h100
    replicaCount: 2
    gpuCount: 4           # request 4 GPUs per balloon pod
  - acceleratorName: cpu
    replicaCount: 0
```

!!! note "`replicaCount: 0` disables a balloon type"
    Setting `replicaCount: 0` for a GPU type disables node warming for that type. Cold-start delays will occur on the first endpoint creation after a period of inactivity on that GPU type. This is the default for all GPU types in `values_sample.yaml` — production deployments should set non-zero counts for GPU types in active use.

### 4.3 Multi-node Endpoints (LWS)

**What they are:**
Multi-node endpoints use `LeaderWorkerSet` (LWS), a Kubernetes CRD designed for distributed inference workloads that span multiple nodes. LWS is required for models too large to fit on a single node's GPU memory (e.g., 70B+ parameter models requiring more than 8 GPUs).

**How they differ from regular Deployments:**

| Aspect | Regular Deployment | LeaderWorkerSet |
|---|---|---|
| K8s resource kind | `Deployment` | `LeaderWorkerSet` (CRD) |
| Autoscaling | HPA or KEDA | None |
| `min_workers` vs `max_workers` | Can differ | Must be equal |
| Istio resources created | VirtualService + DestinationRule | ServiceEntry only |
| K8s Service template | `service.yaml` | `lws-service.yaml` |
| Scale-to-zero | Supported (via KEDA) | Not supported |
| Capacity change | Update `min_workers`/`max_workers` | Delete and recreate |

**When LWS is used:**
The LLM endpoint service selects LWS automatically when `nodes_per_worker > 1` in the matched `recommendedHardware` entry. It can also be specified explicitly in a `CreateModelEndpointV1Request` by setting `nodes_per_worker > 1`.

**Resource creation differences in Service Builder:**
For LWS endpoints, the Service Builder takes a different code branch:

- Creates a `LeaderWorkerSet` resource instead of a `Deployment`
- Creates the K8s Service from `lws-service.yaml` (not the standard `service.yaml`)
- If `istio_enabled: true`, creates a `ServiceEntry` (not a `VirtualService` or `DestinationRule`) — required because LWS routing uses direct IP address resolution rather than Istio's standard hostname-based VirtualService routing
- Does **not** create an HPA or KEDA ScaledObject

**Istio and LWS routing:**
LWS endpoints require a workaround for Istio. The Gateway manually resolves the K8s Service cluster IP and sends requests directly to that IP, bypassing Istio's standard VirtualService routing. A `ServiceEntry` is created to allow this direct IP traffic to pass through Istio's policy enforcement. See `live_sync_model_endpoint_inference_gateway.py` and `live_streaming_model_endpoint_inference_gateway.py` for the implementation details.

!!! warning "No autoscaling for LWS endpoints"
    LeaderWorkerSet endpoints cannot be autoscaled. `min_workers` must equal `max_workers` at creation time. If you need different capacity, delete the endpoint and recreate it with the desired worker count. This is a known limitation with no current workaround.

---

## Appendix: Key Configuration Values Quick Reference

The values below have the highest operational impact. Full reference is in `helm-values.md`.

| Value | Default | Risk | Impact if wrong |
|---|---|---|---|
| `db.runDbMigrationScript` | `false` | **HIGH** | Schema errors on first deploy; no clear error surface |
| `config.values.infra.prometheus_server_address` | unset | **HIGH** | KEDA scale-to-zero silently broken |
| `config.values.launch.vllm_repository` | `vllm` (relative) | **HIGH** | Resolves to Scale's internal ECR in many envs; image pull fails silently |
| `celeryBrokerType` | `sqs` | **HIGH** | Wrong broker for cloud → async endpoints broken |
| `config.values.infra.cloud_provider` | `aws` | **HIGH** | Wrong storage, broker, and auth clients loaded for cloud |
| `balloons[*].replicaCount` | `0` | **MEDIUM** | No GPU node warming → cold-start delays; risks hitting 30-min Celery timeout |
| `celery_autoscaler.enabled` | `true` | **MEDIUM** | Async endpoints never scale if disabled |
| `config.values.launch.istio_enabled` | `true` | **MEDIUM** | Must match actual cluster Istio installation state exactly |

!!! warning "`db.runDbMigrationScript` defaults to `false`"
    On first install, the database schema must be initialized. The default `false` means the migration job does not run, resulting in schema errors at runtime that have no clear error surface. Set `db.runDbMigrationScript: true` on every first install into a new environment. There is an open TODO to change this default to `true`.
