# Proposal: `temporal` Endpoint Type in Launch

**Author:** lily.zhu@scale.com  
**Status:** RFC  
**Ticket:** MLI-6425

---

## Problem

Teams running multi-step GPU pipelines (e.g. robotics ego hand keypoints: SVO processing → Dyn-HaMR → hand annotations) need durable, retryable orchestration across heterogeneous GPU types. Today they have two options:

1. **Launch async endpoints (Celery)** — Launch manages the pods, but Celery gives no cross-step durability. If the H100 pod running Dyn-HaMR crashes mid-run, the whole pipeline restarts from step 1.
2. **Raw K8s Deployments via std-ml-srv** — bypasses Launch entirely; teams lose unified deployment API, GPU scheduling integration, and Launch dashboards.

Temporal solves the durability problem: if a pod crashes mid-activity, Temporal retries *only that activity* from the last heartbeat. Phases already completed are not re-run.

The gap is that Launch has no native way to create and manage Temporal activity worker pods. Teams either re-implement the same ~80-line Temporal worker boilerplate per service, or deploy outside Launch.

---

## Proposed Solution

Add `temporal` as a fourth endpoint type in `ModelEndpointType`. A `temporal` endpoint is a K8s Deployment whose pods connect to Temporal server and pull activity tasks from a named task queue — instead of polling a Celery queue.

**What Launch manages:**
- Pod lifecycle (create / update / delete)
- GPU scheduling and node selection
- Scaling (fixed replicas in MVP; Temporal-aware autoscaling as follow-up)
- Unified visibility in Launch dashboards

**What Launch does not touch:**
- Task submission (the Temporal workflow dispatches activities directly)
- Result routing (Temporal handles activity results)
- `/v1/async-tasks` API (not applicable for `temporal` endpoints)

---

## API Changes

### `POST /v1/model-endpoints`

New field on `CreateModelEndpointV1Request`:

```python
temporal_task_queue: Optional[str]
# Required when endpoint_type="temporal".
# The Temporal task queue that workers will poll.
# Example: "robotics-hand-keypoints-temporal"
```

Example request:

```json
{
  "name": "hand-keypoints-temporal",
  "endpoint_type": "temporal",
  "temporal_task_queue": "robotics-hand-keypoints-temporal",
  "gpus": 1,
  "gpu_type": "nvidia-hopper-h100",
  "cpus": 20,
  "memory": "200Gi",
  "storage": "250Gi",
  "min_workers": 0,
  "max_workers": 10,
  "per_worker": 1,
  "model_bundle_id": "...",
  "labels": {"team": "robotics", "product": "ego"}
}
```

The `model_bundle_id` points to a bundle whose command runs the Temporal activity worker (e.g. `python -m ml_serve.exe.run_service --task-queue robotics-hand-keypoints-temporal`).

---

## Implementation Plan

### Phase 1 — MVP (fixed replicas, ~2 weeks)

**`domain/entities/model_endpoint_entity.py`**
```python
class ModelEndpointType(str, Enum):
    ASYNC = "async"
    SYNC = "sync"
    STREAMING = "streaming"
    TEMPORAL = "temporal"          # new
```

**`common/dtos/model_endpoints.py`**
- Add `temporal_task_queue: Optional[str]` to `CreateModelEndpointV1Request` and `UpdateModelEndpointV1Request`
- Validation: required when `endpoint_type == "temporal"`

**`domain/use_cases/model_endpoint_use_cases.py`**
- `validate_deployment_resources`: allow `min_workers=0` for `TEMPORAL` (same as `ASYNC`)
- No `concurrent_requests_per_worker` limit (workers process one activity at a time by default)

**`infra/gateways/resources/k8s_resource_types.py`**
- Add `_TemporalDeploymentArguments` TypedDict:
  ```python
  class _TemporalDeploymentArguments(TypedDict):
      TEMPORAL_TASK_QUEUE: str
      TEMPORAL_SERVER_HOSTNAME: str
      TEMPORAL_SERVER_PORT: str
      REPLICAS: int   # fixed in MVP; driven by max_workers
  ```
- Add `DeploymentRunnableImageTemporalGpuArguments` and `...CpuArguments` composite TypedDicts

**`infra/gateways/resources/templates/service_template_config_map*.yaml`**
- Add `deployment-runnable-image-temporal-gpu.yaml` and `...-cpu.yaml` templates
- Key differences from async template:
  - No `celery-forwarder` sidecar — the user container IS the Temporal worker
  - No `celery.scaleml.autoscaler/*` annotations
  - `replicas: ${REPLICAS}` (fixed)
  - Env vars injected: `TEMPORAL_TASK_QUEUE`, `TEMPORAL_SERVER_HOSTNAME`, `TEMPORAL_SERVER_PORT`, `CONCURRENCY`
  - Readiness probe: TCP check on Temporal worker port (or omit — workers have no HTTP endpoint)

**`infra/gateways/resources/k8s_endpoint_resource_delegate.py`**
- `delete_resources`: add `TEMPORAL` branch (reuses sync cleanup — no Celery queue to delete)
- `create_or_update_resources`: route `TEMPORAL` to new template

**`infra/gateways/resources/live_endpoint_resource_gateway.py`**
- `create_or_update_resources`: skip Celery queue creation for `TEMPORAL` (add to `else` branch or make explicit)
- `get_resources`: skip SQS queue depth polling for `TEMPORAL`

### Phase 2 — Temporal-aware autoscaling (follow-up, ~3 weeks)

Scale worker replicas based on Temporal task queue backlog. Options:

1. **KEDA `temporal` trigger** — KEDA has a [Temporal scaler](https://keda.sh/docs/scalers/temporal/) that polls `GetTaskQueueStats`. Lowest implementation cost; reuses existing KEDA infrastructure.
2. **Custom autoscaler** — mirrors the existing Celery autoscaler pattern but polls Temporal's gRPC API.

Recommendation: KEDA Temporal trigger. Annotation format:
```yaml
temporal.keda.sh/task-queue: "${TEMPORAL_TASK_QUEUE}"
temporal.keda.sh/namespace: "default"
temporal.keda.sh/targetQueueSize: "${PER_WORKER}"
```

---

## What Changes in Caller Code (ego example)

Before — each service writes ~80 lines of custom Temporal boilerplate:
```python
# launch_hand_keypoints/temporal_worker.py (custom, per-service)
_predict = load_predict_fn(...)

@activity.defn(name="handKeypointsActivity")
async def hand_keypoints_activity(inp):
    heartbeat_task = loop.create_task(_heartbeat_loop())   # manual
    ...

async def main():
    client = await Client.connect(...)                     # manual
    worker = Worker(client, task_queue=..., ...)           # manual
    await worker.run()
```

After — service implements one method; Launch manages the rest:
```python
# launch_hand_keypoints/service.py
class HandKeypointsService(ModelServiceApi):
    def handle(self, req: dict) -> dict:
        result = self._predict(HandKeypointsRequest(**req))
        return {"hands_npz_url": ..., "track_info_url": ...}
```

```bash
# Deploy via Launch API (same as any other endpoint)
launch create-endpoint \
  --name hand-keypoints-temporal \
  --endpoint-type temporal \
  --temporal-task-queue robotics-hand-keypoints-temporal \
  --gpu-type h100 --gpus 1 \
  --min-workers 0 --max-workers 10
```

---

## What This Is Not

- **Not a task submission API.** Launch does not expose `/v1/async-tasks` for `temporal` endpoints. The Temporal workflow is the caller; Launch only manages the worker pods.
- **Not a workflow worker.** Launch manages activity workers only. The workflow definition lives in the caller's codebase and runs on a separate workflow worker (or Temporal Cloud).
- **Not a replacement for Celery async endpoints.** `async` endpoints remain the right choice for request/response workloads where the caller submits tasks via Launch's API. `temporal` is for multi-step pipelines where an external orchestrator coordinates the work.

---

## Alternatives Considered

| Option | Verdict |
|--------|---------|
| Raw K8s Deployment (std-ml-srv `deployment_template_TEMPORAL_gpu.yaml`) | Works today; loses Launch management. Good stopgap, not long-term. |
| Temporal orchestrates existing Launch async endpoints | Extra HTTP round-trip per phase; doesn't use Temporal activities properly. |
| Launch batch jobs per phase | `backoffLimit: 0`, cold start per request, no worker pool. Wrong tool. |
| Temporal Cloud | Doesn't change the worker management problem; workers still need to run somewhere. |

---

## Open Questions

1. **Readiness probe**: Temporal workers have no HTTP endpoint. Should Launch skip the readiness probe for `temporal` endpoints, or should worker images expose a `/healthz` on a sidecar port?
2. **Task submission API**: Should Launch eventually expose a way to *start a Temporal workflow* (not just manage workers)? This would be a larger API addition and is out of scope for Phase 1.
3. **Namespace**: Should `temporal_namespace` be a configurable field, or default to `"default"`?
4. **Multi-queue workers**: Some use cases may want one pod to pull from multiple task queues. Out of scope for now; each endpoint maps to one task queue.
