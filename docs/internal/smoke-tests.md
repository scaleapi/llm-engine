# Smoke Tests

Post-deploy validation checklist for Model Engine. Run these checks after every `helm install` or `helm upgrade` to verify the deployment is working correctly.

**Last verified:** [DATE]
**Verified by:** [AUTHOR]

---

## Test Tiers

Two tiers are defined to allow partial validation when GPU nodes are unavailable:

| Tier | Hardware | Model | What it covers |
|---|---|---|---|
| **Tier A — CPU only** | Any node, no GPU required | `model_engine_server.inference.forwarding.echo_server` (built into the model-engine image) | Service Builder flow, endpoint lifecycle, sync inference, async inference |
| **Tier B — GPU + LLM** | GPU node with NVIDIA driver | `llama-3.1-8b` via vLLM | vLLM image pull from customer registry, GPU scheduling, `POST /v2/chat/completions` sync and streaming |

Run **Tier A** first. If Tier A passes, the control plane, broker, database, and Service Builder are all functioning. Tier B additionally validates GPU scheduling and the LLM API layer.

!!! warning "v2 API only"
    Tier B only tests `POST /v2/chat/completions`. Do not use v1 completions endpoints in these smoke tests.

---

## Required Environment Variables

Set these before running any commands in this document:

```bash
export GATEWAY_URL="http://model-engine.{NAMESPACE}.svc.cluster.local"  # k8s cluster DNS
export AUTH_TOKEN="{your-auth-token}"
export NAMESPACE="{your-namespace}"
```

---

## Phase 1: Pre-flight

Run these checks **before** `helm install`.

### 1.1 Redis reachability

- [ ] Redis responds to ping:

```bash
redis-cli -h {REDIS_HOST} -p {REDIS_PORT} ping
# Expected: PONG
```

### 1.2 Database reachability

- [ ] Database accepts connections:

```bash
psql "{DB_URL}" -c "SELECT 1"
# Expected: returns 1
```

### 1.3 vLLM image pullable from customer registry

!!! warning
    Verify the image is accessible from **your mirrored registry**, not Scale's internal ECR. This is the #1 silent deployment failure.

- [ ] vLLM image is pullable:

```bash
kubectl run vllm-pull-test \
  --image={VLLM_REPOSITORY}:{VLLM_TAG} \
  --restart=Never \
  --command -- echo "pull ok" \
  -n {NAMESPACE}
kubectl logs vllm-pull-test -n {NAMESPACE}
kubectl delete pod vllm-pull-test -n {NAMESPACE}
# Expected: "pull ok"
```

### 1.4 GPU driver functional on GPU nodes

- [ ] NVIDIA driver is working on GPU nodes:

```bash
# List GPU nodes
kubectl get nodes -l k8s.amazonaws.com/accelerator={GPU_TYPE}

# Run nvidia-smi on a GPU node
kubectl debug node/{GPU_NODE_NAME} \
  -it \
  --image=nvidia/cuda:12.0-base \
  -- nvidia-smi
# Expected: GPU device listed with driver version and CUDA version
```

### 1.5 GPU nodes labeled correctly

- [ ] Nodes carry the expected accelerator label:

```bash
kubectl get nodes --show-labels | grep accelerator
# Expected: nodes labeled with k8s.amazonaws.com/accelerator={GPU_TYPE}
```

### 1.6 Broker reachability

=== "AWS (SQS)"
    - [ ] IAM/IRSA SQS access works from the service account:
    ```bash
    kubectl run sqs-test \
      --image=amazon/aws-cli \
      --restart=Never \
      --serviceaccount={MODEL_ENGINE_SERVICE_ACCOUNT} \
      -n {NAMESPACE} \
      -- sqs list-queues --region {AWS_REGION}
    kubectl logs sqs-test -n {NAMESPACE}
    kubectl delete pod sqs-test -n {NAMESPACE}
    ```

=== "Azure (Service Bus)"
    - [ ] Service Bus secret exists:
    ```bash
    kubectl get secret {ASB_SECRET_NAME} -n {NAMESPACE}
    ```
    !!! warning
        Azure Service Bus drops idle AMQP connections after 300 seconds. Ensure `broker_pool_limit=0` is set in helm values. See [Cloud Support Matrix](cloud-matrix.md#azure).

=== "GCP / On-prem (Redis)"
    - [ ] Redis broker reachable:
    ```bash
    redis-cli -h {BROKER_REDIS_HOST} -p {BROKER_REDIS_PORT} ping
    # Expected: PONG
    ```

---

## Phase 2: Post-install Infrastructure

Run these checks immediately after `helm install` completes.

### 2.1 All pods Running and Ready

- [ ] All model-engine pods are `Running`:

```bash
kubectl get pods -n {NAMESPACE} -l app=model-engine-gateway
kubectl get pods -n {NAMESPACE} -l app=model-engine-builder
kubectl get pods -n {NAMESPACE} -l app=model-engine-cacher
kubectl get pods -n {NAMESPACE} -l app=model-engine-celery-autoscaler
```

- [ ] No pods in error states:

```bash
kubectl get pods -n {NAMESPACE} | grep -vE "Running|Completed|NAME"
# Expected: no output
```

### 2.2 Gateway HPA exists

- [ ] HPA is configured for the gateway:

```bash
kubectl get hpa -n {NAMESPACE}
# Expected: model-engine-gateway HPA present
```

### 2.3 Celery Autoscaler StatefulSet ready

- [ ] StatefulSet shows desired replicas ready:

```bash
kubectl get statefulset -n {NAMESPACE} -l app=model-engine-celery-autoscaler
# Expected: READY column matches DESIRED (e.g. 1/1)
```

### 2.4 Cacher log check

- [ ] Cacher is completing cache cycles (should appear within 15 seconds):

```bash
kubectl logs -n {NAMESPACE} -l app=model-engine-cacher --tail=50 | grep -i "cache"
```

- [ ] No Redis auth errors:

```bash
kubectl logs -n {NAMESPACE} -l app=model-engine-cacher --tail=100 \
  | grep -iE "auth|error|exception|redis"
# Expected: no authentication errors
```

### 2.5 Builder Celery worker ready

- [ ] Builder shows Celery worker connected and ready:

```bash
kubectl logs -n {NAMESPACE} -l app=model-engine-builder --tail=100 \
  | grep -iE "ready|celery|connected|worker"
```

- [ ] No broker connection errors:

```bash
kubectl logs -n {NAMESPACE} -l app=model-engine-builder --tail=100 \
  | grep -iE "error|refused|timeout|cannot connect"
# Expected: no output
```

### 2.6 ConfigMap non-empty

- [ ] `model-engine-config` ConfigMap exists and has a populated data section:

```bash
kubectl get configmap model-engine-config -n {NAMESPACE} -o yaml
# Expected: data section is non-empty
```

- [ ] Embedded kubeconfig has valid clusters and users:

```bash
kubectl get configmap model-engine-config -n {NAMESPACE} \
  -o jsonpath='{.data.kubeconfig}' \
  | python3 -c "
import sys, yaml
c = yaml.safe_load(sys.stdin)
print('clusters:', len(c.get('clusters', [])), 'users:', len(c.get('users', [])))
"
# Expected: clusters: 1 users: 1
```

---

## Phase 3: Control Plane

Commands use k8s cluster DNS. To run from outside the cluster, port-forward first:

```bash
kubectl port-forward svc/model-engine 8080:80 -n {NAMESPACE}
# Then set GATEWAY_URL=http://localhost:8080
```

### 3.1 Gateway health check

- [ ] Gateway returns 200:

```bash
curl -sf "{GATEWAY_URL}/healthz" && echo "OK"
# Expected: HTTP 200
```

### 3.2 Authenticated list endpoints

- [ ] Returns 200 with valid list (empty is fine):

```bash
curl -sf \
  -H "Authorization: Bearer {AUTH_TOKEN}" \
  "{GATEWAY_URL}/v1/model-endpoints"
# Expected: HTTP 200, body: {"model_endpoints": []}
```

---

## Phase 4: Endpoint Lifecycle — Tier A (CPU Only)

The echo server (`model_engine_server.inference.forwarding.echo_server`) is built into the model-engine image. It echoes request bodies unchanged — no GPU or model weights required.

### 4.1 Create sync CPU echo endpoint

- [ ] Create endpoint:

```bash
curl -sf -X POST \
  -H "Authorization: Bearer {AUTH_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "smoke-test-echo-sync",
    "model_bundle": {
      "flavor": "streaming_enhanced_runnable_image",
      "repository": "{MODEL_ENGINE_IMAGE_REPOSITORY}",
      "tag": "{MODEL_ENGINE_IMAGE_TAG}",
      "command": [
        "python", "-m",
        "model_engine_server.inference.forwarding.echo_server",
        "--port", "5005"
      ],
      "streaming_command": [
        "python", "-m",
        "model_engine_server.inference.forwarding.echo_server",
        "--port", "5005"
      ],
      "env": {
        "HTTP_HOST": "0.0.0.0",
        "ML_INFRA_SERVICES_CONFIG_PATH": "/workspace/model-engine/model_engine_server/core/configs/default.yaml"
      },
      "protocol": "http",
      "readiness_initial_delay_seconds": 20
    },
    "endpoint_type": "sync",
    "cpus": "1",
    "memory": "1Gi",
    "storage": "2Gi",
    "gpus": 0,
    "min_workers": 1,
    "max_workers": 1,
    "per_worker": 1,
    "labels": {"team": "infra", "product": "smoke-test"},
    "metadata": {}
  }' \
  "{GATEWAY_URL}/v1/model-endpoints"
# Expected: HTTP 200, {"endpoint_creation_task_id": "..."}
```

### 4.2 Poll until READY

- [ ] Wait for status `READY` (typically 3–8 minutes for CPU):

```bash
for i in $(seq 1 50); do
  STATUS=$(curl -sf \
    -H "Authorization: Bearer {AUTH_TOKEN}" \
    "{GATEWAY_URL}/v1/model-endpoints?name=smoke-test-echo-sync" \
    | python3 -c "
import sys, json
eps = json.load(sys.stdin)['model_endpoints']
print(eps[0]['status'] if eps else 'NOT_FOUND')
")
  echo "[$(date)] Status: $STATUS"
  [ "$STATUS" = "READY" ] && break
  sleep 30
done
```

!!! warning "Stuck INITIALIZING >15 min"
    Service Builder cannot reach the broker. Check builder logs:
    `kubectl logs -n {NAMESPACE} -l app=model-engine-builder --tail=200`

!!! warning "Status unknown"
    Redis auth is broken in the cacher. The endpoint may be running but the Gateway cannot read its state. Check cacher logs:
    `kubectl logs -n {NAMESPACE} -l app=model-engine-cacher --tail=200`

### 4.3 Send echo request and verify response

- [ ] Get endpoint ID and send request:

```bash
ENDPOINT_ID=$(curl -sf \
  -H "Authorization: Bearer {AUTH_TOKEN}" \
  "{GATEWAY_URL}/v1/model-endpoints?name=smoke-test-echo-sync" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['model_endpoints'][0]['id'])")

curl -sf -X POST \
  -H "Authorization: Bearer {AUTH_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"args": {"hello": "world"}}' \
  "{GATEWAY_URL}/v1/sync-tasks?model_endpoint_id=$ENDPOINT_ID"
# Expected: {"status": "SUCCESS", "result": {"hello": "world"}, ...}
```

### 4.4 Create async CPU echo endpoint

- [ ] Create async endpoint (same payload, `endpoint_type: async`):

```bash
curl -sf -X POST \
  -H "Authorization: Bearer {AUTH_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "smoke-test-echo-async",
    "model_bundle": {
      "flavor": "streaming_enhanced_runnable_image",
      "repository": "{MODEL_ENGINE_IMAGE_REPOSITORY}",
      "tag": "{MODEL_ENGINE_IMAGE_TAG}",
      "command": [
        "python", "-m",
        "model_engine_server.inference.forwarding.echo_server",
        "--port", "5005"
      ],
      "streaming_command": [
        "python", "-m",
        "model_engine_server.inference.forwarding.echo_server",
        "--port", "5005"
      ],
      "env": {
        "HTTP_HOST": "0.0.0.0",
        "ML_INFRA_SERVICES_CONFIG_PATH": "/workspace/model-engine/model_engine_server/core/configs/default.yaml"
      },
      "protocol": "http",
      "readiness_initial_delay_seconds": 20
    },
    "endpoint_type": "async",
    "cpus": "1",
    "memory": "1Gi",
    "storage": "2Gi",
    "gpus": 0,
    "min_workers": 1,
    "max_workers": 1,
    "per_worker": 1,
    "labels": {"team": "infra", "product": "smoke-test"},
    "metadata": {}
  }' \
  "{GATEWAY_URL}/v1/model-endpoints"
```

- [ ] Poll until `smoke-test-echo-async` is `READY` (same pattern as 4.2).

### 4.5 Send async request and poll for SUCCESS

- [ ] Submit task and poll:

```bash
ASYNC_ENDPOINT_ID=$(curl -sf \
  -H "Authorization: Bearer {AUTH_TOKEN}" \
  "{GATEWAY_URL}/v1/model-endpoints?name=smoke-test-echo-async" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['model_endpoints'][0]['id'])")

TASK_ID=$(curl -sf -X POST \
  -H "Authorization: Bearer {AUTH_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"args": {"y": 1}, "url": null}' \
  "{GATEWAY_URL}/v1/async-tasks?model_endpoint_id=$ASYNC_ENDPOINT_ID" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['task_id'])")
echo "Task ID: $TASK_ID"

for i in $(seq 1 30); do
  RESULT=$(curl -sf \
    -H "Authorization: Bearer {AUTH_TOKEN}" \
    "{GATEWAY_URL}/v1/async-tasks/$TASK_ID")
  STATUS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "[$(date)] Task status: $STATUS"
  [ "$STATUS" = "SUCCESS" ] && echo "Result: $RESULT" && break
  [ "$STATUS" = "FAILURE" ] && echo "Task failed: $RESULT" && break
  sleep 10
done
# Expected final status: SUCCESS
```

### 4.6 Cleanup — Tier A

- [ ] Delete both endpoints:

```bash
for NAME in smoke-test-echo-sync smoke-test-echo-async; do
  ID=$(curl -sf \
    -H "Authorization: Bearer {AUTH_TOKEN}" \
    "{GATEWAY_URL}/v1/model-endpoints?name=$NAME" \
    | python3 -c "import sys,json; eps=json.load(sys.stdin)['model_endpoints']; print(eps[0]['id'] if eps else '')")
  [ -n "$ID" ] && curl -sf -X DELETE \
    -H "Authorization: Bearer {AUTH_TOKEN}" \
    "{GATEWAY_URL}/v1/model-endpoints/$ID"
  echo "Deleted $NAME"
done
# Expected: {"deleted": true} for each
```

---

## Phase 5: LLM Inference — Tier B (GPU + LLM)

!!! warning "GPU required"
    Complete Phase 1 checks 1.3–1.5 before proceeding. GPU nodes must be available and the vLLM image must be mirrored to the customer registry.

### 5.1 Create llama-3.1-8b LLM endpoint

- [ ] Create endpoint with `min_workers=1`:

```bash
curl -sf -X POST \
  -H "Authorization: Bearer {AUTH_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "smoke-test-llama-3-1-8b",
    "model_name": "llama-3.1-8b",
    "source": "hugging_face",
    "inference_framework": "vllm",
    "inference_framework_image_tag": "{VLLM_TAG}",
    "endpoint_type": "streaming",
    "cpus": 20,
    "gpus": 1,
    "gpu_type": "{GPU_TYPE}",
    "memory": "20Gi",
    "storage": "40Gi",
    "optimize_costs": false,
    "min_workers": 1,
    "max_workers": 1,
    "per_worker": 1,
    "labels": {"team": "infra", "product": "smoke-test"},
    "metadata": {},
    "public_inference": false
  }' \
  "{GATEWAY_URL}/v1/llm/model-endpoints"
# Expected: HTTP 200
```

### 5.2 Poll until READY

- [ ] Allow up to 30 minutes for GPU image pull and model load:

```bash
for i in $(seq 1 60); do
  STATUS=$(curl -sf \
    -H "Authorization: Bearer {AUTH_TOKEN}" \
    "{GATEWAY_URL}/v1/llm/model-endpoints/smoke-test-llama-3-1-8b" \
    | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','NOT_FOUND'))")
  echo "[$(date)] Status: $STATUS"
  [ "$STATUS" = "READY" ] && break
  sleep 30
done
```

!!! warning "Status unknown"
    Redis authentication is broken in the K8s Cacher. Check cacher logs immediately:
    ```bash
    kubectl logs -n {NAMESPACE} -l app=model-engine-cacher --tail=100 \
      | grep -iE "auth|redis|error"
    ```

!!! warning "Stuck INITIALIZING — vLLM pod Pending"
    Check whether the vLLM pod has a scheduling or image pull issue:
    ```bash
    kubectl get pods -n {ENDPOINT_NAMESPACE} | grep smoke-test-llama
    kubectl describe pod {VLLM_POD_NAME} -n {ENDPOINT_NAMESPACE} | grep -A 10 Events
    ```

### 5.3 POST /v2/chat/completions — sync

- [ ] Send sync chat completions request and verify non-empty response:

```bash
curl -sf -X POST \
  -H "Authorization: Bearer {AUTH_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "smoke-test-llama-3-1-8b",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 50,
    "temperature": 0.0
  }' \
  "{GATEWAY_URL}/v2/chat/completions" \
  | python3 -c "
import sys, json
r = json.load(sys.stdin)
content = r['choices'][0]['message']['content']
assert content, 'Empty response content'
print('Response:', content)
"
# Expected: non-empty assistant message
```

### 5.4 POST /v2/chat/completions — streaming (SSE)

- [ ] Verify SSE chunks arrive:

```bash
curl -sf -X POST \
  -H "Authorization: Bearer {AUTH_TOKEN}" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "smoke-test-llama-3-1-8b",
    "messages": [{"role": "user", "content": "Count from one to five."}],
    "max_tokens": 50,
    "temperature": 0.0,
    "stream": true
  }' \
  "{GATEWAY_URL}/v2/chat/completions" \
  | grep "^data:" | grep -v "\[DONE\]" | head -5
# Expected: one or more lines of the form:
#   data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":"..."}}]}
```

### 5.5 Cleanup — Tier B

- [ ] Delete the LLM endpoint:

```bash
curl -sf -X DELETE \
  -H "Authorization: Bearer {AUTH_TOKEN}" \
  "{GATEWAY_URL}/v1/llm/model-endpoints/smoke-test-llama-3-1-8b"
# Expected: {"deleted": true}
```

- [ ] Verify GPU worker pods are terminated:

```bash
kubectl get pods -n {ENDPOINT_NAMESPACE} | grep smoke-test-llama
# Expected: no output
```

---

## Common Failure Signatures

Seeded from real deployment incidents.

| Symptom | Likely cause | Where to look |
|---|---|---|
| Endpoint stuck `INITIALIZING` >15 min | Service Builder cannot reach message broker | `kubectl logs -n {NAMESPACE} -l app=model-engine-builder --tail=200` |
| Endpoint status `unknown` | Redis auth failure in K8s Cacher — Gateway cannot read endpoint state | `kubectl logs -n {NAMESPACE} -l app=model-engine-cacher --tail=200` — look for `AUTH` or `NOAUTH` errors |
| `ImagePullBackOff` on vLLM pods | `vllm_repository` points to Scale's internal ECR; image not mirrored | `kubectl describe pod {POD} -n {NS}` Events; check `vllm_repository` in helm values |
| GPU worker pods `Pending` indefinitely | NVIDIA driver not initialized, driver image not pullable, or no matching accelerator label | `kubectl describe pod {POD}` Events; `kubectl debug node/{NODE} -- nvidia-smi` |
| Random 503s on inference after idle period | Azure Service Bus drops idle AMQP connections after 300s | Gateway logs for `503`/`timeout`; verify `broker_pool_limit=0` in helm values |
| Permission errors / empty kubeconfig on endpoint creation | `model-engine-config` ConfigMap has empty kubeconfig due to race condition at startup | `kubectl get configmap model-engine-config -n {NAMESPACE} -o yaml` — check `kubeconfig` key |
| `GET /v1/model-endpoints` returns 500 | DB unreachable or schema not migrated (`db.runDbMigrationScript: false` on first install) | Gateway pod logs — look for `psycopg2` or `sqlalchemy` errors |
| Builder logs show `KombuError` or `OperationalError` | `celeryBrokerType` mismatch or broker credentials wrong | Check `celeryBrokerType` in helm values matches deployed broker |
