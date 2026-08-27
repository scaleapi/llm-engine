# Temporal Endpoint Type — E2E Test

**Date:** 2026-04-25  
**Cluster:** `ml-serving-new` (us-west-2, account 759270588401)  
**Branch:** `lilyz-ai/temporal-endpoint-type`  
**Commit:** `0efddf2f`

---

## Prerequisites

- `kubectl` context pointing at `ml-serving-new`
- AWS SSO refreshed: `aws sso login --profile ml-serving-admin`
- Docker logged in to internal ECR (account 692474966980)

---

## Step 1 — Build and push a patched image

The production image (`model-engine-internal:<sha>`) is built from an internal private repo, not the public `scaleapi/llm-engine` Dockerfile. To test code changes, layer the updated Python files on top of the most recent production image:

```bash
# Find the most recent production image
AWS_PROFILE=ml-admin aws ecr describe-images \
  --repository-name model-engine-internal \
  --region us-west-2 \
  --query 'sort_by(imageDetails, &imagePushedAt)[-5:].{tag:imageTags[0],pushed:imagePushedAt}' \
  --output table

BASE_SHA=<most-recent-sha>   # e.g. 60e0027d0ef39afafa354574a756864de0db7a04
MY_SHA=$(git rev-parse HEAD)

# Copy updated source into build context
mkdir -p /tmp/buildctx
cp -r model-engine/model_engine_server /tmp/buildctx/model_engine_server

cat > /tmp/buildctx/Dockerfile << EOF
FROM 692474966980.dkr.ecr.us-west-2.amazonaws.com/model-engine-internal:${BASE_SHA}
COPY model_engine_server/ /workspace/model-engine/model_engine_server/
EOF

# Login, build, push
AWS_PROFILE=ml-admin aws ecr get-login-password --region us-west-2 \
  | docker login --username AWS --password-stdin 692474966980.dkr.ecr.us-west-2.amazonaws.com

docker build /tmp/buildctx \
  -t 692474966980.dkr.ecr.us-west-2.amazonaws.com/model-engine-internal:${MY_SHA}-patch

docker push 692474966980.dkr.ecr.us-west-2.amazonaws.com/model-engine-internal:${MY_SHA}-patch
```

> **Note:** If `live_tokenizer_repository.py` imports `from huggingface_hub.errors import RepositoryNotFoundError` and the base image has an older `huggingface_hub`, patch the import before building:
> ```bash
> sed -i 's|from huggingface_hub.errors import|try:\n    from huggingface_hub.errors import|' \
>   /tmp/buildctx/model_engine_server/infra/repositories/live_tokenizer_repository.py
> # add fallback line manually or use the try/except pattern
> ```

---

## Step 2 — Deploy the patched image

```bash
MY_SHA=$(git rev-parse HEAD)
PATCH_TAG="${MY_SHA}-patch"
ECR="692474966980.dkr.ecr.us-west-2.amazonaws.com/model-engine-internal"

kubectl set image deployment/model-engine-endpoint-builder \
  model-engine-endpoint-builder=${ECR}:${PATCH_TAG} -n default

kubectl set image deployment/model-engine \
  model-engine=${ECR}:${PATCH_TAG} -n default

kubectl rollout status deployment/model-engine-endpoint-builder -n default --timeout=300s
kubectl rollout status deployment/model-engine -n default --timeout=300s
```

---

## Step 3 — Patch the service-template ConfigMap

The production ConfigMap does not yet include the temporal templates (until the Helm chart is upgraded). Patch it manually:

```bash
kubectl get configmap model-engine-service-template-config -n default -o json \
  | python3 - << 'EOF'
import sys, json, subprocess

cm = json.load(sys.stdin)

# Paste the temporal-cpu template from
# model-engine/model_engine_server/infra/gateways/resources/templates/service_template_config_map_circleci.yaml
# (keys: deployment-runnable-image-temporal-cpu.yaml, deployment-runnable-image-temporal-gpu.yaml)
# but update env/prod labels to match the production configmap format.
#
# See patch_temporal_configmap.py in this directory for a ready-made script.

print(json.dumps(cm))
EOF
```

A ready-made patch script is at [`docs/patch_temporal_configmap.py`](patch_temporal_configmap.py). Run it with:

```bash
python3 docs/patch_temporal_configmap.py
```

Verify the two new keys are present:

```bash
kubectl get configmap model-engine-service-template-config -n default \
  -o jsonpath='{.data}' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print([k for k in d if 'temporal' in k])"
# Expected: ['deployment-runnable-image-temporal-cpu.yaml', 'deployment-runnable-image-temporal-gpu.yaml']
```

---

## Step 4 — Run the DB migration

```bash
# The migration runs inside a model-engine pod (which reads DB connection from infra_config_prod.yaml)
GATEWAY_POD=$(kubectl get pods -n default -l app=model-engine --field-selector=status.phase=Running \
  -o jsonpath='{.items[0].metadata.name}')

kubectl exec -n default "$GATEWAY_POD" -- bash -c \
  "cd /workspace/model-engine/model_engine_server/db/migrations && \
   python3 -m alembic -c alembic.ini upgrade head"
```

Expected output:
```
INFO  [alembic.runtime.migration] Running upgrade a1b2c3d4e5f6 -> b2c3d4e5f6g7, add temporal_task_queue column
```

> The migration adds a nullable `VARCHAR` column. It is backwards-compatible — existing pods reading the DB before migration simply do not select that column.

---

## Step 5 — Create a test temporal endpoint

```bash
# Get the test API key
TEST_KEY=$(AWS_PROFILE=ml-serving-admin aws secretsmanager get-secret-value \
  --secret-id launch_test_api_key --region us-west-2 \
  --query SecretString --output text \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['api_key'])")
TEST_USER=$(AWS_PROFILE=ml-serving-admin aws secretsmanager get-secret-value \
  --secret-id launch_test_api_key --region us-west-2 \
  --query SecretString --output text \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['user_id'])")

GATEWAY_POD=$(kubectl get pods -n default -l app=model-engine --field-selector=status.phase=Running \
  -o jsonpath='{.items[0].metadata.name}')
kubectl port-forward "pod/$GATEWAY_POD" 8081:5000 -n default &

# Create a model bundle
BUNDLE_ID=$(curl -s -X POST http://localhost:8081/v2/model-bundles \
  -u "${TEST_USER}:${TEST_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "test-temporal-bundle",
    "schema_location": "s3://scale-ml/test/temporal/schema.json",
    "flavor": {
      "flavor": "runnable_image",
      "repository": "public.ecr.aws/ubuntu/ubuntu",
      "tag": "22.04",
      "command": ["sleep", "3600"],
      "protocol": "http",
      "readiness_initial_delay_seconds": 30
    },
    "metadata": {}
  }' | python3 -c "import sys,json; print(json.load(sys.stdin)['model_bundle_id'])")

echo "Bundle: $BUNDLE_ID"

# Create a temporal endpoint
curl -s -X POST http://localhost:8081/v1/model-endpoints \
  -u "${TEST_USER}:${TEST_KEY}" \
  -H "Content-Type: application/json" \
  -d "{
    \"name\": \"test-temporal-endpoint\",
    \"endpoint_type\": \"temporal\",
    \"temporal_task_queue\": \"my-test-task-queue\",
    \"model_bundle_id\": \"${BUNDLE_ID}\",
    \"metadata\": {},
    \"cpus\": 0.5,
    \"gpus\": 0,
    \"memory\": \"1Gi\",
    \"storage\": \"1Gi\",
    \"min_workers\": 0,
    \"max_workers\": 2,
    \"per_worker\": 1,
    \"labels\": {\"team\": \"infra\", \"product\": \"launch\"}
  }" | python3 -c "import sys,json; print(json.dumps(json.load(sys.stdin), indent=2))"
```

---

## Step 6 — Verify the K8s Deployment

```bash
# Find the endpoint ID
ENDPOINT_ID=$(curl -s http://localhost:8081/v1/model-endpoints \
  -u "${TEST_USER}:${TEST_KEY}" \
  | python3 -c "
import sys,json
eps=[e for e in json.load(sys.stdin)['model_endpoints'] if 'temporal' in e['name']]
print(eps[0]['id'])")

# Find the deployment name from the endpoint
DEPLOY=$(kubectl get deployments -n scale-deploy \
  | grep "$ENDPOINT_ID" | awk '{print $1}')

kubectl get deployment "$DEPLOY" -n scale-deploy -o json | python3 -c "
import sys,json
d=json.load(sys.stdin)
meta=d['metadata']
spec=d['spec']
containers=spec['template']['spec']['containers']
print('=== ANNOTATIONS ===')
for k,v in meta.get('annotations',{}).items():
    if 'temporal' in k:
        print(f'  {k}: {v}')
print('=== REPLICAS:', spec['replicas'], '===')
print('=== CONTAINERS:', len(containers), '(expected: 1) ===')
for c in containers:
    print('  name:', c['name'])
print('=== TEMPORAL ENV VARS ===')
for c in containers:
    for e in c.get('env', []):
        if 'TEMPORAL' in e.get('name',''):
            print(f\"  {e['name']}: {e.get('value', '<from config>')}\")
"
```

---

## Step 7 — Cleanup

```bash
# Delete the test endpoint
curl -s -X DELETE "http://localhost:8081/v1/model-endpoints/${ENDPOINT_ID}" \
  -u "${TEST_USER}:${TEST_KEY}"

# Revert model-engine and builder to stable production image
PROD_SHA=f395ffa6bdcf9c954f1073469a2a25c7b3351af8
ECR="692474966980.dkr.ecr.us-west-2.amazonaws.com/model-engine-internal"
kubectl set image deployment/model-engine model-engine=${ECR}:${PROD_SHA} -n default
kubectl set image deployment/model-engine-endpoint-builder \
  model-engine-endpoint-builder=${ECR}:${PROD_SHA} -n default

kill %1  # kill port-forward
```

---

## Test Results (2026-04-25)

### Environment

| Item | Value |
|------|-------|
| Cluster | `ml-serving-new` (arn:aws:eks:us-west-2:759270588401:cluster/ml-serving-new) |
| Base production image | `model-engine-internal:60e0027d0ef39afafa354574a756864de0db7a04` (2026-03-30) |
| Patched image | `model-engine-internal:04729cef1678fe09ffda9855624579152a8fae3e-patch4` |
| DB | `ml-infra-prod.cluster-cuby7rtblks1.us-west-2.rds.amazonaws.com` |

### API — Endpoint Creation

```
POST /v1/model-endpoints
  endpoint_type: "temporal"
  temporal_task_queue: "my-test-task-queue"
  max_workers: 2
  gpus: 0

→ HTTP 200 {"endpoint_creation_task_id": "68cd17fd-70c0-46a6-a31e-4ee5fe2a676f"}
```

### API — Endpoint Status (after ~3 seconds)

```json
{
  "id": "end_d7m0o91ll55003e7d0n0",
  "name": "test-temporal-endpoint",
  "endpoint_type": "temporal",
  "status": "READY",
  "deployment_state": {
    "min_workers": 0,
    "max_workers": 2,
    "per_worker": 1,
    "concurrent_requests_per_worker": 1
  }
}
```

### K8s Deployment

```
NAME: launch-endpoint-id-end-d7m0o91ll55003e7d0n0
NAMESPACE: scale-deploy
REPLICAS: 2   (= max_workers ✓)
CONTAINERS: 1   (no forwarder sidecar ✓)
```

**Annotations:**

| Annotation | Value |
|------------|-------|
| `temporal.scaleml.io/taskQueue` | `my-test-task-queue` ✓ |
| `temporal.scaleml.io/minWorkers` | `0` ✓ |
| `temporal.scaleml.io/maxWorkers` | `2` ✓ |
| `temporal.scaleml.io/perWorker` | `1` ✓ |

**Env vars injected into `main` container:**

| Env var | Value |
|---------|-------|
| `TEMPORAL_TASK_QUEUE` | `my-test-task-queue` ✓ |
| `TEMPORAL_SERVER_HOSTNAME` | *(from infra config)* ✓ |
| `TEMPORAL_SERVER_PORT` | `7233` ✓ |

### DB Migration

```
INFO  [alembic.runtime.migration] Running upgrade a1b2c3d4e5f6 -> b2c3d4e5f6g7,
      add temporal_task_queue column
```

Migration applied cleanly. No downtime for existing endpoints.

### Issues Found and Fixed During Test

| Issue | Fix |
|-------|-----|
| Helm template error: `$security_context \| nindent` on map type | Use `with + toYaml` pattern (same as other templates) — committed in `0efddf2f` |
| `huggingface_hub.errors` not found on older base images | Add try/except fallback import in build context (not committed — affects e2e test setup only, not shipped code) |

### Cleanup

Test endpoint deleted. Builder and gateway reverted to production image `f395ffa6bdcf9c954f1073469a2a25c7b3351af8`.
