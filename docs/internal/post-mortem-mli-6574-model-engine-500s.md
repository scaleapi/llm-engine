# SEV1 Post-Mortem: model-engine Deployment Caused 500s on All Endpoint Operations

**Incident ID:** MLI-6574
**Severity:** SEV1
**Date:** Apr 24–25, 2026
**Duration:** ~58 min (23:50Z deployment → 00:48Z rollback)
**Customer-facing 500s:** ~47 min
**Status:** Resolved; follow-up actions tracked below

---

## Summary

At 23:50Z on Apr 24, a direct `kubectl set image` pushed a model-engine image (`04729cef…`) that declared a new ORM column `endpoints.temporal_task_queue` with no corresponding Alembic migration applied to production. Every `list_model_endpoints` and `get_model_endpoint` call immediately returned 500. Five patch images over ~34 minutes failed to resolve the issue before a rollback to `f395ffa6…` cleared errors at 00:48Z.

---

## Timeline

| Time (UTC) | Event |
|---|---|
| Feb 27, 05:54Z | Stable image `f395ffa6…` deployed; ran cleanly for ~57 days |
| **Apr 24, 23:50:25Z** | **`kubectl set image` → `04729cef…` (rev 293); ORM references missing `temporal_task_queue` column; all endpoint list/get → 500** |
| 23:55:43Z | `-internal` tag rolled (rev 294); same bug, same 500s |
| Apr 25, 00:09:32Z | `-patch` (rev 296) pushed; does not include migration; 500s continue |
| 00:15:29Z | `-patch2` (rev 297); same |
| 00:20:27Z | `-patch3` (rev 298); same |
| 00:24:11Z | `-patch4` (rev 299); same |
| **00:37Z** | **PagerDuty fires; Envoy 5xx ratio = 0.052** |
| **00:48Z** | **Rollback to `f395ffa6…` (rev 300); errors clear; alert auto-resolves** |
| 01:22Z | 0 errors in last 2 min; incident closed |

---

## Root Cause

**Background terms:**
- **ORM column** — a Python class attribute like `temporal_task_queue = Column(Text)` on the `Endpoint` model. SQLAlchemy (our ORM) translates these attributes into SQL; every declared column is included in the `SELECT` it generates when loading `Endpoint` objects.
- **Alembic migration** — a versioned SQL script (e.g. `ALTER TABLE endpoints ADD COLUMN temporal_task_queue TEXT`) that updates the live database schema to match what the application code expects. Without it, the DB is missing the column the ORM declares.

The `temporal_task_queue` column was added to the `Endpoint` ORM model (`hosted_model_inference.py`) as part of the temporal endpoint type feature (MLI-6425), but **no Alembic migration was written for it**. Because SQLAlchemy includes every ORM-declared column in its generated `SELECT` statement, all endpoint operations — list, get, create, update, delete — issued a query referencing `temporal_task_queue`. PostgreSQL returned `column "temporal_task_queue" does not exist` on every call, causing a 500 across the board.

### Why the existing migration gate didn't protect us

The Helm chart already enforces migration-first ordering:

```yaml
# charts/model-engine/templates/database_migration_job.yaml
annotations:
  "helm.sh/hook": pre-install,pre-upgrade
  "helm.sh/hook-weight": "-1"   # runs before any pod rollout
```

This pre-upgrade hook runs `alembic upgrade head` before the gateway deployment starts. **The incident bypassed this entirely because the deployment used `kubectl set image`, not `helm upgrade`.** `kubectl set image` updates the pod spec directly, skipping all Helm hooks.

### Contributing factors

1. **`kubectl set image` used for production image updates** — bypasses the migration pre-hook built into the Helm chart.
2. **No ORM/DB schema validation at startup** — the app started successfully and served traffic with an invalid schema; errors only appeared at query time. The `/readyz` probe passes as long as the process is up, regardless of DB state.
3. **No migration written for the new column** — the ORM model was updated without a paired migration file.
4. **No rollback SLA** — 5 patch attempts over 34 minutes before rollback was chosen.
5. **Delayed alerting** — PagerDuty did not fire until 00:37Z, 47 min into the incident.

---

## Impact

| Dimension | Detail |
|---|---|
| Customer-facing 500s | ~47 min (23:50Z → 00:37Z alert; resolved 00:48Z) |
| Affected operations | All endpoint list, get, create, update, delete |
| Affected users | All model-engine users |

---

## Action Items

### P0 — Prevent `kubectl set image` in production (preventative, not reactive)

**What:** `kubectl set image` updates pods directly, bypassing all Helm pre-upgrade hooks including the migration job. The fix must be preventative: a `kubectl set image` attempt against the production deployment is rejected by the API server before it takes effect.

Two preventative controls:

**a. RBAC — block `kubectl set image` at the API server**

Developer access to `ml-serving-new` is via the `ml_infra_admin` EKS access entry, which grants cluster-admin cluster-wide. That is why `kubectl set image` succeeds today. Scoping it away from `scale-deploy` rejects `kubectl set image` before any pod is updated.

Scoping away cluster-admin from `scale-deploy` would also break `just deploy prod` run locally (it uses the developer's personal kubectl context). The fix: `just deploy prod` mints a short-lived `ml-k8s-admin` SA token for the helm step. That SA already has cluster-admin (via `kubernetes_cluster_role_binding.ml_k8s_admin` in `workloads.tf`), so helm upgrade continues to work. Developers retain read/exec access via the `scale_developer` role below.

**b. justfile guard — block deploying unmerged code**

`just deploy prod` checks that the current commit is merged to `origin/master` before proceeding. Deploying a local branch to production is rejected before any image is built or pushed.

**Files to change:**

**1. `Terracode-ML/scaleapi-ml-serving/clusters/ml-serving-new/eks.tf` (lines 552–561)**

```hcl
# Before:
  access_scope {
    type = "cluster"
  }

# After:
  access_scope {
    type       = "namespace"
    namespaces = [for ns in local.workload_namespaces_default : ns if ns != "scale-deploy"]
    # ["mlflow", "image-builder", "pyspark", "default", "gpu-operator"]
  }
```

**2. `Terracode-ML/scaleapi-ml-serving/clusters/ml-serving-new/workloads.tf`**

Add a `scale-deploy`-scoped Role (read/exec/log, no deployment mutations) and grant `create` on `serviceaccounts/token` so developers can mint the SA token needed by `just deploy prod`:

```hcl
resource "kubernetes_role" "scale_developer" {
  metadata {
    name      = "scale-developer"
    namespace = "scale-deploy"
  }
  rule {
    api_groups = ["apps"]
    resources  = ["deployments", "replicasets"]
    verbs      = ["get", "list", "watch"]   # no patch/update — blocks kubectl set image
  }
  rule {
    api_groups = [""]
    resources  = ["pods", "pods/log", "pods/exec", "configmaps", "services", "endpoints"]
    verbs      = ["get", "list", "watch", "create", "delete"]
  }
  rule {
    api_groups = ["batch"]
    resources  = ["jobs"]
    verbs      = ["get", "list", "watch", "create", "delete"]
  }
  rule {
    api_groups = [""]
    resources  = ["serviceaccounts/token"]
    verbs      = ["create"]   # allows: kubectl create token ml-k8s-admin -n scale-deploy
  }
}

resource "kubernetes_role_binding" "scale_developer" {
  metadata {
    name      = "scale-developer-binding"
    namespace = "scale-deploy"
  }
  role_ref {
    api_group = "rbac.authorization.k8s.io"
    kind      = "Role"
    name      = kubernetes_role.scale_developer.metadata[0].name
  }
  subject {
    kind = "Group"
    name = "ml-infra-admin"
  }
}
```

**3. `model-engine-internal/justfile`**

Add a master branch guard and have the prod deploy step mint a short-lived SA token:

```just
# New recipe — fails if current commit is not merged to origin/master
_check_master_branch env:
    #!/bin/bash
    if [ "{{ env }}" = "prod" ]; then
        git fetch origin master --quiet
        if ! git merge-base --is-ancestor HEAD origin/master; then
            echo "ERROR: current commit is not merged to origin/master; prod deploy aborted"
            exit 1
        fi
    fi

# Updated _deploy: mint ml-k8s-admin token for prod helm upgrade
_deploy env use_service_identifier: build-push-model-engine (_check_master_branch env) (_set_k8s env) (_mount_aws env)
    #!/bin/bash
    set -euo pipefail
    SA_TOKEN=""
    if [ "{{ env }}" = "prod" ]; then
        SA_TOKEN=$(kubectl create token ml-k8s-admin -n scale-deploy --duration=30m)
    fi
    pushd ../llm-engine/charts
    helm {{ if use_service_identifier == 'true' { 'install model-engine-' + SERVICE_IDENTIFIER } else { 'upgrade model-engine' } }} \
      model-engine \
      -f ../../model-engine-internal/model-engine-internal/resources/values/values_{{ env }}.yaml \
      --description "{{ GIT_MESSAGE }}" \
      {{ if use_service_identifier == 'true' { '--set serviceIdentifier=' + SERVICE_IDENTIFIER } else { '' } }} \
      {{ if env == 'prod' { '--timeout 120m0s' } else { '' } }} \
      --set tag={{ GIT_SHA }} \
      ${SA_TOKEN:+--kube-token="$SA_TOKEN"} \
      --atomic
```

**Files to change:**
- **`Terracode-ML/scaleapi-ml-serving/clusters/ml-serving-new/eks.tf`** (lines 552–561)
- **`Terracode-ML/scaleapi-ml-serving/clusters/ml-serving-new/workloads.tf`**
- **`model-engine-internal/justfile`**

**Owner:** model-engine on-call
**Effort:** ~3 days (Terraform RBAC + Atlantis apply + justfile changes + validation)

---

### Rollout Strategy

**Existing environments — all carry production traffic:**

| Env | Cluster | Purpose |
|---|---|---|
| `training` | `ml-training-new` | production training workloads |
| `launch` | `ml-launch-new` | production Launch API |
| `prod` | `ml-serving-new` | production serving |
| `circleci` | minikube (ephemeral) | CI integration tests only |

There is no staging environment for model-engine. One needs to be created.

**Staging environment — effort estimate:**

The staging backing infrastructure already exists in Terracode-ML `scaleapi-ml-serving/global/` — it was provisioned but never wired up to a model-engine deployment:

| Component | Status | File |
|---|---|---|
| RDS Aurora PostgreSQL (`ml-infra-staging`, secret `staging/ml_infra_pg`) | **Already exists** | `global/db.tf` lines 85–164 |
| ElastiCache Redis (`staging-celery-redis-rg-1`) | **Already exists** | `global/celery-elasticache.tf` lines 44–80 |
| VPC / subnets | **Already exists** (shared ml-serving VPC) | — |
| Route53 DNS (`ml-staging-internal.scale.com`) | **Already exists** | `global/dns.tf` lines 35–49 |
| EKS cluster | **Does not exist** | — |

Add a `scale-deploy-staging` namespace in ml-serving-new. DB, Redis, VPC, and DNS already point at staging endpoints — the namespace is the only missing K8s layer. Estimated ~3–4 days:
- `values_staging.yaml` pointing at `staging/ml_infra_pg` and staging Redis: ~1 day
- K8s namespace + IRSA service account + secrets: ~0.5 day
- `justfile` staging env wiring: ~0.5 day
- Validation / smoke tests: ~1 day

**Process once staging exists:**

```
staging:    any method — kubectl set image, just deploy staging, etc.
            ↓
            validate: migrations apply cleanly, smoke tests pass, no 500s
            ↓
production: just deploy prod   (helm upgrade, migration-first, RBAC-enforced)
```

Any change requiring a DB schema update (new ORM column) must have its Alembic migration merged and applied to production before `just deploy prod` is run.

