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

**How — Kubernetes RBAC:**

Developer access to `ml-serving-new` is granted via the `ml_infra_admin` EKS access entry in Terracode-ML, which binds the `AWSReservedSSO_MLInfraAdmin_*` IAM role to `AmazonEKSClusterAdminPolicy` — effectively cluster-admin. That is why `kubectl set image` currently succeeds from a developer context. The fix requires replacing the blanket cluster-admin grant with a scoped role that excludes `patch`/`update` on `apps/deployments`.

**Files to change:**

**1. `Terracode-ML/scaleapi-ml-serving/clusters/ml-serving-new/eks.tf` (lines 545–561)**

Currently grants cluster-admin to ml-infra admins:
```hcl
resource "aws_eks_access_policy_association" "ml_infra_admin" {
  policy_arn = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"
  access_scope { type = "cluster" }
}
```
Change: replace `AmazonEKSClusterAdminPolicy` with a namespace-scoped policy, or keep cluster-admin for infra but add the namespace-scoped Role below to override deployment mutations in `scale-deploy`.

**2. `Terracode-ML/scaleapi-ml-serving/clusters/ml-serving-new/workloads.tf`**

Add a namespace-scoped `Role` + `RoleBinding` in the `scale-deploy` namespace that restricts deployment mutations to the CI service account only. In Kubernetes, RBAC is additive — you cannot deny via a Role — so the correct approach is to ensure developers are not bound to any role that grants `patch`/`update` on `apps/deployments` in `scale-deploy`. Concretely:

```hcl
resource "kubernetes_role" "model_engine_deployer" {
  metadata {
    name      = "model-engine-deployer"
    namespace = "scale-deploy"
  }
  rule {
    api_groups = ["apps"]
    resources  = ["deployments"]
    verbs      = ["get", "list", "watch"]   # read-only; no patch/update
  }
  # ... all other developer-needed verbs on pods, logs, configmaps, etc.
}
```

Only the `ml-k8s-admin` service account (already cluster-admin, used by `helm upgrade`) retains deployment mutation rights. A developer running `kubectl set image` against `scale-deploy` gets a 403.

**Deploy path — already exists in model-engine-internal:**

No new script is needed. `model-engine-internal/justfile` already provides:
```bash
just deploy prod   # helm upgrade --atomic --timeout 120m0s, runs migration pre-hook first
```

**Files to change:**
- **`Terracode-ML/scaleapi-ml-serving/clusters/ml-serving-new/eks.tf`** (lines 545–561) — scope or replace the ml_infra_admin cluster-admin policy
- **`Terracode-ML/scaleapi-ml-serving/clusters/ml-serving-new/workloads.tf`** — add `model-engine-deployer` Role + RoleBinding for `scale-deploy` namespace, removing `patch`/`update` on `apps/deployments`
- **`model-engine-internal/justfile`** — already correct; `just deploy prod` is the right path

**Owner:** model-engine on-call
**Effort:** ~2 days (Terraform RBAC changes + Atlantis plan/apply + validation)

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

**Staging environment — effort estimate (~8–11 days total):**

Several staging infrastructure components already exist in Terracode-ML `scaleapi-ml-serving/global/`:

| Component | Status | File | Effort |
|---|---|---|---|
| RDS Aurora PostgreSQL | **Already exists** (`ml-infra-staging`) | `global/db.tf` lines 85–164 | 0 days |
| ElastiCache Redis | **Already exists** (`staging-celery-redis-rg-1`) | `global/celery-elasticache.tf` lines 44–80 | 0 days |
| VPC / subnets | **Already exists** (reuse ml-serving VPC) | — | 0 days |
| Route53 DNS (`ml-staging-internal.scale.com`) | **Already exists** | `global/dns.tf` lines 35–49 | 0 days |
| EKS cluster (`ml-staging-new`) | **Does not exist** | new `clusters/ml-staging-new/` | ~4–5 days |
| IAM / IRSA roles | Partial (extend from prod) | `global/iam-irsa.tf` | ~1–2 days |
| K8s namespace + secrets | Does not exist | applied via Helm/kubectl post-cluster | ~0.5 day |
| `values_staging.yaml` + service/infra configs | Does not exist | `model-engine-internal/resources/values/` | ~1 day |
| `justfile` staging env wiring | Does not exist | `model-engine-internal/justfile` | ~0.5 day |

The dominant cost is the new EKS cluster (node groups, GPU operators, Istio, autoscaler, IRSA wiring). The DB, Redis, VPC, and DNS are already provisioned.

**Process once staging exists:**

```
staging:    any method — kubectl set image, just deploy staging, etc.
            ↓
            validate: migrations apply cleanly, smoke tests pass, no 500s
            ↓
production: just deploy prod   (helm upgrade, migration-first, RBAC-enforced)
```

Any change requiring a DB schema update (new ORM column) must have its Alembic migration merged and applied to production before `just deploy prod` is run.

---

## Lessons Learned

1. **Migration hooks only protect you if you use Helm.** The project had the right guardrail (pre-upgrade hook at weight -1); it was bypassed by using `kubectl set image` directly. The operational pattern matters as much as the technical control.
2. **Fail loudly at startup, not silently on every request.** A startup check that raises an exception is strictly better than an app that boots clean and then 500s on every call. The readiness probe would have contained the blast radius to a failed rollout.
3. **Every ORM column needs a migration.** The ORM model is a lie until the migration runs; treat them as inseparable — the migration file is part of the same diff as the column declaration.
4. **After two failed patches, go backward.** The cost of 5 failed patches was 34 extra minutes of outage. When the root cause is unclear, rollback is always faster than forward.
