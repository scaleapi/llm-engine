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

Developer access to `ml-serving-new` is granted via the `ml_infra_admin` EKS access entry in Terracode-ML, which binds `AWSReservedSSO_MLInfraAdmin_*` to `AmazonEKSClusterAdminPolicy` — cluster-admin cluster-wide. That is why `kubectl set image` succeeds from a developer context today.

**Important:** scoping away cluster-admin from `scale-deploy` also blocks `just deploy prod` run locally, because it uses the developer's personal kubectl context (`_set_k8s prod` switches to `ml-serving-new`). To make the RBAC change safe, **production deploys must move to CI**, where the CI runner authenticates as a service account that retains cluster-admin (`ml_integration_test_lambda`, already granted `AmazonEKSClusterAdminPolicy` in `eks.tf` lines 529–543). Developers trigger the CI deploy job rather than running `just deploy prod` locally.

**Files to change:**

**1. `Terracode-ML/scaleapi-ml-serving/clusters/ml-serving-new/eks.tf` (lines 552–561)**

Kubernetes RBAC is purely additive — a namespace Role cannot deny what a cluster-admin grant allows. The only fix is to remove the cluster-admin grant in `scale-deploy`. Change the `ml_infra_admin` access scope from cluster-wide to all namespaces except `scale-deploy`:

```hcl
# Before:
resource "aws_eks_access_policy_association" "ml_infra_admin" {
  for_each      = toset(data.aws_iam_roles.ml_infra_admin.arns)
  cluster_name  = local.cluster_name
  policy_arn    = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"
  principal_arn = each.value

  access_scope {
    type = "cluster"
  }
}

# After:
resource "aws_eks_access_policy_association" "ml_infra_admin" {
  for_each      = toset(data.aws_iam_roles.ml_infra_admin.arns)
  cluster_name  = local.cluster_name
  policy_arn    = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"
  principal_arn = each.value

  access_scope {
    type       = "namespace"
    namespaces = [for ns in local.workload_namespaces_default : ns if ns != "scale-deploy"]
    # ["mlflow", "image-builder", "pyspark", "default", "gpu-operator"]
  }
}
```

**2. `Terracode-ML/scaleapi-ml-serving/clusters/ml-serving-new/workloads.tf`**

Add a `scale-deploy`-scoped Role that gives developers read/exec/log access without deployment mutations:

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

**3. `model-engine-internal` — move production deploy to CI**

Add a CircleCI deploy job (or manual workflow trigger) that runs `just deploy prod` authenticated as `ml_integration_test_lambda` (already has cluster-admin on `ml-serving-new`). Developers merge to main or manually trigger the job rather than running `just deploy prod` locally.

**Files to change:**
- **`Terracode-ML/scaleapi-ml-serving/clusters/ml-serving-new/eks.tf`** (lines 552–561) — scope `ml_infra_admin` cluster-admin to all namespaces except `scale-deploy`
- **`Terracode-ML/scaleapi-ml-serving/clusters/ml-serving-new/workloads.tf`** — add `scale_developer` Role + RoleBinding for `scale-deploy`
- **`model-engine-internal/.circleci/config.yml`** (or equivalent) — add CI deploy job authenticated as `ml_integration_test_lambda`

**Owner:** model-engine on-call
**Effort:** ~3 days (RBAC Terraform + CI deploy job + Atlantis plan/apply + validation)

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

