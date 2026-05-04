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

**a. RBAC — block `kubectl set image`, restrict deploy to a dedicated role**

Today `ml_infra_admin` has cluster-admin cluster-wide, which is why `kubectl set image` succeeds. The fix splits access into two roles:

- **`ml_infra_admin`** (developers): scoped away from `scale-deploy`; gets read/exec/log only via the `scale_developer` Role. Cannot deploy.
- **`ml-serving-deployer`** (new, deploy-only IAM role): scoped to `scale-deploy` with deployment-mutation rights. Only this role can run `just deploy prod`.

Separating "I can read/debug prod" from "I can deploy to prod" at the IAM level — no token minting, no workarounds, easy to audit who has deploy access.

**Cross-service impact:** `scale-deploy` is a shared namespace. Running `kubectl get deployments -n scale-deploy` on `ml-serving-new` shows two non-model-engine deployments:

- `k8s-startup-watcher` — part of the llm-engine Helm chart, managed by `just deploy prod`; unaffected.
- `auto-hillclimb-ui` — a Streamlit app owned by the **genai team** (`genai/genai/applications/auto_hillclimb/`), deployed manually with `ml_infra_admin` credentials, no Terraform backing. **This deploy would break.**

The hundreds of `launch-endpoint-id-*` deployments are model-engine user endpoints created by the celery workers via the `ml-k8s-admin` service account — not by developer kubectl — so those are unaffected.

Before applying the RBAC change: coordinate with the genai team. They either need to be granted `ml-serving-deployer` access, or their `auto-hillclimb-ui` deploy process needs to be moved to a separate namespace or a dedicated deployer role.

**b. justfile guard — block deploying unmerged code**

`just deploy prod` checks that the current commit is merged to `origin/master` before proceeding. Deploying a local branch to production is rejected before any image is built or pushed.

**Files to change:**

**1. `Terracode-ML/scaleapi-ml-serving/clusters/ml-serving-new/eks.tf`**

Scope `ml_infra_admin` away from `scale-deploy` (lines 552–561), and add a new EKS access entry for `ml-serving-deployer`:

```hcl
# Existing — scope ml_infra_admin away from scale-deploy:
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

# New — deploy-only role for scale-deploy:
data "aws_iam_roles" "ml_serving_deployer" {
  name_regex  = "AWSReservedSSO_MLServingDeployer.*"
  path_prefix = "/aws-reserved/sso.amazonaws.com/"
}

resource "aws_eks_access_entry" "ml_serving_deployer" {
  for_each      = toset(data.aws_iam_roles.ml_serving_deployer.arns)
  cluster_name  = local.cluster_name
  principal_arn = each.value
  type          = "STANDARD"
}

resource "aws_eks_access_policy_association" "ml_serving_deployer" {
  for_each      = toset(data.aws_iam_roles.ml_serving_deployer.arns)
  cluster_name  = local.cluster_name
  policy_arn    = "arn:aws:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"
  principal_arn = each.value

  access_scope {
    type       = "namespace"
    namespaces = ["scale-deploy"]
  }
}
```

**2. `Terracode-ML/scaleapi-ml-serving/clusters/ml-serving-new/workloads.tf`**

Add `scale_developer` (read/exec/log for `ml_infra_admin`, no deploy) and `scale_deployer` (deployment mutations for `ml-serving-deployer`):

```hcl
# Developers: read/exec/log only — cannot kubectl set image or helm upgrade
resource "kubernetes_role" "scale_developer" {
  metadata { name = "scale-developer"; namespace = "scale-deploy" }
  rule {
    api_groups = ["apps"]
    resources  = ["deployments", "replicasets"]
    verbs      = ["get", "list", "watch"]
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
  metadata { name = "scale-developer-binding"; namespace = "scale-deploy" }
  role_ref {
    api_group = "rbac.authorization.k8s.io"
    kind      = "Role"
    name      = kubernetes_role.scale_developer.metadata[0].name
  }
  subject { kind = "Group"; name = "ml-infra-admin" }
}

# Deployers: full deployment mutation rights — only ml-serving-deployer can deploy
resource "kubernetes_cluster_role_binding" "scale_deployer" {
  metadata { name = "scale-deployer-cluster-admin" }
  role_ref {
    api_group = "rbac.authorization.k8s.io"
    kind      = "ClusterRole"
    name      = "cluster-admin"
  }
  subject { kind = "Group"; name = "ml-serving-deployer" }
}
```

**3. `model-engine-internal/justfile`**

Add a master branch guard; update `_mount_aws` to use `ml-serving-deployer` profile for prod:

```just
# New — fails if current commit is not merged to origin/master
_check_master_branch env:
    #!/bin/bash
    if [ "{{ env }}" = "prod" ]; then
        git fetch origin master --quiet
        if ! git merge-base --is-ancestor HEAD origin/master; then
            echo "ERROR: current commit is not merged to origin/master; prod deploy aborted"
            exit 1
        fi
    fi

# Updated _mount_aws: prod deploys use ml-serving-deployer profile
_mount_aws env: _aws-mountable_folder
    {{ if env == 'training' { '...' } else { '' } }}
    {{ if env == 'prod'     { 'cd .. && python3 scripts_py3/scale_scripts/exe/generate_mountable_aws_credentials.py -p ml-serving-deployer -e ~/.aws-mountable/credentials' } else { '' } }}
    {{ if env == 'launch'   { '...' } else { '' } }}

# Updated deploy: add master branch check
deploy env='training': (_check_master_branch env) (_deploy env 'false') (_emit_deploy_event "model-engine" env)
```

**Files to change:**
- **`Terracode-ML/scaleapi-ml-serving/clusters/ml-serving-new/eks.tf`** — scope `ml_infra_admin`, add `ml_serving_deployer` access entry
- **`Terracode-ML/scaleapi-ml-serving/clusters/ml-serving-new/workloads.tf`** — `scale_developer` + `scale_deployer` roles
- **`Terracode-ML/scaleapi-ml-serving/clusters/ml-serving-new/iam.tf`** — add `AWSReservedSSO_MLServingDeployer` IAM role data source and aws-auth mapping
- **`model-engine-internal/justfile`** — `_check_master_branch`, `_mount_aws` prod profile

**Owner:** model-engine on-call
**Effort:** ~3–4 days (new IAM SSO role + Terraform RBAC + Atlantis apply + justfile + validation)

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

