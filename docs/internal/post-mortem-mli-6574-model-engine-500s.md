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

Developer kubeconfig contexts for the production cluster (`ml-serving-new`) must not carry `patch`/`update` on `Deployment` resources. Only the CI service account retains those verbs. When a developer runs `kubectl set image`, the API server returns 403 immediately — no pods are updated, no rollback is needed. Developers retain read access and `pods/exec`, so debugging and log inspection are unaffected.

**Finding the right file to change:**

There is no existing developer `Role`/`ClusterRole` for the model-engine namespace in Terracode-ML — no such manifest was found in `scaleapi-ml-serving/`. Developer access to `ml-serving-new` is likely granted via AWS EKS access entries or the `aws-auth` ConfigMap, which may give developers a cluster-level role (e.g. `system:masters` or a custom group). Before making this change:

1. Run `kubectl describe clusterrolebinding -n default | grep -A5 <your-username>` on `ml-serving-new` to find which ClusterRole/Role your personal context binds to.
2. Locate that binding in Terracode-ML `scaleapi-ml-serving/` (or the relevant EKS access entry config).
3. Remove `patch`/`update` on `apps/deployments` from the developer role, or create a namespace-scoped `Role` that overrides the cluster-level grant for the model-engine namespace.

**Deploy path — already exists in model-engine-internal:**

No new script is needed. `model-engine-internal/justfile` already provides the sanctioned deploy command:

```bash
# From model-engine-internal/
just deploy prod   # helm upgrade with --atomic and --timeout 120m0s
just deploy launch # same path, targets ml-launch-new cluster (staging)
```

This runs:
```bash
helm upgrade model-engine model-engine \
  -f model-engine-internal/resources/values/values_prod.yaml \
  --set tag=<GIT_SHA> \
  --atomic \
  --timeout 120m0s
```

The Helm pre-upgrade hook runs the migration job before any pod rolls. `--atomic` rolls back automatically if it fails.

**Files to change:**
- **Terracode-ML `scaleapi-ml-serving/`:** investigate and restrict developer role — remove `patch`/`update` on `apps/deployments` for the production cluster/namespace (exact file TBD after running the kubectl inspect above)
- **model-engine-internal `justfile`:** already correct; no changes needed — `just deploy prod` is the right path

**Owner:** model-engine on-call
**Effort:** ~1 day (mostly the RBAC investigation + review cycle)

---

### Rollout Strategy

**Existing environments in model-engine-internal:**

| Env | Cluster | Values file | Purpose |
|---|---|---|---|
| `training` | `ml-training-new` | `values_training.yaml` | default dev/test target |
| `launch` | `ml-launch-new` | `values_launch.yaml` | separate launch cluster |
| `prod` | `ml-serving-new` | `values_prod.yaml` | production |
| `circleci` | minikube (ephemeral) | `values_circleci.yaml` | CI integration tests only |

The `launch` environment (`ml-launch-new`) may already serve as a staging environment for production deploys — confirm whether it is actively used as production before treating it as staging. If it is available, no new cluster is needed.

**Staging (using `launch` or `training`):**
- No RBAC restrictions — developers can use `kubectl set image`, `helm upgrade`, or any other mechanism freely.
- Use staging to validate: new image starts cleanly, migrations apply without error, smoke tests pass.
- **Effort:** if `launch` is already available and not serving production traffic, this is ~1 hour of process documentation. If a new cluster is needed, effort is significantly higher (new EKS cluster + Terraform + values file).

**Testing a new image on staging before promoting to production:**
```bash
# From model-engine-internal/ — anything goes in staging:
just deploy launch   # full helm path, migration pre-hook runs
kubectl set image deployment/model-engine-gateway gateway=<your-image>   # quick iteration
```

**Production:**
All image updates go through the existing `just deploy prod` command. `kubectl set image` is blocked at the RBAC level — developer roles do not have `patch`/`update` on `Deployment` resources in the production cluster.

```
staging (launch/training):   any method — iterate freely
                             ↓
                             validate: migrations apply, smoke tests pass, no 500s
                             ↓
production:                  just deploy prod   (helm upgrade, migration-first, RBAC-enforced)
```

Any change that requires a DB schema update (new ORM column) must have its Alembic migration merged and applied to production before `just deploy prod` is run.

---

## Lessons Learned

1. **Migration hooks only protect you if you use Helm.** The project had the right guardrail (pre-upgrade hook at weight -1); it was bypassed by using `kubectl set image` directly. The operational pattern matters as much as the technical control.
2. **Fail loudly at startup, not silently on every request.** A startup check that raises an exception is strictly better than an app that boots clean and then 500s on every call. The readiness probe would have contained the blast radius to a failed rollout.
3. **Every ORM column needs a migration.** The ORM model is a lie until the migration runs; treat them as inseparable — the migration file is part of the same diff as the column declaration.
4. **After two failed patches, go backward.** The cost of 5 failed patches was 34 extra minutes of outage. When the root cause is unclear, rollback is always faster than forward.
