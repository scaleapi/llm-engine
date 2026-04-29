# SEV1 Post-Mortem: model-engine Deployment Caused 500s on All Endpoint Operations

**Incident ID:** MLI-6574
**Severity:** SEV1
**Date:** Apr 24–25, 2026
**Duration:** ~58 min (23:50Z deployment → 00:48Z rollback)
**Customer-facing 500s:** ~47 min
**Status:** Resolved; follow-up actions tracked below

---

## Summary

At 23:50Z on Apr 24, a direct `kubectl set image` pushed a model-engine image (`04729cef…`) that declared a new ORM column `endpoints.temporal_task_queue` with no corresponding Alembic migration applied to production. Every `list_model_endpoints` and `get_model_endpoint` call immediately returned 500. Five patch images over ~34 minutes failed to resolve the issue before a rollback to `f395ffa6…` cleared errors at 00:48Z. At least one user missed a project delivery deadline.

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

The `temporal_task_queue` column was added to the `Endpoint` ORM model (`hosted_model_inference.py`) as part of the temporal endpoint type feature (MLI-6425), but **no Alembic migration was written for it**. The column was referenced in every endpoint query via SQLAlchemy's column loading. When PostgreSQL returned `column "temporal_task_queue" does not exist`, SQLAlchemy surfaced a 500 on every call.

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
| Known downstream impact | At least one user missed a project delivery deadline |

---

## Action Items

### P0 — Require Alembic migration before ORM column is added

**What:** Add a CI check that detects when ORM models have columns not covered by any migration. Block merge if a new `Column(...)` in `hosted_model_inference.py` or `model.py` has no corresponding `add_column` in `alembic/versions/`.

**How:**

Option A (preferred) — use `alembic check` in CI:
```bash
# In CI, in the job that already spins up a Postgres container for integration tests
alembic check   # exits non-zero if autogenerate detects pending schema changes
```

Option B — static linter (simpler, imperfect): parse `alembic/versions/` for `add_column("endpoints", …)` and cross-reference against columns declared in the ORM model. Fails the lint job on mismatch.

**Files to change:** CI config, `Makefile` (add `make check-migrations` target).

**Owner:** model-engine on-call
**Effort:** ~1 day

---

### P0 — Enforce `helm upgrade` for production image rollouts; ban `kubectl set image`

**What:** The `kubectl set image` pattern bypasses all Helm pre-upgrade hooks, including the migration job. All production image updates must go through `helm upgrade` so the migration job runs first.

**How:**

Add a deploy wrapper script used by both CI and on-call:
```bash
#!/bin/bash
# scripts/deploy.sh
set -euo pipefail
TAG=${1:?usage: deploy.sh <image-tag>}
helm upgrade model-engine charts/model-engine \
  --set tag="$TAG" \
  --wait \        # block until all hooks and pods are healthy
  --timeout 10m \
  --atomic        # auto-rollback if migration job or pod readiness fails
```

`--atomic` is the key flag: if the migration job exits non-zero or the pod fails its readiness probe within the timeout, Helm automatically rolls back to the previous release with no manual intervention needed.

**Enforcing the ban — options by strength:**

1. **Kubernetes RBAC (simplest):** Remove `patch`/`update` on `deployments` from developer roles. Only the CI service account (used by `helm upgrade`) retains that permission. Developers keep read and exec access; they just can't mutate deployment specs directly.

2. **GitOps with ArgoCD (most robust):** ArgoCD continuously reconciles cluster state against git. A manual `kubectl set image` is detected as drift and reverted within ~3 minutes. Changes must go through a git commit + PR — there is no persistent path outside the pipeline.

3. **Admission controller (OPA/Kyverno):** Write a policy that rejects any `PATCH` on a deployment's image unless it originates from the CI service account or carries a specific Helm annotation. More surgical than RBAC.

RBAC is the fastest win today; GitOps is the right long-term answer.

**Testing a new image without `kubectl set image`:**

- **Staging:** run `scripts/deploy.sh <sha>` against the staging context — same Helm path as production, with migrations enforced.
- **Standalone pod (no deployment mutation needed):** to verify an image starts cleanly on the live cluster without touching the `Deployment` resource:
  ```bash
  kubectl run test-pod --image=<your-image> --rm -it -- bash
  ```
  This doesn't require `deployments/patch`, so RBAC restrictions don't apply. Use it to manually verify DB connectivity, run the schema validator, or inspect logs before cutting a Helm release.

If emergency direct `kubectl set image` is ever required (e.g., Helm state is corrupted), run the migration job manually first:
```bash
kubectl create job db-migration-manual-$(date +%s) \
  --from=job/$(kubectl get jobs -l app=model-engine-database-migration --sort-by=.metadata.creationTimestamp -o name | tail -1)
kubectl wait --for=condition=complete job/db-migration-manual-... --timeout=600s
# only then: kubectl set image ...
```

**Files to change:** `scripts/deploy.sh` (new), `docs/internal/` runbook, CI pipeline, RBAC role manifests.

**Owner:** model-engine on-call
**Effort:** ~1 day

---

### P1 — Add startup schema validation so bad images fail fast instead of serving 500s

**What:** Add an `@app.on_event("startup")` check in `api/app.py` that compares ORM column declarations against actual DB columns. If any ORM column is missing from the DB, raise an exception before the process begins serving traffic. Kubernetes will keep old pods running and never route traffic to the bad pod.

**How:**

Add to `model-engine/model_engine_server/db/base.py`:
```python
from sqlalchemy import inspect as sa_inspect
from model_engine_server.db.models.hosted_model_inference import Base

def validate_schema_or_raise(engine) -> None:
    inspector = sa_inspect(engine)
    for table in Base.metadata.sorted_tables:
        schema = table.schema
        db_cols = {c["name"] for c in inspector.get_columns(table.name, schema=schema)}
        orm_cols = {c.name for c in table.columns}
        missing = orm_cols - db_cols
        if missing:
            raise RuntimeError(
                f"Schema drift detected: table {schema}.{table.name} "
                f"is missing columns {missing}. "
                f"Run 'alembic upgrade head' before deploying this image."
            )
```

Wire into startup in `api/app.py` (alongside the existing `load_redis` startup event):
```python
from model_engine_server.db.base import get_db_manager, validate_schema_or_raise

@app.on_event("startup")
def validate_db_schema():
    db = get_db_manager()
    with db.session_sync() as session:
        validate_schema_or_raise(session.get_bind())
```

The existing readiness probe on `/readyz` (checked every 2s, `failureThreshold: 30` → ~60s window) will prevent Kubernetes from routing traffic to the pod while startup events are failing. If startup raises, the process exits and Kubernetes keeps the old replica set running.

**Files to change:**
- `model-engine/model_engine_server/db/base.py` — add `validate_schema_or_raise()`
- `model-engine/model_engine_server/api/app.py` — add startup event

**Owner:** model-engine on-call
**Effort:** ~1 day (including unit tests with a mock inspector)

---

### P1 — Write the missing `temporal_task_queue` migration before re-deploying MLI-6425

**What:** PR #815 (`lilyz-ai/temporal-endpoint-type`) adds `temporal_task_queue` to the `Endpoint` ORM. A migration must be written and merged — and applied to production — before this image is deployed again.

**How:**
```bash
cd model-engine/model_engine_server/db/migrations
alembic revision --autogenerate -m "add_temporal_task_queue_column"
# review generated file, confirm it contains:
#   op.add_column("endpoints", sa.Column("temporal_task_queue", sa.Text(), nullable=True), schema="hosted_model_inference")
alembic upgrade head   # validate locally against test DB
```

**Files to change:** new file in `model-engine/model_engine_server/db/migrations/alembic/versions/`.

**Owner:** MLI-6425 author
**Effort:** ~1 hour

---

### P2 — Define rollback SLA in the on-call runbook

**What:** Codify a clear decision rule: if the active incident is not resolved within **15 minutes of the first 5xx spike**, or after **2 failed forward-patch attempts**, initiate rollback immediately.

```
Rollback trigger (whichever comes first):
  - 2 forward-patch attempts failed, OR
  - 15 minutes elapsed since first 5xx spike
  → helm rollback model-engine
```

**Files to change:** `docs/internal/smoke-tests.md` or new `docs/internal/oncall-runbook.md`.

**Owner:** on-call lead
**Effort:** ~0.5 day

---

### P2 — Reduce 5xx alert latency from ~47 min to <5 min

**What:** The PagerDuty alert fired 47 minutes after the bad deploy. The Envoy 5xx monitor evaluation window needs to be tightened to catch a step-function spike within 5 minutes.

**How:** In Datadog, find the monitor for `envoy.cluster.upstream_rq_5xx` (or the Envoy 5xx ratio metric). Set the evaluation window to 2–3 minutes with a threshold of ≥1% error rate sustained for 1 minute.

**Files to change:** Datadog monitor config in Terracode-ML `datadog/` directory for the ml-serving account.

**Owner:** on-call/infra
**Effort:** ~1 hour

---

## Lessons Learned

1. **Migration hooks only protect you if you use Helm.** The project had the right guardrail (pre-upgrade hook at weight -1); it was bypassed by using `kubectl set image` directly. The operational pattern matters as much as the technical control.
2. **Fail loudly at startup, not silently on every request.** A startup check that raises an exception is strictly better than an app that boots clean and then 500s on every call. The readiness probe would have contained the blast radius to a failed rollout.
3. **Every ORM column needs a migration.** The ORM model is a lie until the migration runs; treat them as inseparable — the migration file is part of the same diff as the column declaration.
4. **After two failed patches, go backward.** The cost of 5 failed patches was 34 extra minutes of outage. When the root cause is unclear, rollback is always faster than forward.
