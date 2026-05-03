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

### P0 — Add CI check: every new ORM column requires a paired Alembic migration

**What:** Add a CI check that blocks merge when the ORM model declares a column that has no corresponding migration. This prevents the "no migration written" contributing factor, though it does not by itself prevent `kubectl set image` from bypassing migration execution (that is addressed in the next item).

**How:**

Option A (preferred) — use `alembic check` in CI:
```bash
# In the CI job that already spins up a Postgres container for integration tests
alembic check   # exits non-zero if autogenerate detects pending schema changes
```

Option B — static linter (simpler, imperfect): cross-reference `add_column("endpoints", …)` calls in `alembic/versions/` against columns declared in the ORM model. Fail the lint job on mismatch.

**Files to change:** CI config, `Makefile` (add `make check-migrations` target).

**Owner:** model-engine on-call
**Effort:** ~1 day

---

### P0 — Prevent `kubectl set image` in production (preventative, not reactive)

**What:** `kubectl set image` updates pods directly, bypassing all Helm pre-upgrade hooks including the migration job. The fix must be preventative: `kubectl set image` on the production deployment should be blocked or immediately reverted — not just rolled back after a failure is detected.

**Preferred approach — Kubernetes RBAC:**

Remove `patch`/`update` permissions on `Deployment` resources from developer roles in the production namespace. Only the CI service account retains those permissions. A `kubectl set image` attempt from a developer will be rejected by the API server immediately — no deployment occurs, no rollback needed.

```yaml
# Remove from developer ClusterRole / Role in prod namespace:
# - apiGroups: ["apps"]
#   resources: ["deployments"]
#   verbs: ["patch", "update"]   # <-- remove these
#   verbs: ["get", "list", "watch"]  # <-- keep read-only
```

**Alternative — GitOps (ArgoCD):**

ArgoCD continuously reconciles cluster state against git. A manual `kubectl set image` is detected as drift and reverted within ~3 minutes. This is more robust than RBAC alone because it also catches changes made via other paths (e.g. direct `kubectl edit`).

**Sanctioned production deploy path:**

All production image updates go through a deploy script that runs `helm upgrade` with the migration pre-hook:
```bash
#!/bin/bash
# scripts/deploy.sh
set -euo pipefail
TAG=${1:?usage: deploy.sh <image-tag>}
helm upgrade model-engine charts/model-engine \
  --set tag="$TAG" \
  --wait \
  --timeout 10m \
  --atomic
```

**Testing a new image before deploying to production:**

Use staging (see Rollout Strategy below) or run a standalone pod against the production cluster without mutating the `Deployment`:
```bash
kubectl run test-pod --image=<your-image> --rm -it -- bash
```
This requires only `pods/create`, which developer roles retain.

**Files to change:** RBAC role manifests (production namespace), `scripts/deploy.sh` (new), CI pipeline config.

**Owner:** model-engine on-call
**Effort:** ~1 day

---

### Rollout Strategy

**Staging:** No restrictions. Developers can use `kubectl set image`, `helm upgrade`, or any other mechanism to test images against the staging cluster. Staging exists specifically for iteration before production.

**Production:** All image updates must go through `helm upgrade` via `scripts/deploy.sh`. `kubectl set image` is blocked at the RBAC level — developer roles do not have `patch`/`update` on `Deployment` resources in the production namespace. The `helm upgrade` path runs the migration pre-hook automatically before any pod rolls over.

The deploy path in both environments:
```
staging:   any method (kubectl set image, helm upgrade, etc.)
           ↓
           validate: smoke tests pass, no 500s
           ↓
production: scripts/deploy.sh <image-tag>   (helm upgrade, migration-first, RBAC-enforced)
```

Any change that requires a DB schema update (new ORM column) must have its Alembic migration merged and applied to production before the new image is deployed.

---

## Lessons Learned

1. **Migration hooks only protect you if you use Helm.** The project had the right guardrail (pre-upgrade hook at weight -1); it was bypassed by using `kubectl set image` directly. The operational pattern matters as much as the technical control.
2. **Fail loudly at startup, not silently on every request.** A startup check that raises an exception is strictly better than an app that boots clean and then 500s on every call. The readiness probe would have contained the blast radius to a failed rollout.
3. **Every ORM column needs a migration.** The ORM model is a lie until the migration runs; treat them as inseparable — the migration file is part of the same diff as the column declaration.
4. **After two failed patches, go backward.** The cost of 5 failed patches was 34 extra minutes of outage. When the root cause is unclear, rollback is always faster than forward.
