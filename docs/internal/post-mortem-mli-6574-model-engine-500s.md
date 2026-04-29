# SEV1 Post-Mortem: model-engine Deployment Caused 500s on All Endpoint Operations

**Incident ID:** MLI-6574
**Severity:** SEV1
**Date:** Apr 24–25, 2026
**Duration:** ~58 min (23:50Z deployment → 00:48Z rollback)
**Customer-facing impact:** ~47 min of 500s (23:50Z – 00:37Z PagerDuty alert; resolved 00:48Z)
**Status:** Resolved

---

## Summary

A `kubectl set image` at 23:50Z on Apr 24 deployed a model-engine image that referenced a new ORM field (`endpoints.temporal_task_queue`) which did not exist in the production database. Every `list_model_endpoints` and `get_model_endpoint` call returned 500, blocking all endpoint deployments and operations for all users. Five patch attempts over ~34 minutes failed to address the root cause before a rollback to the previous image resolved the incident. At least one user missed a project delivery deadline.

---

## Timeline

| Time (UTC) | Event |
|---|---|
| Feb 27, 05:54Z | Stable image `f395ffa6…` deployed; ran cleanly for ~57 days |
| **Apr 24, 23:50:25Z** | **`kubectl set image` → `04729cef…` (rev 293); ORM references missing `temporal_task_queue` column; all endpoint list/get → 500** |
| 23:55:43Z | `-internal` tag rolled (rev 294); same bug, same 500s |
| Apr 25, 00:09:32Z | `-patch` (rev 296) — does not fix missing column |
| 00:15:29Z | `-patch2` (rev 297) — does not fix missing column |
| 00:20:27Z | `-patch3` (rev 298) — does not fix missing column |
| 00:24:11Z | `-patch4` (rev 299) — does not fix missing column |
| **00:37Z** | **PagerDuty fires; Envoy 5xx ratio = 0.052** |
| **00:48Z** | **Rollback to `f395ffa6…` (rev 300); errors clear; alert auto-resolves** |
| 01:22Z | 0 errors in last 2 min; incident closed |

---

## Root Cause

The ORM model for the `endpoints` table was updated to declare a new column `temporal_task_queue` (added as part of the temporal endpoint type feature, MLI-6425). The corresponding Alembic database migration **was never applied to production** before the new image was rolled out.

When model-engine started, SQLAlchemy attempted to reference `endpoints.temporal_task_queue` in every endpoint query. Because the column did not exist in the live DB, PostgreSQL returned an error on every `list_model_endpoints` and `get_model_endpoint` call, causing universal 500s.

The five patch images deployed during the incident did not address this — they lacked the migration and the ORM field continued to reference the non-existent column.

**Contributing factors:**
- No migration-before-rollout enforcement: the deployment pipeline does not block a rollout if pending Alembic migrations exist.
- No startup schema validation: model-engine does not fail fast on ORM/DB schema drift; errors only surfaced at query time.
- No rollback SLA: 5 patch attempts were made over ~34 minutes before rollback was chosen. The correct fix (rollback) was not prioritized early enough.
- Delayed alerting: PagerDuty did not fire until 00:37Z, ~47 min after the bad deployment.

---

## Impact

| Dimension | Detail |
|---|---|
| Duration of 500s | ~47 min (23:50Z – 00:37Z alert; resolved 00:48Z) |
| Affected operations | All endpoint list, get, create, update, delete |
| Affected users | All model-engine users |
| Known downstream impact | At least one user missed a project delivery deadline |

---

## Action Items

| # | Action | Owner | Priority |
|---|---|---|---|
| 1 | **Enforce migration-first deployment**: CI/CD pipeline must verify all pending Alembic migrations are applied before new image goes live (or gate rollout on a migration job completing). | Infra/model-engine | High |
| 2 | **Add migration drift detection**: Startup health check that fails fast if ORM schema diverges from live DB schema, preventing silent 500s. | model-engine | High |
| 3 | **Define rollback SLA**: If ≥2 patch attempts fail within 15 minutes, initiate rollback immediately. Document this in the on-call runbook. | On-call/Infra | Medium |
| 4 | **Improve alerting latency**: Envoy 5xx alert threshold and evaluation window should catch this class of incident in <10 min, not 47 min. | Infra | Medium |
| 5 | **Proactive incident communication**: Users mid-deployment should receive Slack/status-page notification during active SEV1s. | On-call/Eng | Medium |

---

## Lessons Learned

- **Schema changes must be deployed atomically with or ahead of the code that depends on them.** A two-phase deploy (migration first, code second) or a migration-gating CI step would have prevented this entirely.
- **Fail fast on startup beats silently failing on every request.** A startup check that validates ORM↔DB schema alignment would have contained the blast radius to a failed rollout rather than a live outage.
- **Five patches in 34 minutes is a sign to stop and rollback.** When the root cause is unclear, reverting to a known-good state is faster and safer than iterating forward.
