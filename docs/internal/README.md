# Model Engine — Operators Guide

Internal documentation for service owners and deployment engineers installing, operating, and debugging model engine in customer environments.

> **Not the end-user docs?** The user-facing Python client documentation is at the root of this site.

---

## Contents

| Document | What it covers | When to use it |
|---|---|---|
| [Architecture](architecture.md) | System overview, component deep-dives, lifecycle flows, autoscaling | Before your first deployment; when debugging unfamiliar failures |
| [Helm Values Reference](helm-values.md) | Every configurable value, organized by concern, with high-risk callouts | During installation and upgrades |
| [Smoke Tests](smoke-tests.md) | Post-deploy validation checklist (Tier A: CPU, Tier B: GPU+LLM) | After every `helm install` or `helm upgrade` |
| [Cloud Support Matrix](cloud-matrix.md) | Per-cloud config reference, behavior differences, image mirroring | When deploying to a specific cloud |

---

## Quick Links

- **Installing for the first time?** → Start with [Architecture](architecture.md) for the mental model, then [Helm Values](helm-values.md) for configuration.
- **Validating a deployment?** → Go straight to [Smoke Tests](smoke-tests.md).
- **Deploying to Azure / GCP / on-prem?** → See [Cloud Support Matrix](cloud-matrix.md) for what to configure differently.
- **Something broken?** → [Troubleshooting Guide](https://scale.atlassian.net/wiki/spaces/EPD/pages/MODEL_ENGINE_TROUBLESHOOTING) (Confluence).

---

## Contributing

These docs live at `docs/internal/` in the repo. Update them when you change behavior or configuration — the PR template includes a reminder.

**Rule:** If a PR author would need to update this doc when changing the code → it belongs here. Operational notes from specific deployments belong in Confluence.
