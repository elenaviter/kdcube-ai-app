---
id: repo:kdcube-ai-app/app/ai-app/docs/ops/health-README.md
title: "Health"
summary: "Liveness and readiness endpoints for deployers, traffic routers, and autoscalers, including proc aggregate application readiness."
tags: ["ops", "health", "readiness", "liveness", "endpoints", "autoscaling"]
keywords: ["health endpoint", "readiness", "liveness", "ingress", "proc", "metrics", "draining", "required application readiness", "HTTP 200"]
updated_at: 2026-08-18
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/ops/deployment-options-index-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/ops/ops-overview-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/ops/s3-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/arch/proc/application-startup-health-and-readiness-README.md
---
# Health Endpoints (Liveness/Readiness)

This doc lists the **health endpoints intended for deployers/autoscalers/compose**.
Only include endpoints that should be used for **liveness/readiness checks**.

---

## Chat Ingress (chat‑ingress)

Endpoint:
- `GET /health`

Checks:
- Service is up
- `draining` flag (returns 503 when draining)
- Socket.IO enabled
- SSE enabled
- Instance id + port

Readiness:
- `200` when healthy
- `503` when draining

Code:
- `src/kdcube-ai-app/kdcube_ai_app/apps/chat/ingress/web_app.py`

---

## Chat Processor (chat‑proc)

Endpoints:

- `GET /health/live`
- `GET /health`

Checks:

- `/health/live` checks process liveness and draining only.
- `/health` checks draining plus aggregate readiness of applications declared
  `service.readiness: required`.
- Both include the proc instance id.
- `/health` includes bounded per-application readiness mode, state, and ready
  status. Independent app preparation is visible but does not change the
  status code.

Readiness:

- `/health/live` returns `200` while the process is live and `503` while
  draining.
- `/health` returns `200` when the process is not draining and every required
  application is ready.
- `/health` returns `503` while draining or while a required application is
  not ready.
- Every application door independently returns an app-scoped `503` for an
  unready application, including apps that do not block aggregate readiness.

Use `/health/live` for process replacement decisions and `/health` for traffic
readiness. A long independent app build must not cause a healthy proc process
to be restarted.

The full state, admission, and deployment-adapter contract is owned by
[Application Startup, Health, And Readiness](../arch/proc/application-startup-health-and-readiness-README.md).

Code:
- `src/kdcube-ai-app/kdcube_ai_app/apps/chat/proc/web_app.py`

---

## Metrics Service

Endpoint:
- `GET /health`

Checks:
- Service is up (returns `{status: "ok"}`)

Readiness:
- `200` when healthy

Code:
- `src/kdcube-ai-app/kdcube_ai_app/apps/metrics/web_app.py`

---

## Knowledge Base (KB)

Endpoints:
- `GET /api/kb/health`
- `GET /api/kb/health/process`

Checks:
- KB stats
- Orchestrator health + queue stats
- Storage path
- Per‑process capacity (process endpoint)

Readiness:
- `200` when healthy
- `503` when unavailable

Code:
- `src/kdcube-ai-app/kdcube_ai_app/apps/knowledge_base/api/web_app.py`

---

## Notes

- `draining` indicates the instance should be removed from load‑balancers.
