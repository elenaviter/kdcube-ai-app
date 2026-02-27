# ECS Deployment Plan (Ingress + Proc + Metrics)

This document describes a **target ECS layout** for the split services and how to transition from docker‑compose.

---

## 1) Services

**Core services (separate ECS services):**

1. **Ingress** (`kdcube-chat-ingress`)
2. **Processor** (`kdcube-chat-proc`)
3. **Metrics** (`kdcube-metrics`)

**Optional:**

- `kdcube-web-proxy` (nginx/openresty)
- `kdcube-web-ui`
- `proxylogin` (if delegated auth is required)

---

## 2) Networking + Routing

Recommended routing rules (path‑based):

- `/chatbot/api/*` → ingress
- `/chatbot/api/integrations/*` → processor
- `/metrics/*` → metrics (internal only, if exposed)

**SSE:**  
Ensure ALB idle timeout is long enough for SSE connections.

---

## 3) Task Definitions

Each service should have its own task definition and task role.

**Ingress task:**
- Image: `kdcube-chat-ingress`
- Env: `.env.ingress` equivalent
- Depends on Redis + Postgres

**Processor task:**
- Image: `kdcube-chat-proc`
- Env: `.env.proc` equivalent
- Needs access to bundles + exec workspace

**Metrics task:**
- Image: `kdcube-metrics`
- Env: `.env.metrics` equivalent

---

## 4) Storage + Bundles

Two options for bundles:

1. **EFS mounted read‑only** (simplest for ECS)
2. **Git‑based bundle fetch at runtime** (future)

Processor tasks must access bundles.

**Current recommended approach (before Git bundles):**

- Build a **customer‑specific processor image** that bakes bundles into `/bundles`.
- Run that image in ECS (no host mounts required).

---

## 5) Environment Strategy

Use a **single shared `GATEWAY_CONFIG_JSON`** for all components:

- `tenant` + `project`
- capacity + rate limits
- `limits.proc.max_queue_size`

Keep secrets in AWS Secrets Manager / SSM.

---

## 6) ECS Launch Type

**Fargate:**
- Works for ingress + metrics
- For processor with Docker‑in‑Docker tools, you may need EC2 or a separate executor service.

**EC2:**
- Allows Docker socket mount for tools.

---

## 7) Rolling Migration

1. Keep docker‑compose running (production baseline).
2. Deploy **metrics service** in ECS (read‑only).
3. Move **ingress** to ECS (processor still on EC2).
4. Move **processor** to ECS (after bundles + exec workspace are solved).

---

## 8) Observability

- CloudWatch logs per service
- Metrics service exporting to CloudWatch or Prometheus
