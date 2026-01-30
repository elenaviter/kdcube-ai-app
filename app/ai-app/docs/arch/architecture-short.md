# KDCube AI App — System Architecture (Short)

This is a **concise overview** of the current system and integrations.
For a deeper dive, see `architecture-long.md` in this folder.

---

## 1) System at a glance

```mermaid
graph TD
  %% Entry / Auth
  UI[Web UI / Client] -->|HTTPS + masked cookie| NGINX[Web Proxy / Nginx]
  AUTH["ProxyLogin (Delegated Auth + 2FA)"] -->|token exchange| NGINX
  NGINX -->|real auth/id cookies| GATE[Chat API + Gateway]
  KB[Knowledge Base Service] --> GATE
  CP[Control Plane / Project Mgmt] --> GATE

  %% Transport + Gateway
  NGINX -->|SSE / Socket.IO| GATE
  NGINX -->|REST| GATE
  GATE -->|session mgmt| SESS[Session Manager]
  GATE -->|rate limit/backpressure| GW[Gateway + Throttling]

  %% Queue + Processing
  GATE -->|enqueue| Q[Redis Queues]
  Q --> PROC[Chat Processor Workers]

  %% Orchestration
  PROC --> BUNDLES[Dynamic Bundles / Workflows]
  BUNDLES -->|events| RELAY[ChatRelay + Redis Pub/Sub]
  RELAY -->|fan-out| GATE

  %% Context management
  BUNDLES --> CTX[Context Management]
  CTX -->|storage| PG[(Postgres RDS)]
  CTX -->|artifacts| S3[(S3)]
  KB -->|storage| PG
  KB -->|artifacts| S3
  CP -->|policies + quotas| PG

  %% Runtime + providers
  BUNDLES --> RT["Runtime (LLM + Tools)"]
  RT --> DOCKER[Ephemeral Docker Exec]
  RT --> TOOLS[External Tools / APIs]

  subgraph EXTPROV["External Providers (LLMs/Search/Embeddings)"]
    OAI[OpenAI]
    ANTH[Anthropic]
    GEM[Gemini]
    BRAVE[Brave Search]
    DDG[DuckDuckGo]
  end

  RT --> OAI
  RT --> ANTH
  RT --> GEM
  RT --> BRAVE
  RT --> DDG

  %% Cache/Queues/PubSub
  BUNDLES -->|cache/queues/pubsub| REDIS["(Redis / ElastiCache)"]

  classDef aws fill:#e8f4ff,stroke:#7aa7d6,color:#0b2b4f;
  classDef ext fill:#f2f7ee,stroke:#8fbf7a,color:#1f3b1c;
  classDef infra fill:#f7f2ff,stroke:#b69ad6,color:#2b1b4f;

  class PG,REDIS,S3 aws;
  class OAI,ANTH,GEM,BRAVE,DDG,TOOLS ext;
  class AUTH infra;
```

---

## 2) Supported client transports

- **SSE**: primary streaming transport (current UI default)
- **Socket.IO**: fully supported alternative
- **REST**: non‑streaming endpoints (profile/admin/monitoring/etc.)

---

## 3) Auth & token transport

- **Delegated auth** via ProxyLogin; hosted UI for 2FA is always available.
- **Infra auth (cookie‑only mode)**: client stores only a masked (non‑real) token cookie.
  Nginx exchanges it via ProxyLogin to real tokens, sets auth/id cookies, and forwards to API.
- Server accepts tokens from **headers, cookies, SSE query params, Socket.IO auth payload** (for compatibility).

---

## 4) Multi‑tenancy + storage

- **Postgres**: per‑tenant + per‑project schema (prod/dev separated) + **control_plane** schema.
- **S3** (prod): bucket per tenant/project or shared bucket with prefix segmentation; KB artifacts live here too.
- **Redis**: cache + messaging (Pub/Sub) + rate‑limit counters.
- **Neo4j**: optional, currently off.

---

## 5) Limits & economics

- **Gateway rate limiting** + backpressure + circuit breakers.
- **Economics rate limiting**: tier policies, per‑user quotas, concurrency locks.
- **Input limits**: message/attachment size limits enforced at transport layer.

---

## 6) Streaming flow (SSE or Socket.IO)

```mermaid
sequenceDiagram
  participant UI as Client UI
  participant API as Chat API
  participant RL as Redis Relay
  participant Q as Redis Queue
  participant W as Worker / Bundle

  UI->>API: open stream (SSE / Socket.IO connect)
  UI->>API: send message (SSE / Socket.IO)
  API->>Q: enqueue task (per user_type queue)
  W->>Q: dequeue + lock
  W->>RL: publish chat_* events to session channel
  RL-->>API: fan-out to connected stream
  API-->>UI: chat_start/step/delta/complete
```

---

## 7) Key docs

- Comm integrations: [README-comm.md](../comm/README-comm.md)
- Comm architecture: [comm-system.md](../comm-system.md)
- Gateway: [gateway-README.md](../../../infra/gateway/gateway-README.md)
- Economics: [economics-usage.md](../../sdk/infra/economics/economics-usage.md)
- Control plane: [control-plane-management.md](../../sdk/infra/control_plane/control-plane-management.md)
- Monitoring: [README-monitoring-observability.md](../../api/monitoring/README-monitoring-observability.md)
