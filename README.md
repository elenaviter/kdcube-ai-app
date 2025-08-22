# KDCube AI App — Chat & Knowledge Base Platform

> A library and a platform for building scalable **agentic chat** apps and maintaining .
> Features
> - **multitenancy support**
> - **pluggable agentic app bundle**
> - **hybrid KB search**
> - **real-time Socket.IO streaming**
> - **traceability and monitoring**
> - **spendings tracking**
> - **enterprise auth** including a **“service user acts on behalf of end-user”** pattern.

> SPDX-License-Identifier: MIT • © 2025 Elena Viter

---

## TL;DR

* **CB (Chat App)** – FastAPI + Socket.IO chat service with gateway protection, accounting, circuit breakers, and a 
  **pluggable “agentic bundle”** system (bring your own LangGraph/LangChain app).
* **KB (Knowledge Base)** – Ingest **Markdown, URLs, PDFs** (PDFs via **Marker**), process through modular stages, index to PostgreSQL + **pgvector**, and expose **enhanced hybrid search** (BM25 → ANN → semantic → optional cross-encoder rerank) with **source backtracking**.
* **Auth & Multitenancy** – Tenant/project aware; IdP-backed (Cognito example) with **on-behalf** flows.
* **Real-time** – Redis-backed Socket.IO fan-out for streaming tokens, steps, and KB search results.
* **Planned & Supported** – **Multiple agentic bundles** loaded independently (per env/config), and a shared history store at the library level.

---

## Architecture (in words)

```
[Client UI]  <--Socket.IO/REST-->  [CB Chat Service]  --(tools calls)-->  [KB Service]
     |                                   |                                   |
     |                               [Orchestrator]                  [Postgres + pgvector]
     |                                   |                                   |
     +---------------- Redis (pub/sub + Socket.IO message broker) -----------+
```

* **CB** streams tokens/events to the UI and can call **KB** (REST or Socket.IO) as a tool for RAG/search.
* **KB** ingests & processes content, builds indices, and serves enhanced search with precise navigation.
* **Redis** enables multi-process Socket.IO, serves as a message broker, provides pub/sub for real-time updates.
  It is also used for monitoring, as a sessions store, as a heartbeating center and as a gateway middleware inventory (rate limit, backpressure, etc)
* **PostgreSQL** stores retrieval segments, metadata, vectors, and audit events.
* **Gateway** middleware protects all endpoints; circuit breakers + backpressure defend capacity.
* **Accounting** envelopes bind request metadata across async workers and storage backends.

\* Orchestrator implementation uses Dramatiq in the all-in-one stack.
---

## ai-app repository map (key parts)

```
apps/
  chat/
    api/
      socketio/chat.py           # modular Socket.IO handler
      resolvers.py               # gateway/auth/orchestrator wiring
      monitoring/…               # health & metrics. Will be moved to separate app soon
    default_app/agentic_app.py   # built-in example workflow agentic app bundle
    web_app.py                   # CB FastAPI app
    reg.py / inventory.py        # supported model configs (for moderator interface) & config plumbing
  knowledge_base/
    api/
      search/search.py           # search endpoints (enhanced + highlight)
      registry/registry.py       # moderator endpoints (upload/url/preview/delete)
      socketio/kb.py             # KB Socket.IO handler
      web_app.py                 # KB FastAPI app
    core.py                      # KnowledgeBase core API (add/process/search/backtrack)
    db/providers/…               # Postgres providers & SQL deploy
    modules/…                    # extraction, segmentation, embedding, indexing, summarization
    storage.py                   # FS/S3 storage layer (+ collaborative)
integrations/
    kb/
      clients/
        rest_client.py              # REST client ([CB] service → KB)
        socket_client.py            # Socket.IO client ([CB] service → KB)
infra/
  plugin/agentic_loader.py       # agentic bundle loader (decorators + singleton cache)
  gateway/…, accounting/…, orchestration/…   # infra building blocks
tools/
  fetch.py, parser.py etc.       # This is service', not AI tools
deployment/docker/all_in_one/  # all-in-one Compose (backend + frontend + proxy)   
```

---

## CB — Chat Service

### Highlights

* **Modular Socket.IO** streaming (`/socket.io`): `chat_start`, `chat_step`, `chat_delta`, `chat_complete`, `chat_error`.
* **REST**: `/landing/chat` (fire-and-forget, results over WS), `/landing/chat-real` (queue-backed), `/profile`.
* **Gateway & Auth**: every request establishes `UserSession` with roles/permissions/tenant/project headers; supports **on-behalf** session routing.
* **Accounting Envelopes**: bound to background work & persisted via storage backends.
* **Agentic bundles**: load a user-supplied **app bundle** (LangGraph/LC) at runtime.

## Agentic app bundles (BYO workflows)

Bundles are discovered via decorators in the module you load:

```python
from kdcube_ai_app.infra.plugin.agentic_loader import (
    agentic_workflow, agentic_workflow_factory, agentic_initial_state
)

@agentic_workflow(name="my-chat", version="1.0.0", priority=100)
class MyWorkflow: ...

@agentic_initial_state(name="init")
def create_initial_state(user_message: str) -> dict:
    ...
```

Environment (for single chatbot app):

```
AGENTIC_BUNDLE_PATH=/abs/path/to/bundle_root   # folder with __init__.py
AGENTIC_BUNDLE_MODULE=agentic_app              # module inside that folder (e.g., 'agentic_app')
AGENTIC_SINGLETON=true|false                   # optional
```

> **Planned:** multiple bundles can be configured and loaded **independently** (e.g., different tenants/routes or runtime selection).

> **Chat history** is currently supplied to workflows by client app;
**history persistence** will be implemented in the shared **library layer** so any bundle can opt-in or override.

---

## KB — Knowledge Base Service

### Ingestion & processing

* **Ingest**: sources such as `markdown`, `url` (HTML fetched with robust MIME discovery), **`pdf` (via Marker)**.
* Versioned storage for datasources
* **Deduplicate**: content hash index avoids re-ingesting identical data.
* **Process** (modular):

    1. **Extraction** → Markdown (`extraction_0.md`) + assets
    2. **Segmentation** → continuous & retrieval segments with **line/char ranges**
    3. **Metadata**
    4. **Embedding** (default 1536 dims; configurable)
    5. **Search indexing**
    6. **(Optional) Summarization**
* **Search**: Enhanced **hybrid pipeline** (prefix-aware BM25 → ANN → semantic → optional cross-encoder rerank) with **navigation/backtrack** to source.
* **Moderation** endpoints to upload/add/delete/preview resources.
* **Socket.IO** endpoint (`/socket.io`) for **kb\_search** with correlated `request_id`s.

### Storage layout (FS/S3 backed)

```
tenants/<tenant>/projects/<project>/knowledge_base/
├── data/
│   ├── raw/                # originals (text or bytes)
│   ├── extraction/         # Marker/HTML→MD; extraction_0.md + json
│   ├── segmentation/       # {continuous,retrieval}/segments.json
│   ├── embedding/          # per-segment vectors (size_<dim>)
│   ├── search_indexing/    # index state/metadata
│   └── metadata/           # derived metadata trees
└── log/knowledge_base/YYYY/MM/DD/operations.jsonl
```

Everything is addressable by **RN**:
`ef:<tenant>:<project>:knowledge_base:<stage>:<resource_id>:<version>[:extra]`

### Database (PostgreSQL)

* Tables:
  * `datasource(id,version,uri,title,provider,status,expiration,metadata,...)`
  * `retrieval_segment(id,version,resource_id,content,title,entities,tags,search_vector,embedding VECTOR(1536),...)`   * `content_hash(name,value UNIQUE,type,...)`
  * `events` for audit
* Views for **active/expired** datasets
* Trigger to maintain weighted **tsvector** (title, content, entities, tags)

> Requires extensions: `vector`, `pg_trgm`, `btree_gin`.

### Enhanced hybrid search (two-stage + rerank)

1. **BM25** high-recall, prefix-aware (`to_tsquery('english', 'term:*')`)
2. **ANN** fallback via pgvector `<=>`
3. **Semantic scoring** (`1 - cosine`) on combined candidates
4. **Optional cross-encoder rerank** (+ threshold pruning)

Results include **source\_info**, **heading path**, **char/line ranges**, **base\_segment\_guids**, and **RNs** for raw/extraction/segmentation → enables **exact highlighting & navigation** in UIs.

### Moderator endpoints (selected)

* `POST /api/kb/{project}/upload` – multipart file upload (Markdown/PDF supported)
* `POST /api/kb/{project}/add-url` – add a URL resource
* `POST /api/kb/{project}/upload/process` — dispatch file processing (extraction → indexing)
* `POST /api/kb/{project}/add-url/process` — dispatch url resource processing (extraction → indexing)
* `GET  /api/kb/{project}/resources` – list + processing status + RNs
* `GET  /api/kb/{project}/resource/{id}/preview|download` – binary-safe preview/download
* `DELETE /api/kb/{project}/resource/{id}` – delete resource

### Search & content endpoints

* `POST /api/kb/{project}/search/enhanced` – search with navigation info
* `POST /api/kb/{project}/content/highlighted` – apply citation highlighting on extraction MD by RN
* `POST /api/kb/{project}/content/by-rn` – resolve RN → content (binary-aware; returns preview URLs)
* `POST /api/kb/{project}/content/segment` – fetch a **base segment** by index (+ context & highlights)

### KB Socket.IO (service→service or UI)

**Client → Server**: `"kb_search"` with:

```json
{
  "request_id": "<uuid>",
  "query": "usage of unauthorized ai app",
  "top_k": 5,
  "resource_id": null,
  "on_behalf_session_id": "<session-id>",
  "project": "default-project",
  "tenant": "home-tenant"
}
```

* **Emit**: `"kb_search"` (`{request_id, query, top_k, resource_id?, on_behalf_session_id, project, tenant}`)
* **Receive**: `"kb_search_result"` (correlated via `request_id`), plus `"session_info"`, `"socket_error"`
* **Connect auth**: `bearer_token` + `id_token` + `project` + `tenant` in Socket.IO `auth`

---

## Clients (KB as a tool from CB or other services)

### REST client (service principal + on-behalf)

```python
from kdcube_ai_app.apps.integrations.kb.rest_client import KBServiceClient, build_service_idp_from_env
idp = build_service_idp_from_env()          # Cognito example
client = KBServiceClient(idp, base_url=os.getenv("KDCUBE_KB_BASE_URL","http://localhost:8000"))

result = await client.enhanced_search_on_behalf(
    project=os.getenv("DEFAULT_PROJECT_NAME"),
    query="usage of unauthorized ai app",
    on_behalf_session_id="<session-id>",
    top_k=5,
)
```

### Persistent Socket.IO client (with auto-refresh)

```python
from kdcube_ai_app.apps.integrations.kb.socket_client import PersistentKBServiceSocketClient, IdpConfig

idp_cfg = IdpConfig("cognito", region=..., user_pool_id=..., client_id=..., username=..., password=..., use_admin_api=True)

client = PersistentKBServiceSocketClient(
    kb_socket_url=os.getenv("KB_SOCKET_URL","http://localhost:8000/socket.io"),
    idp_cfg=idp_cfg,
    project=os.getenv("DEFAULT_PROJECT_NAME"),
    tenant=os.getenv("TENANT_ID"),
)
await client.start()
resp = await client.submit_kb_search(query="usage of unauthorized ai app", on_behalf_session_id="<session-id>")
```

---

## Run it

### All-In-One Docker Compose
A ready-to-run stack that brings up **Postgres (pgvector)**, **Redis**, **Dramatiq**, **KB**, **CB**, **Web UI**, and an **Nginx proxy**.

For details, see [kdcube-ai-app dockercompose README.md](app/ai-app/deployment/docker/all_in_one/README.MD).

### Manually

### Minimal environment
See [sample_env](app/ai-app/deployment/manual/sample_env)

### Infra prerequisites (Redis, Postgres, S3)
* Python 3.11+
* Follow [compose-persistence.md](app/ai-app/services/kdcube-ai-app/kdcube_ai_app/readme/kb/compose-persistence.md) to run and configure Postgres / Redis. 
* (Optional) **S3** credentials if using S3 storage

### Start services

```bash
# KB
uvicorn kdcube_ai_app.apps.knowledge_base.api.web_app:app --host 0.0.0.0 --port ${KB_APP_PORT} --reload

# Dramatiq worker

# CB
uvicorn kdcube_ai_app.apps.chat.web_app:app --host 0.0.0.0 --port ${CHAT_APP_PORT} --reload
```

---

## Permissions & security

* **Gateway** issues/reconfirm a `UserSession` for each request/connection.
* WebSocket connection requires `bearer_token` + `id_token` + session established via HTTP. 
  Additionally, `project` + `tenant` are required. All this data must be placed in `auth` payload.
  
* Role/permission checks gate **read**/**write**/**admin** KB endpoints.
* **On-behalf**: service principals can call KB/CB on behalf of a **user session** (`on_behalf_session_id`).
* Headers for service calls are built from the **IdP token bundle**; optional passthrough of user tokens is supported.

**Typical error**: `User has no permissions assigned.`
→ ensure roles/permissions for the principal or the on-behalf session.

---

## Troubleshooting

* CB/KB. **Socket.IO connects but search fails** → check IdP tokens, project/tenant in `auth` payload, and KB permissions.
* KB. PDF processing. **No extraction output** → verify Marker is available and content type detection (PDF/HTML) is correct.
* KB. **Duplicate ingestion** → dedup is by **content hash**; if identical, return will include `status="duplicate"`/`"unchanged"`.

---

## Roadmap

* KB: Add option for repositories synchronization
* KB: Add graph DB 
* KB: Code ingestion
* CB: **Multiple agentic bundle dynamic loading** (apps hub)
* CB: Agentic context DB available at the library level on demand
* CB: Agentic context DB with retrieval.
* CB: Code execution sandbox (e.g., for LangGraph/LC)

---

## License
MIT © 2025 Elena Viter. See [LICENSE](./LICENSE).


