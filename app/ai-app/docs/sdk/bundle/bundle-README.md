# AI Bundles — Overview & Setup

This document describes what an **AI Bundle** is in KDCube, how bundles are loaded, and how to register them.

**Why this matters (short pitch)**  
Git‑defined AI bundles enable **safe hot reloads** across replicas:

- **Only new requests** are routed to the updated bundle path
- **In‑flight requests** finish on the previously loaded version
- **Old bundle dirs** are cleaned up periodically (best‑effort, ref‑aware)

## What is an AI Bundle?

An AI bundle is a **self‑contained Python package** that defines a workflow (agent) and optional tools/skills.
It is resolved by bundle id at runtime and loaded by `agentic_loader`.

Reference implementation:

- `kdcube_ai_app/apps/chat/sdk/examples/bundles/react@2026-02-10-02-44/entrypoint.py`

Main bundle loading entrypoints:

- Web: `kdcube_ai_app/apps/chat/api/web_app.py`
- Queue processor: `kdcube_ai_app/apps/chat/processor.py`

## Minimal bundle layout

```
<bundle_root>/
  entrypoint.py                # workflow factory/class
  orchestrator/…               # optional orchestration
  tools_descriptor.py          # optional tool registry
  skills_descriptor.py         # optional skill registry
```

Your entrypoint should expose an agentic workflow factory/class using the decorators
from `kdcube_ai_app/infra/plugin/agentic_loader.py`.

## Bundle registry (AGENTIC_BUNDLES_JSON)

Bundles are registered via `AGENTIC_BUNDLES_JSON` or through the Admin UI (Bundled Integrations).

### 1) Local path bundle

```bash
export AGENTIC_BUNDLES_JSON='{
  "default_bundle_id": "demo-react@2-1",
  "bundles": {
    "demo-react@2-1": {
      "id": "demo-react@2-1",
      "name": "ReAct Demo",
      "path": "/bundles",
      "module": "demo-react@2-1.entrypoint",
      "singleton": false,
      "description": "ReAct agent demo bundle"
    }
  }
}'
```

**Rule (recommended):**
Keep `path` at the parent bundles root (`/bundles`) and set
`module` to `<bundle_folder>.entrypoint`.  
This keeps **local** and **git** definitions consistent and avoids relative‑import issues.

```bash
export AGENTIC_BUNDLES_JSON='{
  "default_bundle_id": "demo.react",
  "bundles": {
    "demo.react": {
      "id": "demo.react",
      "name": "ReAct Demo",
      "path": "/bundles",
      "module": "react@2026-02-10-02-44.entrypoint",
      "singleton": false,
      "description": "ReAct agent demo bundle"
    }
  }
}'
```

### 2) Git‑defined bundle (new)

Use `repo` + optional `ref` and `subdir`.
The bundle will be cloned into the **bundles root** and loaded from there.

**Important (current default):**  
Git resolution can be **disabled** with `BUNDLE_GIT_RESOLUTION_ENABLED=0`.  
When disabled, `git_*` fields are treated as **metadata only** and no clone/pull happens.  
This is recommended until Git bundles are fully configured (keys, creds, networking).

```bash
export AGENTIC_BUNDLES_JSON='{
  "default_bundle_id": "demo.git",
  "bundles": {
    "demo.git": {
      "id": "demo.git",
      "name": "Git bundle",
      "repo": "https://github.com/org/my-bundle.git",
      "ref": "main",
      "subdir": "bundles",
      "module": "demo.git.entrypoint",
      "singleton": false,
      "description": "Bundle loaded from Git"
    }
  }
}'
```

## Bundle attributes (meaning)

| Field | Meaning |
| --- | --- |
| `id` | Stable bundle id used in routing and registry. |
| `name` | Human‑friendly name (UI only). |
| `path` | Filesystem path to the **parent directory** that contains the bundle folder (required for local bundles). |
| `module` | Python module entrypoint **including the bundle folder** (e.g. `my_bundle.entrypoint`). |
| `singleton` | If `true`, reuse the workflow instance across requests. |
| `description` | Free‑text description shown in admin UI. |
| `version` | Bundle version (often content hash); used for snapshots. |
| `repo` | Git repo URL (enables git bundle). |
| `ref` | Git branch/tag/commit. Also used to derive the local folder name. |
| `subdir` | Optional subdirectory inside repo that contains the bundles root (parent folder). |
| `git_commit` | Current HEAD commit (populated after clone/fetch). |

## Built-in example bundles

The processor auto‑adds the example bundles under `apps/chat/sdk/examples/bundles` when
`BUNDLES_INCLUDE_EXAMPLES=1` (default). Set `BUNDLES_INCLUDE_EXAMPLES=0` to disable them.

**Path derivation for repo bundles**

```
<bundles_root>/<repo>__<bundle_id>__<ref>/<subdir?>
```

If `ref` is omitted, the path is:

```
<bundles_root>/<repo>__<bundle_id>/<subdir?>
```

`repo` is derived from the repo URL (last path segment without `.git`).

**Module resolution**

Keep the same module semantics for local and repo bundles:
- `path` / `subdir` points to the **parent** bundles directory.
- `module` includes the bundle folder name (e.g. `react@2026-02-10-02-44.entrypoint`).

**Git resolution toggle**

```
BUNDLE_GIT_RESOLUTION_ENABLED=0   # disable clone/pull (metadata only)
BUNDLE_GIT_RESOLUTION_ENABLED=1   # enable clone/pull (requires git creds)
```
```

### Bundles root

Bundles are stored under a root directory. In Docker deployments you often have **two roots**:

- **Host root** (`HOST_BUNDLES_PATH`) — used for git clones or manually provisioned bundles on the host
- **Container root** (`AGENTIC_BUNDLES_ROOT`) — the path used inside the container

Resolution order:

1. `HOST_BUNDLES_PATH` (preferred on host)
2. `AGENTIC_BUNDLES_ROOT` (container‑visible)
3. `/bundles` (fallback)

**Computed path**:

```
<bundles_root>/<repo>__<bundle_id>/<subdir?>
```

## Admin bundle

The built‑in admin bundle (`kdcube.admin`) is packaged inside the SDK and is auto‑injected
into the registry if missing. Today it serves admin UIs; later it can also host product‑level
chatbot capabilities.

## How bundles are resolved

At runtime, the bundle id is resolved via the registry and loaded by:

- `kdcube_ai_app/infra/plugin/bundle_registry.py` → `resolve_bundle(...)`
- `kdcube_ai_app/infra/plugin/agentic_loader.py`

## Control‑plane updates (tenant/project scoped)

The Admin Bundles API can update **any** tenant/project registry.  
Each processor instance listens only to the channel for **its own** tenant/project.

**Flow (control plane → data plane):**

1. Admin UI / API posts an update with optional `tenant` + `project`.
2. The API writes the registry to Redis **for that tenant/project**.
3. It publishes to:
   ```
   kdcube:config:bundles:update:{tenant}:{project}
   ```
4. Only processors running that tenant/project subscribe to that channel and apply the update locally.

This keeps control‑plane operations global, while data‑plane listeners remain isolated per tenant/project.

For Git bundles:
- The repo is cloned/fetched when resolved (host).
- Remote exec (Fargate) can also fetch bundles via git when needed.

Integration points:
- `bundle_registry.resolve_bundle(...)` is the **single** place that resolves bundle spec → path.
- This is called by:
  - REST entrypoint (`apps/chat/api/web_app.py`)
  - Processor queue handler (`apps/chat/processor.py`)

If the bundle is git‑defined and the path is missing, the registry will clone/fetch it.
Configuration updates propagate to all replicas; each replica applies the registry update and
pulls the git bundle (atomic by default), so in‑flight requests keep using the old path safely.

**Source of truth**
- If `repo` is set → git is the source of truth. `path` is derived.
- If `repo` is not set → `path` is the source of truth (no git actions).

## Typical bundle update procedure

### A) Git‑defined bundle update (recommended)

1. **Tag or commit** your new version in Git.
2. **Update the registry** to point to the new `ref` (tag/branch/commit).
3. The resolved path changes because the path includes `ref`:

```
<bundles_root>/<repo>__<bundle_id>__<ref>/<subdir?>
```

**Example**

Old config:
```
bundle_id: demo.react
repo: git@github.com:org/demo-react.git
ref: v1.0.0
subdir: bundle
```

Old path:
```
/bundles/demo-react__demo.react__v1.0.0/bundle
```

New config:
```
ref: v1.1.0
```

New path:
```
/bundles/demo-react__demo.react__v1.1.0/bundle
```

4. The new path is treated as a **new bundle version**, so caches are refreshed safely.
5. Old paths remain until cleanup (atomic updates).

**Important:** if `ref` stays the same and the path doesn’t change, **existing running processes will keep
their already‑loaded module**. For a deterministic update, always use a new `ref` (tag/commit or new branch
name).

### B) Manual filesystem update (local path)

1. Copy or deploy the new bundle to a **new versioned directory**, e.g.:
   ```
   /bundles/my_bundle_v2
   ```
2. Update the registry `path` to the new directory.
3. The new path is treated as a **new bundle version**, so caches are refreshed.

**Important:** if you overwrite files in place and keep the same `path`,
**the running process will not reliably reload** the code. The update is only guaranteed when the **path changes**.

### Summary: when does an update take effect?

Updates are guaranteed when the **bundle path changes**.  
This is why git updates should use a **new `ref`** and manual updates should use a **new path**.

## Git credentials (private repos)

You must provide credentials in the runtime environment for private repos.

**SSH (recommended)**
- `GIT_SSH_KEY_PATH` — path to private key
- `GIT_SSH_KNOWN_HOSTS` — optional known_hosts file
- `GIT_SSH_STRICT_HOST_KEY_CHECKING` — `yes|no`

Example:

```bash
export GIT_SSH_KEY_PATH=/secrets/id_rsa
export GIT_SSH_KNOWN_HOSTS=/secrets/known_hosts
export GIT_SSH_STRICT_HOST_KEY_CHECKING=yes
```

**HTTPS token**
- Use a token in the URL:
  `https://<token>@github.com/org/repo.git`

**Shallow clone**
- `BUNDLE_GIT_SHALLOW=1` → depth=50
- or `BUNDLE_GIT_CLONE_DEPTH=<N>`

**Always pull**
- `BUNDLE_GIT_ALWAYS_PULL=1` forces refresh on every resolve.

**Atomic updates (safe for in‑flight requests)**
- `BUNDLE_GIT_ATOMIC=1` (default)
- New versions are cloned into a new directory; old versions remain until cleanup.
- Cleanup policy:
  - `BUNDLE_GIT_KEEP` (default 3)
  - `BUNDLE_GIT_TTL_HOURS` (default 0 = disabled)

Atomic folder shape:

```
<bundles_root>/<repo>__<bundle_id>__<ref>__<timestamp>/<subdir?>
```

## Ref tracking & cleanup

Each instance tracks active bundle paths (best‑effort) in Redis so cleanup
won’t delete a version that is still in use.

These settings are now **first‑class** in `Settings` (and can still be set via env):

```
OPEX_AGG_CRON
BUNDLE_CLEANUP_ENABLED
BUNDLE_CLEANUP_INTERVAL_SECONDS
BUNDLE_CLEANUP_LOCK_TTL_SECONDS
BUNDLE_REF_TTL_SECONDS
```

**Redis key**
```
kdcube:config:bundles:refs:{tenant}:{project}
```

**TTL**
```
BUNDLE_REF_TTL_SECONDS=3600
```

**Periodic cleanup loop (API)**
```
BUNDLE_CLEANUP_ENABLED=1
BUNDLE_CLEANUP_INTERVAL_SECONDS=3600
BUNDLE_CLEANUP_LOCK_TTL_SECONDS=900
```

Cleanup uses Redis locks so multiple workers/processes don’t collide.

## Notes on Remote Exec (Fargate)

During isolated execution:

1. A lightweight **workspace snapshot** is sent (workdir/outdir).
2. If bundle tools are required:
   - `BUNDLE_SNAPSHOT_URI` is used (preferred), or
   - Git clone is used if `repo` is provided.

## Admin UI

The bundle registry can be edited via the **AI Bundle Dashboard**:

- UI: `kdcube_ai_app/apps/chat/api/integrations/AIBundleDashboard.tsx`
- Backend: `kdcube_ai_app/apps/chat/api/integrations/integrations.py`

This UI supports both `path` and `git_*` fields.
