# Custom CI/CD (Open Source + Customer Repo)

This document is a practical plan for a **two‑repo** CI/CD setup:

- **Platform repo (open source):** `kdcube-ai-app`  
- **Customer repo (closed):** bundles + frontend + proxy config

Goal: keep **docker‑compose (EC2)** working during transition, while preparing **ECS** deployment.

---

## TL;DR (1‑page CI/CD Flow)

**Inputs (single release file in customer repo):**
- `release.yaml` (pins platform tag + frontend tag + bundles list)

**CI (build phase):**
1. Checkout **platform repo** at `platform.ref`.
2. Checkout **customer repo** at `bundles.items[].ref` + `frontend.ref`.
3. For each bundle entry → copy `<subdir>/<bundle_dir>` into `/bundles/<module_first_segment>`.
4. Generate **runtime descriptor** (`AGENTIC_BUNDLES_JSON`) with `path=/bundles` and `module=...entrypoint`.
5. Build + push images:
   - `kdcube-chat-ingress`
   - `kdcube-chat-proc` (or `kdcube-chat-proc-bundled`)
   - `py-code-exec` (or `kdcube-exec-bundled`)
   - `kdcube-metrics`
   - `kdcube-web-ui`
   - `proxylogin` / `web-proxy` (if used)

**CD (deploy phase):**
- **ECS:** update task definitions with new image tags + envs.
- **EC2 compose:** update `.env.*` files and `docker compose up -d`.

```mermaid
flowchart LR
  R[release.yaml<br/>customer repo] --> CI[CI build]
  CI --> I[kdcube-chat-ingress]
  CI --> P[kdcube-chat-proc-bundled]
  CI --> E[kdcube-exec-bundled]
  CI --> M[kdcube-metrics]
  CI --> U[kdcube-web-ui]
  CI --> PX[proxylogin/web-proxy]
  I --> CD[Deploy ECS/Compose]
  P --> CD
  E --> CD
  M --> CD
  U --> CD
  PX --> CD
```

**Runtime config (where env lives):**
- ECS: task definitions (ingress/proc/metrics)
- Compose: `.env.ingress`, `.env.proc`, `.env.metrics`
- UI config: `config.json` baked or mounted at runtime

---

## 1) Build Outputs (Images)

**Platform images (built from open‑source repo):**

- `kdcube-chat-ingress` (thin ingress service)
- `kdcube-chat-proc` (processor service)
- `kdcube-metrics` (metrics service)
- `kdcube-postgres-setup` (db schema provisioning)
- `kdcube-web-proxy` (nginx/openresty proxy)
- `proxylogin` (auth proxy for delegated flows)

**Customer images (built from customer repo):**

- `kdcube-web-ui` (frontend bundle)
- `kdcube-chat-proc-bundled` (processor image with bundles baked in)
- `kdcube-exec-bundled` (code‑exec image with bundles baked in)

---

## 1.1) Image → Dockerfile Mapping

**Platform repo (open source):**  
Use Dockerfiles from `deployment/docker/custom-ui-managed-infra`.  
(They are identical to the local‑infra variants; keep one source of truth for CI.)

| Image | Dockerfile |
| --- | --- |
| `kdcube-chat-ingress` | `app/ai-app/deployment/docker/custom-ui-managed-infra/Dockerfile_Ingress` |
| `kdcube-chat-proc` | `app/ai-app/deployment/docker/custom-ui-managed-infra/Dockerfile_Chatproc` |
| `kdcube-metrics` | `app/ai-app/deployment/docker/custom-ui-managed-infra/Dockerfile_Metricservice` |
| `kdcube-postgres-setup` | `app/ai-app/deployment/docker/custom-ui-managed-infra/Dockerfile_PostgresSetup` |
| `kdcube-web-proxy` | `app/ai-app/deployment/docker/custom-ui-managed-infra/Dockerfile_ProxyOpenResty` |
| `proxylogin` | `app/ai-app/deployment/docker/custom-ui-managed-infra/Dockerfile_ProxyLogin` |
| `py-code-exec` | `app/ai-app/deployment/docker/custom-ui-managed-infra/Dockerfile_Exec` |

**Customer repo (closed):**

| Image | Dockerfile |
| --- | --- |
| `kdcube-web-ui` | `<customer-repo>/ops/.../Dockerfile_UI` |
| `kdcube-chat-proc-bundled` | `<customer-repo>/ops/.../Dockerfile_ChatProcBundled` (base = `kdcube-chat-proc`) |
| `kdcube-exec-bundled` | `<customer-repo>/ops/.../Dockerfile_ExecBundled` (base = `py-code-exec`) |

---

## 2) Recommended Tagging Strategy

**Platform repo:**

- `:git-sha` (immutable)
- `:branch` (moving tag for dev/staging)
- `:vX.Y.Z` (optional releases)

**Customer repo:**

- `:customer-sha`
- `:customer-branch`

---

## 3) Release Manifest (Single Source of Truth)

To avoid **building on every commit**, define a **release manifest** in the customer repo.
CI runs **only** when this file changes (or when a release tag referencing it is created).

Recommended path: `release.yaml`

**Single file, all versions in one place.** This is the only release input CI should need.

Example:

```yaml
release_name: "prod-2026-02-22"

platform:
  repo: "kdcube-ai-app"
  ref: "v0.3.2"

frontend:
  repo: "customer-repo"
  ref: "ui-v2026.02.22"

# Bundle sources and versions (single source of truth).
# Each bundle entry must include: repo, ref (tag/sha), subdir, bundle_dir, and entrypoint module.
bundles:
  default_bundle_id: "app@2-0"
  items:
    - id: "app@2-0"
      repo: "customer-repo"
      ref: "bundle-v2026.02.22"   # tag or commit sha
      subdir: "service/bundles"
      bundle_dir: "demo_core/bundle/demo-core@2-0"
      # Module path is relative to /bundles.
      # Module first segment must match the folder name placed under /bundles.
      module: "demo-core@2-0.entrypoint"
      read_only: false
    - id: "ops@1.3.0"
      repo: "customer-repo"
      ref: "bundle-v2026.02.22"   # tag or commit sha
      subdir: "service/bundles"
      bundle_dir: "demo_ops/bundle/demo-ops@1-3"
      module: "demo-ops@1-3.entrypoint"
      read_only: false

images:
  build_ingress: true
  build_proc: true
  build_exec: true
  build_metrics: true
```

**Who edits what:**

- Platform team → `VERSION`, tag in open‑source repo.
- Customer bundle team → bundle code + bundle tag.
- Customer frontend team → UI code + UI tag.
- Release manager → `release.yaml` (pins platform + bundles + UI).

**CI trigger rule (recommended):**
- Run release pipeline **only** when `release.yaml` changes on the release branch.

---

## 4) Bundle Descriptor (Generated by CI)

**Location:** generated by CI from `release.yaml` (not stored as a source of truth).

**Why:** The runtime needs `AGENTIC_BUNDLES_JSON`, but the *release file* is the source of truth.
CI uses `release.yaml` to build a **runtime descriptor** that points to `/bundles` in the image.
`path` is always `/bundles`, and `module` is **relative to `/bundles`**.
The module’s **first segment** (before the first `.`) must match the folder name placed under `/bundles`.

### Minimal runtime descriptor schema (generated)

```json
{
  "default_bundle_id": "app@1-0",
  "bundles": {
    "app@1.0.0": {
      "id": "app@1-0",
      "name": "Demo App",
      "path": "/bundles",
      "module": "app@1-0.entrypoint",
      "read_only": false,
      "git_url": "git@github.com:org/customer-repo.git",
      "git_ref": "bundle-v2026.02.22",
      "git_subdir": "apps/bundles/app@1-0"
    }
  }
}
```

### How CI uses the release file (baked image)

1. Read `release.yaml`.
2. For each bundle entry, checkout `repo@ref` and copy:
   - `<subdir>/<bundle_dir>` → `/bundles/<module_first_segment>` inside the image.
3. Generate a **runtime descriptor** (JSON) that points to `/bundles` and uses the per‑bundle `module`.
4. Provide it at runtime:
   - Option A: bake `registry.json` into the image and set  
     `AGENTIC_BUNDLES_JSON=/bundles/registry.json`
   - Option B: inject JSON directly in the task definition env.

In both cases, the **release file** is the source of truth; the descriptor is derived.

This prevents image rebuilds on every commit while still baking bundles for ECS.

---

## 5) Environment Files (Compose‑compatible)

We now split env per component:

- `.env` (compose paths + build contexts)
- `.env.ingress`
- `.env.proc`
- `.env.metrics`
- `.env.postgres.setup`
- `.env.proxylogin` (optional)
- `.env.frontend` (optional)

**Important:** Use the **same `GATEWAY_CONFIG_JSON`** for ingress/proc/metrics (tenant/project is shared).

---

## 6) EC2 Docker‑Compose (Transition Phase)

**Current flow (recommended while migrating):**

1. CI builds images and pushes to registry.
2. Ops updates `.env.*` files on EC2.
3. Compose pulls new images.

Example command sequence:

```bash
docker compose -f docker-compose-decentralized-infra-data.yaml pull
docker compose -f docker-compose-decentralized-infra-data.yaml up -d
```

**Notes:**
- `chat-ingress` + `chat-proc` are now separate services.
- `metrics` can be enabled without user impact.
- Use `.env.postgres.setup` for `postgres-setup`, not `.env.ingress/.env.proc`.

---

## 7) ECS (Target State)

**Split into services:**

1. **Ingress service**
2. **Processor service**
3. **Metrics service**
4. **Proxy/UI (optional split)**

**Routing (example):**

- `/chatbot/api/*` → ingress
- `/chatbot/api/integrations/*` → processor
- `/metrics/*` (if exposed) → metrics (usually internal)

---

## 8) CI Pipeline (Suggested Structure)

### Platform repo pipeline

- Build + push:
  - `kdcube-chat-ingress`
  - `kdcube-chat-proc`
  - `kdcube-metrics`
  - `kdcube-postgres-setup`
  - `kdcube-web-proxy` (if used)
  - `proxylogin`
- Publish release artifacts (optional)

### Customer repo pipeline

- Build + push:
  - `kdcube-web-ui`
- Build + push (bundles baked into processor):
  - `kdcube-chat-proc-bundled`
- Build + push (bundles baked into exec image):
  - `kdcube-exec-bundled`
- Publish updated nginx config + config.json

---

## 8.1) CI: Generate `AGENTIC_BUNDLES_JSON` from `release.yaml`

`release.yaml` is the source of truth. CI should **derive** the runtime bundle descriptor.

**Pseudocode:**

```text
read release.yaml
for each bundles.items[]:
  checkout repo@ref
  module_first_segment = module.split(".")[0]
  copy <subdir>/<bundle_dir> → /bundles/<module_first_segment>
  add entry to descriptor: path=/bundles, module, read_only, git_url/git_ref/git_subdir
write descriptor to /bundles/registry.json (or inject as env JSON)
```

**Minimal Python-like example:**

```python
descriptor = {"default_bundle_id": rel["bundles"]["default_bundle_id"], "bundles": {}}
for b in rel["bundles"]["items"]:
    module_first = b["module"].split(".")[0]
    dst = f"/bundles/{module_first}"
    descriptor["bundles"][b["id"]] = {
        "id": b["id"],
        "name": b.get("name", b["id"]),
        "path": "/bundles",
        "module": b["module"],
        "read_only": b.get("read_only", False),
        "git_url": b["repo"],
        "git_ref": b["ref"],
        "git_subdir": f"{b['subdir']}/{b['bundle_dir']}",
    }
```

**Runtime injection options:**
- Bake `/bundles/registry.json` into the image.
- Or inject JSON into `AGENTIC_BUNDLES_JSON` env at deploy time.

---

## 8.2) CD: Where Env Vars Live (Ingress / Proc / Metrics)

**ECS (recommended):**
- Each service has its own **task definition** with env vars + secrets.
- Use the **same `GATEWAY_CONFIG_JSON`** across ingress/proc/metrics.
- Inject `AGENTIC_BUNDLES_JSON` **only** for the processor service.

**Compose (transition):**
- `.env.ingress`, `.env.proc`, `.env.metrics`, `.env.postgres.setup`
- Compose file references each env file explicitly.

---

## 8.3) Frontend Config (Runtime)

The frontend requires a **runtime config file** (e.g. `config.json`).
This file must be present **inside the UI container at runtime**.

Options:
1. **Bake into the UI image** during build (Dockerfile copies it).
2. **Mount at runtime** (volume/config/sidecar) in ECS.

See the customer Dockerfile and config example:
- `.../ops/.../Dockerfile_UI`
- `.../ops/.../prod/config.json`

For ECS, the recommended approach is to bake `config.json` into the UI image
or provide it via a mounted volume at the path expected by the UI build.

---

## 9) Bundles in Customer Repo (No Git Bundles Yet)

Because Git‑based bundle fetching is not finished yet, **bundles must be baked into a processor image** for ECS.

Recommended approach:

1. Use the platform `kdcube-chat-proc` image as the base.
2. Copy bundles from the customer repo into `/bundles`.
3. Set `AGENTIC_BUNDLES_ROOT=/bundles`.

Example Dockerfile (customer repo):

```dockerfile
FROM <registry>/kdcube-chat-proc:<version>

WORKDIR /app

# Copy bundles into container
COPY ./bundles /bundles

ENV AGENTIC_BUNDLES_ROOT=/bundles
```

This produces `kdcube-chat-proc-bundled`, which ECS can run without host mounts.

### Bundle Descriptor Ownership + Versioning (Now)

**Owner:** the customer repo owns the **bundle descriptor** (`AGENTIC_BUNDLES_JSON`) and controls bundle versions.  
**Platform repo** only provides the loader + runtime.

**Recommended (today):**

- Put the bundle descriptor in the **processor env** (`AGENTIC_BUNDLES_JSON`).
- Keep `path` pointing to `/bundles/...` (because bundles are baked).
- Also include **git metadata** for traceability (`git_url`, `git_ref`, `git_subdir`).

Example entry (works today, tracks version):

```json
{
  "default_bundle_id": "my-bundle@2.0.0",
  "bundles": {
    "my-bundle@2.0.0": {
      "id": "my-bundle@2.0.0",
      "name": "My Bundle",
      "path": "/bundles/my-bundle@2.0.0",
      "module": "my_bundle.entrypoint",
      "git_url": "https://git.example.com/org/my-bundle.git",
      "git_ref": "v2.0.0",
      "git_subdir": "bundle"
    }
  }
}
```

**How version is defined today:**

- The bundle ID (e.g. `my-bundle@2.0.0`) is the visible version.
- `git_ref` is the **true source version** (tag or commit SHA).
- The baked processor image must contain the matching `/bundles/...` folder.

**Important (until Git bundles are ready):**

- Keep `git_url/git_ref` for traceability **only**.
- Set `BUNDLE_GIT_RESOLUTION_ENABLED=0` to prevent git cloning on startup.

**Apply env bundles on deploy (no admin auth):**

- Set `BUNDLES_FORCE_ENV_ON_STARTUP=1` in proc env for the deploy.
- Restart the processor → Redis registry is overwritten from env.

### Code‑Exec Image (Docker/Fargate)

The processor spawns the **code‑exec** image on demand. The exec image needs access to bundle tools/code.

**Current (no Git bundles):**

- Build a **bundled exec image**:
  - Base: `py-code-exec` (platform)
  - Copy bundles into `/bundles`
  - Set `PY_CODE_EXEC_IMAGE` to this bundled tag

Example Dockerfile (customer repo):

```dockerfile
FROM <registry>/py-code-exec:<version>
COPY ./bundles /bundles
ENV AGENTIC_BUNDLES_ROOT=/bundles
```

**Fargate path:**  
The processor uploads bundles to S3 before starting Fargate exec. This still requires bundles to be present locally in the processor (hence bundled proc image remains needed until Git bundles).

**Note (upcoming):**  
We are adding **bundle‑from‑git** support. Once enabled, you will no longer need a fat processor image.  
Bundles will be configured via env/admin (bundle descriptor) and fetched at runtime.

**Bundle‑from‑git checklist (future):**
- Stop building `kdcube-chat-proc-bundled`
- Stop building `kdcube-exec-bundled`
- Set bundle git descriptors in env/admin
- Ensure processor can fetch bundles at runtime

---

## 10) Exec Flow (Diagram)

```mermaid
flowchart LR
  subgraph Processor
    P[kdcube-chat-proc-bundled]
  end

  subgraph Exec
    E[py-code-exec or kdcube-exec-bundled]
  end

  subgraph Storage
    S3[(S3 bundles snapshot)]
  end

  P -->|local bundles /bundles| E
  P -->|Fargate: upload bundles| S3 -->|download| E

  classDef emphasis fill:#f7f7f7,stroke:#444,stroke-width:1px;
  class P,E,S3 emphasis;
```

---

## 11) Minimal Inputs Required From Ops

- Container registry (ECR) names
- IAM policies for CI push
- ECS cluster + VPC + subnets
- ALB routing rules
- Secrets storage (SSM or Secrets Manager)

---

## 12) Common Failure Points (Checklist)

- `GATEWAY_CONFIG_JSON` missing tenant/project
- Redis URL mismatch between ingress/proc/metrics
- Postgres `max_connections` too low for processor concurrency
- ALB idle timeout too low for SSE
- Missing host mount for `/exec-workspace` when running locally

---

## 13) Next Steps

1. Align image names + ECR repos.
2. Decide ECS task layout (1 service per component).
3. Migrate compose → ECS one service at a time.
4. Enable metrics export + autoscaling rules.
