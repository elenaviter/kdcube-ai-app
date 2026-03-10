---
id: ks:docs/service/cicd/assembly-descriptor-README.md
title: "Assembly Descriptor"
summary: "Assembly descriptor schema and runtime usage: platform/frontend refs, bundle items, module/subdir rules, and props resolution."
tags: ["service", "cicd", "assembly", "descriptor", "schema", "bundles", "props", "git"]
keywords: ["assembly.yaml", "release.yaml", "bundles.items", "default_bundle_id", "repo", "ref", "subdir", "module", "AGENTIC_BUNDLES_JSON", "env:", "file:"]
see_also:
  - ks:docs/service/cicd/release-bundle-README.md
  - ks:docs/service/cicd/custom-cicd-README.md
  - ks:docs/service/cicd/release-README.md
  - ks:docs/sdk/bundle/bundle-ops-README.md
  - ks:docs/service/cicd/secrets-descriptor-README.md
---
# Assembly Descriptor (assembly.yaml)

**Where it lives:**
- Assembly file (recommended name: `assembly.yaml`)
- Default template: [`app/ai-app/deployment/assembly.yaml`](../../../deployment/assembly.yaml) (copied into the workdir as `config/assembly.yaml`)

**Filename:** the CLI accepts any path; `assembly.yaml` is the recommended name.
If you do not provide a path, the wizard uses `config/assembly.yaml` in the workdir
and seeds it from `deployment/assembly.yaml`. If you provide a path, the CLI copies
it into `config/assembly.yaml` and uses that file as the source of truth.
Older setups may still use `release.yaml`; rename it or update
`AGENTIC_BUNDLES_JSON` to point at `/config/assembly.yaml`.

The CI pipeline reads this file and uses it for bundle packaging.  
Runtime can also read it directly: mount `assembly.yaml` and point `AGENTIC_BUNDLES_JSON` to it.  
Runtime **only uses the `bundles` section** and ignores `platform`/`frontend`.  

The **CLI** can additionally use the `frontend` section to build and run a
custom UI in the **custom‑ui‑managed‑infra** compose mode.
During setup, the wizard lets you choose whether the descriptor applies to
**bundles**, **frontend**, and/or **platform**.

Sensitive values (LLM keys, tokens, passwords) should live in `secrets.yaml`
instead of `assembly.yaml`. See: [docs/service/cicd/secrets-descriptor-README.md](secrets-descriptor-README.md)

Bundle items may include **props** (runtime overrides).  
Props are **resolved** when the descriptor is applied to Redis.

---

## 1) Descriptor schema (recommended)

```yaml
release_name: "prod-2026-02-22"

platform:
  repo: "kdcube-ai-app"
  ref: "v0.3.2"          # tag or commit

frontend:
  build:
    repo: "private-ui-repo"
    ref: "ui-v2026.02.22"  # tag or commit
    dockerfile: "ops/docker/Dockerfile_UI"
    src: "ui/chat-web-app"
  image: "registry/private-ui:2026.02.22"         # optional; if set, CLI uses this image
  frontend_config: "ops/docker/config.cognito.json"
  nginx_ui_config: "ops/docker/nginx_ui.conf"     # optional
  ui_env_build_relative: "ui/chat-web-app/.env.ui.build"  # optional
  domain: "chat.example.com"                      # optional

domain: "chat.example.com"                        # optional root domain (used for proxylogin URLs)

auth:
  type: "delegated"  # "simple" | "cognito" | "delegated"
  cognito:
    region: "eu-west-1"
    user_pool_id: "eu-west-1_AbCdEf123"
    app_client_id: "4m7u93umj6net8kng4cbe2gukn"
    service_client_id: "service-client-id"        # optional
  proxy_login:
    redis_key_prefix: "proxylogin:<TENANT>:<PROJECT>:"
    token_masquerade: false
    password_reset:
      company: "YourCompany"
      sender: "noreply@yourcompany.com"
      template_name: "YourCompanyPasswordResetTemplate"
      redirect_url: "http://localhost:5174/chatbot/reset-password?user=%[1]s"
    http_urlbase: "http://localhost:5174/auth"

proxy:
  ssl: false                # when true, CLI picks SSL nginx templates
  route_prefix: "/chatbot"  # overrides frontend routesPrefix + nginx routing prefix

ports:
  ingress: "8010"
  proc: "8020"
  metrics: "8090"
  ui: "80"
  ui_ssl: "443"

context:
  tenant: "demo-tenant"
  project: "demo-project"

infra:
  postgres:
    user: "postgres"
    password: "postgres"
    database: "kdcube"
    host: "postgres-db"
    port: "5432"
  redis:
    password: "redispass"
    host: "redis"
    port: "6379"

paths:
  host_kdcube_storage_path: "/srv/kdcube/data/kdcube-storage"
  host_bundles_path: "/srv/kdcube/data/bundles"
  host_bundle_storage_path: "/srv/kdcube/data/bundle-storage"
  host_exec_workspace_path: "/srv/kdcube/data/exec-workspace"

bundles:
  default_bundle_id: "react@2026-02-10-02-44"
  items:
    - id: "react@2026-02-10-02-44"
      name: "ReAct (example)"
      repo: "git@github.com:kdcube/kdcube-ai-app.git"
      ref: "v0.3.2"   # tag or commit
      subdir: "app/ai-app/services/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles"
      module: "react@2026-02-10-02-44.entrypoint"
      props:
        knowledge:
          repo: "git@github.com:kdcube/kdcube-ai-app.git"
          ref: "v0.3.2"
          docs_root: "app/ai-app/docs"
          src_root: "app/ai-app/services/kdcube-ai-app/kdcube_ai_app"
          deploy_root: "app/ai-app/deployment"
          validate_refs: true

    - id: "app@2-0"
      name: "Customer App"
      repo: "git@github.com:org/customer-repo.git"
      ref: "bundle-v2026.02.22"
      subdir: "service/bundles"
      module: "app@2-0.entrypoint"
```

**Fields (bundle item):**

- `id`: bundle id (versioned id).
- `repo`: git repo URL.
- `ref`: git tag or commit SHA.  
  **Branch names are only for local/dev** and require `BUNDLE_GIT_ALWAYS_PULL=1`.
- `subdir`: path **inside repo** to the bundles root (parent folder).
- `module`: module **relative to `subdir`** (for example `id.entrypoint`).
- `props` (optional): runtime props resolved at deployment time.

**Ref resolution (for string values):**
- `env:NAME` → environment variable `NAME`
- `file:/path/to/secret` → file contents

Any string **without** these prefixes is treated as a literal value.

Resolved values are written into Redis as runtime props.

### Frontend section (CLI usage)
The `frontend` section is used by `kdcube-setup` to build a customer UI when
you run **custom‑ui‑managed‑infra** compose. It is ignored by runtime services.

Required (build mode):
- `build.repo`: frontend git repo (SSH or HTTPS)
- `build.ref`: git tag or commit (or branch for dev)
- `build.dockerfile`: path to Dockerfile_UI (relative to repo root)
- `build.src`: path to UI source (relative to repo root)
- `frontend_config`: path to UI config template (relative to repo root)

Alternative (image mode):
- `image`: prebuilt UI image to use (if set, the CLI skips building `web-ui`).
  If both `build` and `image` are present, the image takes precedence.

Optional:
- `nginx_ui_config`: path to UI nginx config (relative to repo root)
- `ui_env_build_relative`: build‑time env file path (relative to repo root)
- `frontend.domain`: deployment domain (informational for now)
- `domain`: root domain used when expanding `YOUR_DOMAIN` in proxylogin URLs
- `paths.*`: local-only host path overrides. The CLI will write these into the
  workdir copy of the descriptor when needed; they are not required in the
  public template.

**CLI behavior:**
- If `frontend.build` is provided, clones the frontend repo into
  `~/.kdcube/kdcube-runtime/frontend` (if needed).
- If `frontend.image` is set, uses that image for `web-ui` and skips building it.
- Sets `UI_BUILD_CONTEXT`, `UI_DOCKERFILE_PATH`, `UI_SOURCE_PATH`, and related
  env values in the workdir.
- Uses `frontend_config` as the template for `config.json`, then patches tenant,
  project, and auth values.
  If no frontend repo is cloned, the `frontend_config` path is resolved relative
  to the descriptor location on disk.
- If `platform.ref` is present, the CLI offers **assembly-descriptor** as an
  install source (pulls that tag from DockerHub).

### Auth section (CLI usage)
The `auth` section lets you predefine authentication for the wizard.
If `auth.type` is `cognito` or `delegated`, the CLI will ask whether to
use the Cognito settings from the descriptor or re-enter them.

Supported keys:
- `auth.type`: `simple`, `cognito`, or `delegated`
- `auth.cognito.region`
- `auth.cognito.user_pool_id` (legacy aliases are normalized on save)
- `auth.cognito.app_client_id` (legacy aliases are normalized on save)
- `auth.cognito.service_client_id` (legacy aliases are normalized on save)
- `auth.proxy_login.redis_key_prefix` (tenant/project placeholders are expanded)
- `auth.proxy_login.token_masquerade`
- `auth.proxy_login.password_reset.company`
- `auth.proxy_login.password_reset.sender`
- `auth.proxy_login.password_reset.template_name`
- `auth.proxy_login.password_reset.redirect_url`
- `auth.proxy_login.http_urlbase`

If `domain` is set at the root of the descriptor, the CLI uses it to expand
`YOUR_DOMAIN` placeholders in proxylogin URLs. If not set, it falls back to
`localhost` (and includes the UI port if it’s not 80/443).

### Context / infra / paths (CLI usage)
When you run the wizard with an assembly descriptor, it will:
- Use `context.tenant` and `context.project` as defaults for prompts.
- Use `infra.postgres` and `infra.redis` values as defaults.
- Use `paths.*` as defaults for local host paths.
- Write back any values you enter, keeping `assembly.yaml` as the source of truth.

For infra, the wizard will explicitly ask whether to use Postgres/Redis settings
from the descriptor or override them. If you choose to use the descriptor, it
will not prompt for those values.

These sections are used by the CLI to **render .env files** for docker-compose.

---

## 2) Module & subdir rules (important)

Use a **single convention** for both local and git bundles:

- `subdir` points to the **parent bundles folder**
- `module` is `<id>.entrypoint`

This matches the local path convention:
`path=/bundles` + `module=<id>.entrypoint`.

---

## 3) Release ownership

**Who edits what:**
- Platform team → `VERSION` + platform tag
- Customer bundle team → bundle code + bundle tag
- Customer frontend team → UI code + UI tag
- Release manager → `assembly.yaml` (single source of truth)

---

## 4) Runtime env

**Proc requires:**

- `AGENTIC_BUNDLES_JSON` (this file path or inline JSON)
- `BUNDLES_FORCE_ENV_ON_STARTUP=1` (only for a single rollout if you need to overwrite Redis)

**Props application:**
- Props in `assembly.yaml` are applied when the descriptor is used to seed or reset Redis.
- If you do **not** reset Redis, existing runtime props remain unchanged.
