---
id: ks:docs/service/cicd/cli-README.md
title: "CLI (kdcube)"
summary: "CLI design for local env bootstrapping, compose setup, and release descriptor validation."
tags: ["service", "cicd", "cli", "env", "deployment"]
keywords: ["kdcube cli", "env init", "docker compose", "local dev", "release.yaml"]
see_also:
  - ks:docs/service/cicd/release-README.md
  - ks:docs/service/cicd/release-descriptor-README.md
  - ks:docs/service/environment/setup-dev-env-README.md
  - ks:docs/service/environment/setup-for-dockercompose-README.md
---
# KDCube CLI (Design)

This document defines the **initial CLI surface** and behavior. The CLI is for:

- **Platform developers** running services on host (PyCharm/IntelliJ or shell).
- **Compose users** running the all‑in‑one stack.
- **Release tooling** (validate and render release descriptors).

CLI root (code): `services/kdcube-ai-app/kdcube_cli`

---

## 1) Immediate use cases

1) **Generate local env files (platform dev)**
   - Create `.env` files in service locations
   - Create required local directories
   - Merge with existing `.env` values

2) **Generate compose env files (all‑in‑one)**
   - Produce `.env.*` files in `deployment/docker/all_in_one_kdcube`
   - Create data folders

3) **Validate release descriptor**
   - Validate schema + refs

4) **Render bundle registry**
   - Convert release descriptor bundles → `AGENTIC_BUNDLES_JSON`

---

## 2) Commands (initial)

### 2.1 `kdcube env init` (platform dev, host)

**Goal:** Create envs for running ingress/proc on host + infra via `local-infra-stack`.

```
kdcube env init \
  --mode dev-host \
  --repo /path/to/kdcube-ai-app
```

**Sources:**
- `deployment/docker/local-infra-stack/sample_env/*`

**Targets (default paths):**
- `deployment/docker/local-infra-stack/.env`
- `deployment/docker/local-infra-stack/.env.postgres.setup`
- `deployment/docker/local-infra-stack/.env.proxylogin`
- `services/kdcube-ai-app/kdcube_ai_app/apps/chat/api/.env.ingress`
- `services/kdcube-ai-app/kdcube_ai_app/apps/chat/proc/.env.proc`
- `services/kdcube-ai-app/kdcube_ai_app/apps/metrics/.env.metrics`
- `ui/chat-web-app/public/private/config.hardcoded.json` (generated UI config)

**Folders created (default set):**
- `deployment/docker/local-infra-stack/data/postgres`
- `deployment/docker/local-infra-stack/data/redis`
- `deployment/docker/local-infra-stack/data/clamav-db`
- `deployment/docker/local-infra-stack/data/bundles`
- `deployment/docker/local-infra-stack/data/bundle-local-storage`
- `deployment/docker/local-infra-stack/data/exec-workspace`
- `deployment/docker/local-infra-stack/logs`

### 2.2 `kdcube env init` (compose)

```
kdcube env init \
  --mode compose \
  --repo /path/to/kdcube-ai-app
```

**Sources:**
- `deployment/docker/all_in_one_kdcube/sample_env/*`

**Targets:**
- `<workdir>/config/.env`
- `<workdir>/config/.env.ingress`
- `<workdir>/config/.env.proc`
- `<workdir>/config/.env.metrics`
- `<workdir>/config/.env.postgres.setup`
- `<workdir>/config/.env.proxylogin` (optional)
- `<workdir>/config/frontend.config.hardcoded.json`

**Folders created:**
- `<workdir>/data/*` (same subfolders as above)
- `<workdir>/logs/*`

### 2.3 `kdcube release validate`

```
kdcube release validate --file release.yaml
```

Validates release descriptor schema and prints errors with line numbers.

### 2.4 `kdcube release render-bundles`

```
kdcube release render-bundles --file release.yaml --out bundles.json
```

Renders `bundles.items` to a runtime registry payload for `AGENTIC_BUNDLES_JSON`.

---

## 3) Env merge semantics

The CLI **never overwrites existing values** by default.

Rules (default):
1) If a key exists in target `.env` and is non‑empty → **keep**.
2) If a key exists but is empty → fill from template if available.
3) If a key is missing → add from template.

**Secrets are never printed**, but the CLI reports missing values as:
```
MISSING (secret)
```

### 3.1 Update mode (explicit)

To overwrite existing values, use:

```
kdcube env init --mode dev-host --repo ... --update
```

`--update` will:
- overwrite non‑secret values
- **never** overwrite secrets unless `--update-secrets` is explicitly provided

---

## 4) Secret handling (default)

Keys treated as secrets by default (pattern‑based):
- `*_SECRET`, `*_TOKEN`, `*_KEY`, `*_PASSWORD`
- `AWS_*`, `OPENAI_*`, `ANTHROPIC_*`, `STRIPE_*`

Secrets are written to env files if provided by templates or overrides, but **never printed**.

---

## 5) Overrides

You can override any value:

```
kdcube env init --mode dev-host --repo ... \
  --set EXEC_WORKSPACE_ROOT=/path/to/exec \
  --set AGENTIC_BUNDLES_ROOT=/bundles
```

Overrides apply **after** merge rules.

---

## 6) Sample bundles (local only)

If `release.yaml` does not contain bundles, the CLI can seed sample bundles:

```
kdcube bundles seed --preset samples
```

This is intended for **local development** only.

---

## 7) Future commands (next phase)

- `kdcube doctor` (validate env + filesystem + runtime dependencies)
- `kdcube compose up` (wrapper around docker compose)
- `kdcube release tag` (tag + VERSION validation)
