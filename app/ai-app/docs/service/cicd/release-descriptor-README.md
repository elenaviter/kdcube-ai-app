---
id: ks:docs/service/cicd/release-descriptor-README.md
title: "Release Descriptor"
summary: "Release descriptor schema and runtime usage: platform/frontend refs, bundle items, module/subdir rules, and props resolution."
tags: ["service", "cicd", "release", "descriptor", "schema", "bundles", "props", "git"]
keywords: ["release.yaml", "bundles.items", "default_bundle_id", "repo", "ref", "subdir", "module", "AGENTIC_BUNDLES_JSON", "env:", "file:"]
see_also:
  - ks:docs/service/cicd/release-bundle-README.md
  - ks:docs/service/cicd/custom-cicd-README.md
  - ks:docs/service/cicd/release-README.md
  - ks:docs/sdk/bundle/bundle-ops-README.md
---
# Release Descriptor (release.yaml)

**Where it lives:**
- Customer repo (closed) as `release.yaml`

The CI pipeline reads this file and uses it for bundle packaging.  
Runtime can also read it directly: mount `release.yaml` and point `AGENTIC_BUNDLES_JSON` to it.  
Runtime **only uses the `bundles` section** and ignores `platform`/`frontend`.

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
  repo: "customer-repo"
  ref: "ui-v2026.02.22"  # tag or commit

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
- Release manager → `release.yaml` (single source of truth)

---

## 4) Runtime env

**Proc requires:**

- `AGENTIC_BUNDLES_JSON` (this file path or inline JSON)
- `BUNDLES_FORCE_ENV_ON_STARTUP=1` (only for a single rollout if you need to overwrite Redis)

**Props application:**
- Props in `release.yaml` are applied when the descriptor is used to seed or reset Redis.
- If you do **not** reset Redis, existing runtime props remain unchanged.
