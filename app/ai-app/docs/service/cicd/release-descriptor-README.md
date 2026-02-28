# Release Descriptor (release.yaml)

**Where it lives:**
- Customer repo (closed) as `release.yaml`

The CI pipeline reads this file and derives the runtime descriptor
(`AGENTIC_BUNDLES_JSON`) and bundle packaging.  
You can also mount this file directly and point `AGENTIC_BUNDLES_JSON` to it
(runtime loader will read the `bundles` section).

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
- `ref`: git tag/branch/commit.
- `subdir`: path **inside repo** to the bundles root (parent folder).
- `module`: module **relative to `subdir`** (for example `id.entrypoint`).

---

## 2) How CI derives runtime descriptor

CI turns `release.yaml` into `AGENTIC_BUNDLES_JSON`.

**No translation needed:** runtime accepts the same field names
(`repo/ref/subdir/module`) as the release descriptor.

Two delivery modes:

### A) Baked bundles (copy into image)

1. Checkout `repo@ref`
2. Copy `<subdir>/<id>` into image at:
   ```
   /bundles/<id>
   ```
3. Generate runtime descriptor with:
   - `path=/bundles`
   - `module=<id>.entrypoint`

Example runtime entry:

```json
{
  "id": "react@2026-02-10-02-44",
  "path": "/bundles",
  "module": "react@2026-02-10-02-44.entrypoint"
}
```

### B) Git‑defined bundles (clone at runtime)

CI emits:

```json
{
  "id": "react@2026-02-10-02-44",
  "repo": "git@github.com:kdcube/kdcube-ai-app.git",
  "ref": "v0.3.2",
  "subdir": "app/ai-app/services/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles",
  "module": "react@2026-02-10-02-44.entrypoint"
}
```

**Note:** `subdir` points to the **parent bundles folder** so `module` includes the bundle folder name.

---

## 3) Module & subdir rules (important)

Use a **single convention** for both local and git bundles:

- `subdir` points to the **parent bundles folder**
- `module` is `<id>.entrypoint`

This matches the local path convention:
`path=/bundles` + `module=<id>.entrypoint`.

---

## 4) Release ownership

**Who edits what:**
- Platform team → `VERSION` + platform tag
- Customer bundle team → bundle code + bundle tag
- Customer frontend team → UI code + UI tag
- Release manager → `release.yaml` (single source of truth)

---

## 5) Runtime env

**Proc requires:**

- `AGENTIC_BUNDLES_JSON` (derived from this file)
- `BUNDLES_FORCE_ENV_ON_STARTUP=1` (only for a single rollout if you need to overwrite Redis)
