## Bundle Release Process

This doc describes how to release a bundle and update the **release descriptor**
(`release.yaml`). It applies to both **baked** bundles and **git‑defined** bundles.

---

## 1) Prepare the bundle

1. Update the bundle code.
2. Decide the new **bundle id** (versioned id). Example:  
   `react@2026-02-10-02-44` or `app@2-0`.
3. Ensure the bundle entrypoint contains the agentic decorator (`@agentic_workflow` or factory).

---

## 2) Tag the bundle source

Tag the repo where the bundle lives:

```
git tag <bundle-tag>
git push origin <bundle-tag>
```

Use that tag/commit as `ref` in the release descriptor.

---

## 3) Update release.yaml

Edit `release.yaml` in the **customer repo** and add/update the bundle entry.

### Example (bundle inside monorepo)

```yaml
bundles:
  default_bundle_id: "react@2026-02-10-02-44"
  items:
    - id: "react@2026-02-10-02-44"
      name: "ReAct (example)"
      repo: "git@github.com:kdcube/kdcube-ai-app.git"
      ref: "v0.3.2"
      subdir: "app/ai-app/services/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles"
      module: "react@2026-02-10-02-44.entrypoint"
```

**Rule:** `subdir` points to the **parent bundles directory** and `module` includes the bundle folder
(`id.entrypoint`). Keep `module` aligned with your local path convention.

---

## 4) Decide delivery mode

### A) Baked bundles (copy into image)
- CI copies `<subdir>/<id>` into `/bundles/<id>`.
- Runtime descriptor uses:
  - `path=/bundles`
  - `module=<id>.entrypoint`

### B) Git‑defined bundles (clone at runtime)
- Runtime descriptor uses:
  - `repo`, `ref`, `subdir`, `module`
- Proc must have git enabled and SSH keys (if private).

---

## 5) Deploy with env reset (optional)

If you need to **override existing Redis registry** on deploy:

```
BUNDLES_FORCE_ENV_ON_STARTUP=1
```

Apply for one rollout, then return to `0`.

---

## 6) Validate

- Check the bundle registry in `/admin/integrations/bundles`.
- Run a test chat with `agentic_bundle_id=<your bundle id>`.
- Verify streaming and steps appear in the client.
