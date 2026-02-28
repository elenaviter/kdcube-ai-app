# Release + Versioning

We use **one unified version** for the monorepo (platform + SDK) until the SDK is split.

---

## 1) Version Source

Root file:

```
VERSION
```

Example:

```
0.1.0
```

---

## 2) Tagging Convention

- Release tags: `vX.Y.Z`
- Example: `v0.1.0`

---

## 3) Image Tags

When building images, publish:

- `:vX.Y.Z` (semver)
- `:git-sha` (immutable)

Example:

```
kdcube-chat-ingress:v0.1.0
kdcube-chat-ingress:8f3c9e1
```

---

## 4) Release Process (Manual)

1. Update `VERSION` (e.g. `0.1.1`)
2. Create a git tag:
   ```
   git tag v0.1.1
   git push origin v0.1.1
   ```
3. CI builds + publishes images

---

## 4.1) Bundle registry during release

The processor consumes a **runtime bundle descriptor** (`AGENTIC_BUNDLES_JSON`).  
During release, ensure one of these:

- **Baked bundles:** CI generates `AGENTIC_BUNDLES_JSON` (path `/bundles/...`) and injects it into proc.
- **Git bundles:** CI injects `AGENTIC_BUNDLES_JSON` with `repo/ref/subdir`.

If you need to **override existing Redis registry**, deploy proc with:

```
BUNDLES_FORCE_ENV_ON_STARTUP=1
```

Then turn it off after rollout.

For git bundles in multi‑replica deployments, enable:

```
BUNDLE_GIT_REDIS_LOCK=1
```

This makes each replica pull **once** on startup (no cross‑replica contention).

---

## 5) Release Process (CI)

Suggested CI logic:

- Read `VERSION`
- If tag `vX.Y.Z` exists, use that for image tags
- Always add `:git-sha`

---

## 6) When to Split SDK Versioning

Only split when:

- SDK cadence diverges from platform
- Compatibility matrix is required

Until then, **keep unified versioning**.
