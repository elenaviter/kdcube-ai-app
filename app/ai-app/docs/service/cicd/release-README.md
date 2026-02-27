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

