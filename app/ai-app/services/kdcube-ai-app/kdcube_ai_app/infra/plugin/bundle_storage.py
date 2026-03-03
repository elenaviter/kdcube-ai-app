# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

from __future__ import annotations

import os
import pathlib
from typing import Optional

from kdcube_ai_app.infra.plugin.git_bundle import resolve_bundles_root


def _sanitize_segment(raw: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "-" for ch in (raw or ""))
    safe = safe.strip("-_")
    return safe or "default"


def resolve_bundle_storage_root() -> pathlib.Path:
    """
    Resolve the shared storage root for bundle-managed assets (indexes, repos, etc).
    Prefer explicit env, otherwise default under bundles root.
    """
    root = os.environ.get("BUNDLE_SHARED_STORAGE_ROOT") or os.environ.get("BUNDLE_STORAGE_ROOT")
    if root:
        # Accept plain filesystem paths. If a file:// URI is provided, normalize to path.
        if isinstance(root, str) and root.startswith("file://"):
            try:
                from urllib.parse import urlparse
                parsed = urlparse(root)
                if parsed.scheme == "file" and parsed.path:
                    return pathlib.Path(parsed.path).expanduser().resolve()
            except Exception:
                pass
        return pathlib.Path(root).expanduser().resolve()
    bundles_root = resolve_bundles_root()
    return (bundles_root / "_bundle_storage").resolve()


def bundle_storage_dir(
    *,
    bundle_id: str,
    version: Optional[str] = None,
    tenant: Optional[str] = None,
    project: Optional[str] = None,
    ensure: bool = True,
) -> pathlib.Path:
    """
    Compute a per-bundle storage directory.
    If tenant/project are provided, isolate under <tenant>/<project>/...
    If version/git_commit/ref is provided, append it to avoid stale data reuse.
    """
    root = resolve_bundle_storage_root()
    parts = []
    if tenant:
        parts.append(_sanitize_segment(tenant))
    if project:
        parts.append(_sanitize_segment(project))
    bid = _sanitize_segment(bundle_id or "bundle")
    if version:
        parts.append(f"{bid}__{_sanitize_segment(version)}")
    else:
        parts.append(bid)
    storage = root.joinpath(*parts).resolve()
    if ensure:
        storage.mkdir(parents=True, exist_ok=True)
    return storage


def storage_for_spec(
    *,
    spec: Optional[object],
    tenant: Optional[str] = None,
    project: Optional[str] = None,
    ensure: bool = True,
) -> Optional[pathlib.Path]:
    """
    Convenience helper that accepts a BundleSpec-like object.
    Expects attributes: id, version, git_commit, ref.
    """
    if spec is None:
        return None
    bundle_id = getattr(spec, "id", None) or ""
    # Prefer git_commit/ref for stable, user-controlled versioning.
    # Fall back to spec.version only if neither is present.
    version = (
        getattr(spec, "git_commit", None)
        or getattr(spec, "ref", None)
        or getattr(spec, "version", None)
    )
    if not bundle_id:
        return None
    return bundle_storage_dir(
        bundle_id=bundle_id,
        version=version,
        tenant=tenant,
        project=project,
        ensure=ensure,
    )
