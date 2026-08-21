# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""Resolve workspace sources with the active turn's trusted context.

The framework-neutral materialization and checkout engines accept a source
resolver callback. This bridge binds that callback to the ContextBrowser and
event-source registry carried by an in-process agent adapter. Namespace owners
retain authorization and storage semantics; the harness receives only a pinned
conversation-file ref and the local materialization it may copy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.layout import (
    artifact_outdir_for,
)
from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.materialization import (
    MaterializedWorkspaceSource,
    canonical_logical_ref,
    canonical_workspace_path_for_ref,
)
from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.references import (
    CONVERSATION_FILE_REF_PREFIX,
    physical_path_to_logical_path,
)


def _artifact_local_path(artifact_root: Path, physical_path: str) -> Path:
    candidate = artifact_root / str(physical_path or "").strip()
    try:
        candidate.resolve().relative_to(artifact_root.resolve())
    except ValueError as error:
        raise ValueError("resolved workspace source escapes the artifact root") from error
    return candidate


def _materialized_row(
    result: Mapping[str, Any],
    *,
    object_ref: str,
) -> dict[str, Any]:
    rows = [
        row
        for row in (result.get("materialized") or [])
        if isinstance(row, Mapping)
    ]
    for row in rows:
        if str(row.get("object_ref") or row.get("source_ref") or "").strip() == object_ref:
            return dict(row)
    if rows:
        return dict(rows[0])

    rehosted = [
        str(path or "").strip()
        for path in (result.get("rehosted") or [])
        if str(path or "").strip()
    ]
    if rehosted:
        physical_path = rehosted[0]
        return {
            "object_ref": object_ref,
            "physical_path": physical_path,
            "logical_path": physical_path_to_logical_path(physical_path),
        }

    detail = (
        result.get("errors")
        or result.get("missing")
        or result.get("invalid")
        or "not materialized"
    )
    raise FileNotFoundError(f"could not materialize {object_ref}: {detail}")


async def resolve_context_workspace_source(
    *,
    ref: str,
    staging_dir: Path,
    ctx_browser: Any,
    outdir: Path,
    state: Optional[dict[str, Any]] = None,
    tool_id: str = "workspace.checkout",
    tool_call_id: str = "",
) -> MaterializedWorkspaceSource:
    """Resolve one object locator under the active turn's trusted authority."""
    del staging_dir  # Existing context resolvers materialize below artifact outdir.
    raw = str(ref or "").strip()
    runtime_ctx = getattr(ctx_browser, "runtime_ctx", None)
    current_conversation_id = str(
        getattr(runtime_ctx, "conversation_id", "") or ""
    ).strip()
    artifact_root = artifact_outdir_for(Path(outdir))

    if raw.startswith(CONVERSATION_FILE_REF_PREFIX):
        # The storage implementations predate the harness extraction. This is
        # their shared adapter seam; model-facing tools no longer call them
        # independently.
        from kdcube_ai_app.apps.chat.sdk.solutions.react.workspace import (
            hydrate_workspace_paths,
        )

        physical_path = canonical_workspace_path_for_ref(
            raw,
            current_conversation_id=current_conversation_id,
        )
        result = await hydrate_workspace_paths(
            ctx_browser=ctx_browser,
            paths=[physical_path],
            outdir=Path(outdir),
        )
        local_path = _artifact_local_path(artifact_root, physical_path)
        if not local_path.exists():
            detail = result.get("errors") or result.get("missing") or "not materialized"
            raise FileNotFoundError(f"could not materialize {raw}: {detail}")
        return MaterializedWorkspaceSource(
            requested_ref=raw,
            resolved_ref=canonical_logical_ref(
                raw,
                current_conversation_id=current_conversation_id,
            ),
            local_path=local_path,
            kind="directory" if local_path.is_dir() else "file",
        )

    namespace = raw.partition(":")[0].strip() if ":" in raw else ""
    event_sources = getattr(runtime_ctx, "event_sources", None)
    rehoster = (
        getattr(event_sources, "namespace_rehoster", lambda _namespace: None)(
            namespace
        )
        if namespace and event_sources is not None
        else None
    )
    if rehoster is None:
        raise ValueError(
            "no authorized namespace rehoster is registered for "
            f"{namespace or '<none>'}"
        )

    result = await event_sources.rehost_namespace_ref(
        raw,
        ctx_browser=ctx_browser,
        outdir=Path(outdir),
        state=state or {},
        tool_id=tool_id,
        tool_call_id=tool_call_id,
    )
    row = _materialized_row(result, object_ref=raw)
    physical_path = str(row.get("physical_path") or "").strip()
    logical_path = str(row.get("logical_path") or "").strip()
    if not physical_path and logical_path:
        physical_path = canonical_workspace_path_for_ref(
            logical_path,
            current_conversation_id=current_conversation_id,
        )
    local_path = _artifact_local_path(artifact_root, physical_path)
    if not logical_path or not local_path.exists():
        raise FileNotFoundError(
            f"namespace rehoster returned no usable pinned artifact for {raw}"
        )
    return MaterializedWorkspaceSource(
        requested_ref=raw,
        object_ref=raw,
        resolved_ref=canonical_logical_ref(
            logical_path,
            current_conversation_id=current_conversation_id,
        ),
        local_path=local_path,
        mime=str(row.get("mime") or "").strip(),
        kind="directory" if local_path.is_dir() else "file",
    )
