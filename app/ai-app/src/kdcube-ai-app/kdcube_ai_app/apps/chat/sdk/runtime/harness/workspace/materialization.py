# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""Framework-neutral materialization for the distributed turn workspace.

The harness distinguishes records from objects. Timeline refs such as
``conv:ev:`` are read as records; only a ``conv:fi:`` ref or a provider-owned
object ref resolved by a trusted adapter can become bytes in the workspace.
"""

from __future__ import annotations

import inspect
import mimetypes
import os
import shutil
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Awaitable, Callable, Mapping, Optional

from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.references import (
    ARTIFACT_CONVERSATION_PREFIX,
    ARTIFACT_NAMESPACE_ATTACHMENTS,
    ARTIFACT_NAMESPACE_FILES,
    ARTIFACT_NAMESPACE_PROJECTS,
    ARTIFACT_NAMESPACE_SNAPSHOTS,
    CONVERSATION_FILE_REF_PREFIX,
    build_logical_artifact_path,
    build_physical_artifact_path,
    peel_conversation_prefix,
    qualify_conversation_ref,
    split_logical_artifact_ref,
)


READABLE_RECORD_PREFIXES = (
    "conv:ar:",
    "conv:ev:",
    "conv:so:",
    "conv:su:",
    "conv:tc:",
    "conv:ws:",
)
MATERIALIZATION_TEMP_PREFIX = ".kdcube-materialize-"


@dataclass(frozen=True)
class MaterializedWorkspaceSource:
    """One trusted source made available as a local file or directory."""

    requested_ref: str
    resolved_ref: str
    local_path: Path
    object_ref: str = ""
    mime: str = ""
    kind: str = "file"


WorkspaceSourceResolver = Callable[..., Awaitable[MaterializedWorkspaceSource | Mapping[str, Any] | None]]


class WorkspaceMaterializationError(ValueError):
    """A source cannot safely become a workspace object."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def is_readable_record_ref(ref: str) -> bool:
    raw = str(ref or "").strip()
    return raw.startswith(READABLE_RECORD_PREFIXES)


def is_materializable_ref(ref: str) -> bool:
    raw = str(ref or "").strip()
    return bool(raw and not is_readable_record_ref(raw) and ":" in raw)


def _parse_materialized_fi_ref(ref: str) -> tuple[str, str, str, str]:
    """Parse exact refs and the supported directory-root forms."""
    raw = str(ref or "").strip()
    conversation_id, turn_id, namespace, relpath = split_logical_artifact_ref(raw)
    if turn_id and namespace:
        return conversation_id, turn_id, namespace, relpath.strip("/")
    if not raw.startswith(CONVERSATION_FILE_REF_PREFIX):
        return "", "", "", ""

    prefix, conversation_id, unscoped = peel_conversation_prefix(raw)
    body = unscoped[len(prefix) :] if prefix and unscoped.startswith(prefix) else ""
    for candidate in (
        ARTIFACT_NAMESPACE_PROJECTS,
        ARTIFACT_NAMESPACE_SNAPSHOTS,
        ARTIFACT_NAMESPACE_FILES,
    ):
        marker = f".{candidate}"
        if body.endswith(marker) or body.endswith(marker + "/"):
            turn_id = body[: body.rfind(marker)].strip()
            if turn_id:
                return conversation_id, turn_id, candidate, ""
    return "", "", "", ""


def canonical_workspace_path_for_ref(
    ref: str,
    *,
    current_conversation_id: str = "",
) -> str:
    """Map a materialized ``conv:fi:`` ref to its collision-safe local path."""
    conversation_id, turn_id, namespace, relpath = _parse_materialized_fi_ref(ref)
    if not turn_id or not namespace:
        raise WorkspaceMaterializationError(
            "invalid_materialized_ref",
            f"resolver did not return a canonical conv:fi ref: {ref}",
        )
    source_conversation = conversation_id or str(current_conversation_id or "").strip()
    physical_conversation = (
        source_conversation
        if source_conversation and source_conversation != str(current_conversation_id or "").strip()
        else ""
    )
    if relpath:
        return build_physical_artifact_path(
            turn_id=turn_id,
            namespace=namespace,
            relpath=relpath,
            conversation_id=physical_conversation,
        )
    root = f"{turn_id}/{namespace}"
    return f"{ARTIFACT_CONVERSATION_PREFIX}{physical_conversation}/{root}" if physical_conversation else root


def canonical_logical_ref(
    ref: str,
    *,
    current_conversation_id: str = "",
) -> str:
    raw = str(ref or "").strip()
    return qualify_conversation_ref(raw, str(current_conversation_id or "").strip())


def _safe_stage_name(ref: str) -> str:
    _conversation_id, _turn_id, _namespace, relpath = _parse_materialized_fi_ref(ref)
    raw = PurePosixPath(relpath).name if relpath else "source"
    cleaned = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in raw).strip()
    return cleaned or "source"


def _coerce_resolved_source(
    value: MaterializedWorkspaceSource | Mapping[str, Any],
    *,
    requested_ref: str,
) -> MaterializedWorkspaceSource:
    if isinstance(value, MaterializedWorkspaceSource):
        return value
    if not isinstance(value, Mapping):
        raise WorkspaceMaterializationError(
            "invalid_source_resolver_result",
            "workspace source resolver returned an unsupported result",
        )
    local_path = Path(str(value.get("local_path") or value.get("path") or "").strip())
    resolved_ref = str(
        value.get("resolved_ref")
        or value.get("logical_path")
        or value.get("ref")
        or ""
    ).strip()
    object_ref = str(value.get("object_ref") or value.get("source_ref") or "").strip()
    kind = str(value.get("kind") or "").strip().lower()
    if kind not in {"file", "directory"}:
        kind = "directory" if local_path.is_dir() else "file"
    return MaterializedWorkspaceSource(
        requested_ref=requested_ref,
        resolved_ref=resolved_ref,
        local_path=local_path,
        object_ref=object_ref,
        mime=str(value.get("mime") or "").strip(),
        kind=kind,
    )


async def resolve_workspace_source(
    *,
    ref: str,
    staging_dir: Path,
    tenant: str,
    project: str,
    user_id: str,
    conversation_id: str = "",
    storage_path: Optional[str] = None,
    source_resolver: Optional[WorkspaceSourceResolver] = None,
) -> MaterializedWorkspaceSource:
    """Resolve one object locator through an adapter or the built-in fi reader."""
    raw = str(ref or "").strip()
    if not raw:
        raise WorkspaceMaterializationError("missing_ref", "materialization requires a ref")
    if is_readable_record_ref(raw):
        raise WorkspaceMaterializationError(
            "record_ref_not_materializable",
            f"{raw.split(':', 2)[0]}:{raw.split(':', 2)[1]} identifies a record; read it and use its object_ref or artifact ref",
        )

    staging_dir.mkdir(parents=True, exist_ok=True)
    if source_resolver is not None:
        resolved = source_resolver(ref=raw, staging_dir=staging_dir)
        if inspect.isawaitable(resolved):
            resolved = await resolved
        if resolved is not None:
            source = _coerce_resolved_source(resolved, requested_ref=raw)
            _validate_local_source(source)
            return source

    if not raw.startswith(CONVERSATION_FILE_REF_PREFIX):
        namespace = raw.partition(":")[0] or "<none>"
        raise WorkspaceMaterializationError(
            "source_resolver_required",
            f"no trusted workspace source resolver is registered for namespace {namespace}",
        )

    conversation, turn_id, namespace, relpath = _parse_materialized_fi_ref(raw)
    if not turn_id or not namespace or not relpath:
        raise WorkspaceMaterializationError(
            "directory_resolver_required",
            "directory or root conv:fi refs require an adapter resolver that can enumerate the pinned source",
        )
    from kdcube_ai_app.apps.chat.sdk.runtime.harness.events.resolver import (
        read_event_ref_bytes,
    )

    data, meta = await read_event_ref_bytes(
        ref=raw,
        tenant=tenant,
        project=project,
        user_id=user_id,
        conversation_id=conversation_id,
        storage_path=storage_path,
    )
    target = staging_dir / _safe_stage_name(raw)
    target.write_bytes(bytes(data or b""))
    resolved_ref = build_logical_artifact_path(
        turn_id=str(meta.get("turn_id") or turn_id),
        namespace=str(meta.get("namespace") or namespace),
        relpath=str(meta.get("relpath") or relpath),
        conversation_id=str(meta.get("conversation_id") or conversation or conversation_id),
    )
    return MaterializedWorkspaceSource(
        requested_ref=raw,
        resolved_ref=resolved_ref or raw,
        local_path=target,
        object_ref=raw if not raw.startswith(CONVERSATION_FILE_REF_PREFIX) else "",
        mime=mimetypes.guess_type(target.name)[0] or "application/octet-stream",
        kind="file",
    )


def _validate_local_source(source: MaterializedWorkspaceSource) -> None:
    path = Path(source.local_path)
    if not source.resolved_ref.startswith(CONVERSATION_FILE_REF_PREFIX):
        raise WorkspaceMaterializationError(
            "invalid_source_resolver_result",
            "workspace source resolver must return the pinned conv:fi ref in resolved_ref",
        )
    if not path.exists() or not (path.is_file() or path.is_dir()):
        raise WorkspaceMaterializationError(
            "source_not_materialized",
            f"workspace source resolver returned a missing path: {path}",
        )
    paths = [path]
    if path.is_dir():
        paths.extend(path.rglob("*"))
    if any(candidate.is_symlink() for candidate in paths):
        raise WorkspaceMaterializationError(
            "source_symlink_not_allowed",
            "materialized workspace sources cannot contain symlinks",
        )


def _copy_source(source: Path, target: Path) -> None:
    if source.resolve() == target.resolve():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.copytree(source, target)
    else:
        if target.exists() and target.is_dir():
            shutil.rmtree(target)
        temp_target = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex}")
        shutil.copy2(source, temp_target)
        os.replace(temp_target, target)


def _safe_materialization_target(root: Path, physical_path: str) -> Path:
    if root.is_symlink():
        raise WorkspaceMaterializationError(
            "workspace_root_symlink_not_allowed",
            "workspace artifact root cannot be a symlink",
        )
    target = root / PurePosixPath(physical_path)
    cursor = root
    for part in PurePosixPath(physical_path).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise WorkspaceMaterializationError(
                "materialization_target_symlink_not_allowed",
                f"materialization target crosses a symlink: {physical_path}",
            )
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise WorkspaceMaterializationError(
            "materialization_target_escape",
            f"materialization target escapes the workspace: {physical_path}",
        ) from error
    return target


def _set_reference_read_only(path: Path) -> None:
    candidates = [path]
    if path.is_dir():
        candidates.extend(path.rglob("*"))
    for candidate in candidates:
        try:
            mode = candidate.stat().st_mode
            candidate.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
        except OSError:
            continue


def _source_stats(path: Path) -> tuple[int, int]:
    if path.is_file():
        return 1, path.stat().st_size
    files = [candidate for candidate in path.rglob("*") if candidate.is_file()]
    return len(files), sum(candidate.stat().st_size for candidate in files)


async def pull_refs_into_workspace(
    *,
    refs: list[str],
    artifact_root: Path,
    tenant: str,
    project: str,
    user_id: str,
    conversation_id: str = "",
    storage_path: Optional[str] = None,
    source_resolver: Optional[WorkspaceSourceResolver] = None,
) -> list[dict[str, Any]]:
    """Materialize object refs at their canonical, collision-safe workspace paths."""
    root = Path(artifact_root)
    if root.is_symlink():
        raise WorkspaceMaterializationError(
            "workspace_root_symlink_not_allowed",
            "workspace artifact root cannot be a symlink",
        )
    root.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    for raw_ref in refs or []:
        ref = str(raw_ref or "").strip()
        if not ref:
            continue
        staging_dir = root / f"{MATERIALIZATION_TEMP_PREFIX}{uuid.uuid4().hex}"
        try:
            source = await resolve_workspace_source(
                ref=ref,
                staging_dir=staging_dir,
                tenant=tenant,
                project=project,
                user_id=user_id,
                conversation_id=conversation_id,
                storage_path=storage_path,
                source_resolver=source_resolver,
            )
            physical_path = canonical_workspace_path_for_ref(
                source.resolved_ref,
                current_conversation_id=conversation_id,
            )
            target = _safe_materialization_target(root, physical_path)
            _copy_source(source.local_path, target)
            _set_reference_read_only(target)
            file_count, size = _source_stats(target)
            reports.append({
                "ref": ref,
                "requested_ref": ref,
                "object_ref": source.object_ref or (ref if not ref.startswith(CONVERSATION_FILE_REF_PREFIX) else ""),
                "ok": True,
                "kind": "directory" if target.is_dir() else "file",
                "logical_path": canonical_logical_ref(
                    source.resolved_ref,
                    current_conversation_id=conversation_id,
                ),
                "physical_path": physical_path,
                "path": str(target),
                "filename": target.name,
                "file_count": file_count,
                "size": size,
                "mime": source.mime or (mimetypes.guess_type(target.name)[0] if target.is_file() else "") or "application/octet-stream",
                "read_only": True,
            })
        except Exception as error:
            reports.append({
                "ref": ref,
                "requested_ref": ref,
                "ok": False,
                "error": str(error),
                "error_code": getattr(error, "code", "materialization_failed"),
            })
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)
    return reports
