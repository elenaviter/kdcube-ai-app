# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""Transactional checkout into a harness turn's editable workspace areas."""

from __future__ import annotations

import os
import shutil
import stat
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional

from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.materialization import (
    WorkspaceSourceResolver,
    resolve_workspace_source,
)
from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.references import (
    ARTIFACT_NAMESPACE_FILES,
    ARTIFACT_NAMESPACE_PROJECTS,
    build_logical_artifact_path,
)


CHECKOUT_STRATEGIES = {"replace", "overlay"}
CHECKOUT_TEMP_PREFIX = ".kdcube-checkout-"


class WorkspaceCheckoutError(ValueError):
    def __init__(self, code: str, message: str, *, details: Any = None):
        super().__init__(message)
        self.code = code
        self.details = details


def _normalize_target(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise WorkspaceCheckoutError("checkout_target_missing", "checkout item requires 'to'")
    if "\\" in raw or raw.startswith("/"):
        raise WorkspaceCheckoutError(
            "checkout_target_invalid",
            "checkout target must be a workspace-relative POSIX path",
        )
    target = PurePosixPath(raw.rstrip("/"))
    if any(part in {"", ".", ".."} for part in target.parts):
        raise WorkspaceCheckoutError(
            "checkout_target_invalid",
            f"unsafe checkout target: {raw}",
        )
    normalized = str(target)
    allowed = (
        normalized == ARTIFACT_NAMESPACE_PROJECTS
        or normalized.startswith(f"{ARTIFACT_NAMESPACE_PROJECTS}/")
        or normalized == ARTIFACT_NAMESPACE_FILES
        or normalized.startswith(f"{ARTIFACT_NAMESPACE_FILES}/")
    )
    if not allowed:
        raise WorkspaceCheckoutError(
            "checkout_target_outside_editable_area",
            "checkout target must be under git/projects or files",
        )
    return normalized


def normalize_checkout_items(items: list[Mapping[str, Any]] | None) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    errors: list[dict[str, Any]] = []
    for index, item in enumerate(items or []):
        if not isinstance(item, Mapping):
            errors.append({"index": index, "reason": "item_must_be_object"})
            continue
        extra_keys = sorted(
            str(key) for key in item.keys() if str(key) not in {"from", "to", "strategy"}
        )
        if extra_keys:
            errors.append({
                "index": index,
                "reason": "checkout_item_unknown_fields",
                "fields": extra_keys,
            })
            continue
        source_ref = str(item.get("from") or "").strip()
        strategy = str(item.get("strategy") or "").strip().lower()
        try:
            target = _normalize_target(item.get("to"))
        except WorkspaceCheckoutError as error:
            errors.append({"index": index, "reason": error.code, "message": str(error)})
            continue
        if not source_ref:
            errors.append({"index": index, "reason": "checkout_source_missing"})
            continue
        if strategy not in CHECKOUT_STRATEGIES:
            errors.append({
                "index": index,
                "reason": "checkout_strategy_invalid",
                "message": "strategy must be replace or overlay",
            })
            continue
        normalized.append({"from": source_ref, "to": target, "strategy": strategy})
    if errors:
        raise WorkspaceCheckoutError(
            "checkout_items_invalid",
            "one or more checkout items are invalid",
            details=errors,
        )
    if not normalized:
        raise WorkspaceCheckoutError("checkout_items_missing", "checkout requires at least one item")
    _reject_overlapping_targets(normalized)
    return normalized


def _reject_overlapping_targets(items: list[dict[str, str]]) -> None:
    targets = [(index, PurePosixPath(item["to"])) for index, item in enumerate(items)]
    collisions: list[dict[str, Any]] = []
    for left_index, left in targets:
        for right_index, right in targets[left_index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                collisions.append({
                    "left_index": left_index,
                    "left": str(left),
                    "right_index": right_index,
                    "right": str(right),
                })
    if collisions:
        raise WorkspaceCheckoutError(
            "checkout_targets_overlap",
            "checkout items cannot target the same path or ancestor/descendant paths",
            details=collisions,
        )


def _ensure_no_symlink_path(root: Path, relative: str) -> Path:
    if root.is_symlink():
        raise WorkspaceCheckoutError(
            "checkout_workspace_symlink_not_allowed",
            "current-turn workspace root cannot be a symlink",
        )
    target = root / PurePosixPath(relative)
    cursor = root
    for part in PurePosixPath(relative).parts:
        cursor = cursor / part
        if cursor.exists() and cursor.is_symlink():
            raise WorkspaceCheckoutError(
                "checkout_target_symlink_not_allowed",
                f"checkout target crosses a symlink: {relative}",
            )
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise WorkspaceCheckoutError(
            "checkout_target_escape",
            f"checkout target escapes the current workspace: {relative}",
        ) from error
    return target


def _reject_symlink_tree(path: Path, *, relative: str) -> None:
    for candidate in path.rglob("*"):
        if candidate.is_symlink():
            raise WorkspaceCheckoutError(
                "checkout_target_symlink_not_allowed",
                f"overlay target contains a symlink: {relative}",
            )


def _copy_tree_overlay(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        destination = target / child.name
        if child.is_dir():
            if destination.exists() and not destination.is_dir():
                destination.unlink()
            _copy_tree_overlay(child, destination)
        else:
            if destination.exists() and destination.is_dir():
                shutil.rmtree(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, destination)


def _set_editable(path: Path) -> None:
    candidates = [path]
    if path.is_dir():
        candidates.extend(path.rglob("*"))
    for candidate in candidates:
        try:
            mode = candidate.stat().st_mode
            candidate.chmod(mode | stat.S_IWUSR)
        except OSError:
            continue


def _remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _target_logical_ref(*, turn_id: str, conversation_id: str, target: str) -> str:
    if target == ARTIFACT_NAMESPACE_PROJECTS:
        namespace, relpath = ARTIFACT_NAMESPACE_PROJECTS, "."
    elif target.startswith(f"{ARTIFACT_NAMESPACE_PROJECTS}/"):
        namespace = ARTIFACT_NAMESPACE_PROJECTS
        relpath = target[len(ARTIFACT_NAMESPACE_PROJECTS) + 1 :]
    elif target == ARTIFACT_NAMESPACE_FILES:
        namespace, relpath = ARTIFACT_NAMESPACE_FILES, "."
    else:
        namespace = ARTIFACT_NAMESPACE_FILES
        relpath = target[len(ARTIFACT_NAMESPACE_FILES) + 1 :]
    if relpath == ".":
        owner = f"conv_{conversation_id}." if conversation_id else ""
        return f"conv:fi:{owner}{turn_id}.{namespace}"
    return build_logical_artifact_path(
        turn_id=turn_id,
        namespace=namespace,
        relpath=relpath,
        conversation_id=conversation_id,
    )


async def checkout_workspace_items(
    *,
    items: list[Mapping[str, Any]],
    artifact_root: Path,
    current_turn_id: str,
    tenant: str,
    project: str,
    user_id: str,
    conversation_id: str = "",
    storage_path: Optional[str] = None,
    source_resolver: Optional[WorkspaceSourceResolver] = None,
) -> dict[str, Any]:
    """Resolve all sources, then atomically apply them to current editable state."""
    if not str(current_turn_id or "").strip():
        raise WorkspaceCheckoutError("checkout_turn_missing", "checkout requires the current turn id")
    normalized = normalize_checkout_items(items)
    root = Path(artifact_root)
    if root.is_symlink():
        raise WorkspaceCheckoutError(
            "checkout_workspace_symlink_not_allowed",
            "workspace artifact root cannot be a symlink",
        )
    root.mkdir(parents=True, exist_ok=True)
    current_root = root / str(current_turn_id).strip()
    if current_root.is_symlink():
        raise WorkspaceCheckoutError(
            "checkout_workspace_symlink_not_allowed",
            "current-turn workspace root cannot be a symlink",
        )
    transaction = root / f"{CHECKOUT_TEMP_PREFIX}{uuid.uuid4().hex}"
    sources_root = transaction / "sources"
    prepared_root = transaction / "prepared"
    backup_root = transaction / "backups"
    transaction.mkdir(parents=True, exist_ok=False)

    prepared: list[dict[str, Any]] = []
    try:
        for index, item in enumerate(normalized):
            source = await resolve_workspace_source(
                ref=item["from"],
                staging_dir=sources_root / str(index),
                tenant=tenant,
                project=project,
                user_id=user_id,
                conversation_id=conversation_id,
                storage_path=storage_path,
                source_resolver=source_resolver,
            )
            target = _ensure_no_symlink_path(current_root, item["to"])
            target_root_only = item["to"] in {ARTIFACT_NAMESPACE_PROJECTS, ARTIFACT_NAMESPACE_FILES}
            if source.local_path.is_file() and target_root_only:
                raise WorkspaceCheckoutError(
                    "checkout_file_target_is_area_root",
                    "a file source needs an exact file target below git/projects or files",
                    details={"index": index, "to": item["to"]},
                )
            if item["strategy"] == "overlay" and not source.local_path.is_dir():
                raise WorkspaceCheckoutError(
                    "checkout_overlay_requires_directory",
                    "overlay is defined only for directory sources",
                    details={"index": index, "from": item["from"]},
                )

            candidate = prepared_root / str(index)
            if item["strategy"] == "overlay" and target.exists():
                if not target.is_dir():
                    raise WorkspaceCheckoutError(
                        "checkout_overlay_target_not_directory",
                        "overlay target must be a directory",
                        details={"index": index, "to": item["to"]},
                    )
                _reject_symlink_tree(target, relative=item["to"])
                shutil.copytree(target, candidate)
                _copy_tree_overlay(source.local_path, candidate)
            elif source.local_path.is_dir():
                shutil.copytree(source.local_path, candidate)
            else:
                candidate.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source.local_path, candidate)
            _set_editable(candidate)
            prepared.append({
                "index": index,
                "source": source,
                "target": target,
                "prepared": candidate,
                **item,
            })

        applied: list[dict[str, Any]] = []
        backups: list[tuple[Path, Path]] = []
        installed: list[Path] = []
        try:
            for row in prepared:
                target = row["target"]
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() or target.is_symlink():
                    backup = backup_root / str(row["index"])
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(target, backup)
                    backups.append((target, backup))
                os.replace(row["prepared"], target)
                installed.append(target)
                source = row["source"]
                applied.append({
                    "from": row["from"],
                    "resolved_from": source.resolved_ref,
                    **({"object_ref": source.object_ref} if source.object_ref else {}),
                    "to": row["to"],
                    "strategy": row["strategy"],
                    "kind": "directory" if target.is_dir() else "file",
                    "physical_path": f"{current_turn_id}/{row['to']}",
                    "logical_path": _target_logical_ref(
                        turn_id=current_turn_id,
                        conversation_id=conversation_id,
                        target=row["to"],
                    ),
                })
        except Exception:
            for target in reversed(installed):
                _remove_path(target)
            for target, backup in reversed(backups):
                if backup.exists() or backup.is_symlink():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(backup, target)
            raise

        return {
            "ok": True,
            "turn_id": current_turn_id,
            "items": applied,
            "checked_out_from": [row["from"] for row in applied],
            "editable_roots": [
                f"{current_turn_id}/{ARTIFACT_NAMESPACE_PROJECTS}",
                f"{current_turn_id}/{ARTIFACT_NAMESPACE_FILES}",
            ],
        }
    finally:
        shutil.rmtree(transaction, ignore_errors=True)
