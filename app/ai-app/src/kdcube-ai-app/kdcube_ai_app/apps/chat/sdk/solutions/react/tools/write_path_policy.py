# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace import resolve_artifact_path
from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.references import (
    ARTIFACT_NAMESPACE_FILES,
    normalize_physical_path,
    physical_path_to_logical_path,
    qualify_conversation_ref,
    split_physical_artifact_path,
)


WRITE_PATH_ALREADY_EXISTS = "write_path_already_exists"


@dataclass(frozen=True)
class ReactWriteTarget:
    physical_path: str
    logical_path: str
    relpath: str

    def violation_extra(self) -> dict[str, str]:
        return {
            "path": self.physical_path,
            **({"artifact_path": self.logical_path} if self.logical_path else {}),
        }


def duplicate_write_message_for_path(path: str) -> str:
    return (
        f"`react.write` cannot create `{path}` because that current-turn path already exists. "
        "The existing file was not changed. Use `react.patch` for an intentional in-place edit, or choose a "
        "different filename for a separate full version."
    )


def duplicate_write_message(target: ReactWriteTarget) -> str:
    return duplicate_write_message_for_path(target.physical_path)


def resolve_react_write_target(
    *,
    path: str,
    turn_id: str,
    conversation_id: str = "",
) -> Optional[ReactWriteTarget]:
    physical_path, relpath, _ = normalize_physical_path(
        path,
        turn_id=turn_id,
        default_namespace=ARTIFACT_NAMESPACE_FILES,
    )
    if not physical_path or not relpath:
        return None
    physical_turn_id, _, _ = split_physical_artifact_path(physical_path)
    if not physical_turn_id or physical_turn_id != turn_id:
        return None
    local_logical_path = physical_path_to_logical_path(physical_path)
    logical_path = qualify_conversation_ref(local_logical_path, conversation_id)
    return ReactWriteTarget(
        physical_path=physical_path,
        logical_path=logical_path,
        relpath=relpath,
    )


def react_write_target_is_visible(*, ctx_browser: Any, target: ReactWriteTarget) -> bool:
    try:
        visible_paths = set(ctx_browser.timeline_visible_paths() or set())
    except Exception:
        return False
    candidates = {target.physical_path}
    if target.logical_path:
        candidates.add(target.logical_path)
    local_logical_path = physical_path_to_logical_path(target.physical_path)
    if local_logical_path:
        candidates.add(local_logical_path)
    return bool(candidates.intersection(visible_paths))


def react_write_target_is_materialized(*, outdir: Path, target: ReactWriteTarget) -> bool:
    try:
        return resolve_artifact_path(
            Path(outdir),
            target.physical_path,
            create_root=False,
        ).exists()
    except Exception:
        return False


__all__ = [
    "ReactWriteTarget",
    "WRITE_PATH_ALREADY_EXISTS",
    "duplicate_write_message",
    "duplicate_write_message_for_path",
    "react_write_target_is_materialized",
    "react_write_target_is_visible",
    "resolve_react_write_target",
]
