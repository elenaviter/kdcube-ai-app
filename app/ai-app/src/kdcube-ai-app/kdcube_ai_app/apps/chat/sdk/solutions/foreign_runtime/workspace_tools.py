# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
#
# ── workspace_tools.py ── the turn's own files, for a runtime that has none ──
#
# A hosted agent is handed refs, not bytes: the message says a post or a file is
# attached and names it, and the agent decides whether it needs to open it. LOW
# COST BY DESIGN — nothing is materialized until the agent asks, because most
# turns never touch the attachment.
#
# lg-react binds this as an ordinary in-process tool (`pull_files`, beside
# `run_python`). A CLI runtime has no in-process binding step: its only door for
# tools is MCP. So the same contract arrives as a LOCAL STDIO server this module
# describes — spawned beside the agent, speaking to nothing over the network.
#
# WHY NOT THE NAMED-SERVICE DOOR: pulling is a property of the TURN, not a
# service capability. Namespaces come and go — an administrator narrows the
# inventory, a user unchecks a namespace, a grant lapses — and none of that may
# take away an agent's ability to open a file its own conversation carries. An
# agent with zero namespaces still reads, edits and answers about the files it
# was given; only DOMAIN questions need the door.
#
# The boundary that keeps it honest: this pulls refs the CONVERSATION already
# carries (a `conv:fi:` attachment, a file an earlier turn produced), resolved by
# the platform's own byte resolver under the turn's identity. It is not a way to
# name arbitrary storage, and it is not a second route to something a capability
# pick removed.

from __future__ import annotations

import mimetypes
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

__all__ = [
    "WORKSPACE_MCP_SERVER_ID",
    "WORKSPACE_PULL_TOOL",
    "WORKSPACE_CHECKOUT_TOOL",
    "WORKSPACE_PUBLISH_TOOL",
    "WorkspacePublishError",
    "build_workspace_hosting_service",
    "workspace_mcp_server",
    "workspace_artifact_root",
    "workspace_pull_dir",
    "pull_into_workspace",
    "checkout_into_workspace",
    "publish_workspace_files",
]

#: The server id and tool name the runtime's permission rules name.
#: NOT "workspace": Claude Code reserves that name and refuses the server with
#: `reserved_name`, which the lane sees only as tools that never arrive.
WORKSPACE_MCP_SERVER_ID = "turn_workspace"
WORKSPACE_PULL_TOOL = f"mcp__{WORKSPACE_MCP_SERVER_ID}__pull"
WORKSPACE_CHECKOUT_TOOL = f"mcp__{WORKSPACE_MCP_SERVER_ID}__checkout"
WORKSPACE_PUBLISH_TOOL = f"mcp__{WORKSPACE_MCP_SERVER_ID}__publish"

#: The adapter-visible mount point differs by harness, but everything below it
#: follows the canonical turn workspace layout.
WORKSPACE_SUBDIR = ".kdcube/turn-workspace"


def workspace_artifact_root(workspace: Path | str) -> Path:
    base = Path(workspace)
    cursor = base
    if cursor.is_symlink():
        raise ValueError("workspace root cannot be a symlink")
    for part in Path(WORKSPACE_SUBDIR).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"workspace artifact path crosses a symlink: {WORKSPACE_SUBDIR}")
    return cursor


def workspace_pull_dir(workspace: Path | str) -> Path:
    """Compatibility alias for callers that need the materialization root."""
    return workspace_artifact_root(workspace)


def _bound_communicator(entrypoint: Any) -> Any:
    """Resolve a communicator without requiring a chat-owned request lane."""
    try:
        comm = getattr(entrypoint, "comm", None)
    except RuntimeError:
        comm = None
    if comm is not None:
        return comm

    from kdcube_ai_app.apps.chat.sdk.runtime.comm_ctx import get_comm

    return get_comm()


def build_workspace_hosting_service(entrypoint: Any) -> Any:
    """Construct the standard trusted conversation file host for an adapter."""
    from kdcube_ai_app.apps.chat.sdk.config import get_settings
    from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.hosting import (
        ApplicationHostingService,
    )
    from kdcube_ai_app.apps.chat.sdk.storage.conversation_store import ConversationStore

    return ApplicationHostingService(
        store=ConversationStore(get_settings().STORAGE_PATH),
        comm=_bound_communicator(entrypoint),
        logger=getattr(entrypoint, "logger", None),
    )


def workspace_mcp_server(
    *,
    workspace: Path | str,
    tenant: str,
    project: str,
    user_id: str,
    conversation_id: str = "",
    turn_id: str = "",
    storage_path: str = "",
    python_executable: str = "",
    broker_socket: str = "",
    broker_token: str = "",
) -> Dict[str, Any]:
    """The `.mcp.json` entry for the turn's workspace server.

    A stdio child of the agent process: no URL, no bearer, no grant, nothing for
    a capability pick to remove. Identity travels in the environment rather than
    in tool arguments, so the agent cannot pull as somebody else by asking."""
    env = {
        "KDCUBE_WS_WORKSPACE": str(workspace),
        "KDCUBE_WS_TENANT": str(tenant or ""),
        "KDCUBE_WS_PROJECT": str(project or ""),
        "KDCUBE_WS_USER": str(user_id or ""),
        "KDCUBE_WS_CONVERSATION": str(conversation_id or ""),
        "KDCUBE_WS_TURN": str(turn_id or ""),
    }
    if storage_path:
        env["KDCUBE_WS_STORAGE"] = str(storage_path)
    if broker_socket and broker_token:
        env["KDCUBE_WS_BROKER_SOCKET"] = str(broker_socket)
        env["KDCUBE_WS_BROKER_TOKEN"] = str(broker_token)
    return {
        "type": "stdio",
        "command": str(python_executable or sys.executable or "python3"),
        "args": ["-m", "kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.workspace_mcp"],
        "env": env,
    }


async def pull_into_workspace(
    refs: List[str],
    *,
    workspace: Path | str,
    tenant: str,
    project: str,
    user_id: str,
    conversation_id: str = "",
    storage_path: Optional[str] = None,
    source_resolver: Any = None,
) -> List[Dict[str, Any]]:
    """Materialize `refs` under the canonical source-scoped workspace layout.

    One report row per ref, successes and failures alike — a ref that cannot be
    resolved never aborts the batch, because an agent asking for three files and
    getting two plus a reason can still work."""
    from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.materialization import (
        pull_refs_into_workspace,
    )

    root = workspace_artifact_root(workspace)
    root.mkdir(parents=True, exist_ok=True)
    return await pull_refs_into_workspace(
        refs=[str(ref).strip() for ref in refs or [] if str(ref).strip()],
        artifact_root=root,
        tenant=str(tenant or ""),
        project=str(project or ""),
        user_id=str(user_id or ""),
        conversation_id=str(conversation_id or ""),
        storage_path=storage_path or None,
        source_resolver=source_resolver,
    )


async def checkout_into_workspace(
    items: List[Mapping[str, Any]],
    *,
    workspace: Path | str,
    turn_id: str,
    tenant: str,
    project: str,
    user_id: str,
    conversation_id: str = "",
    storage_path: Optional[str] = None,
    source_resolver: Any = None,
) -> Dict[str, Any]:
    """Create/reset editable current-turn state from durable object locators."""
    from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.checkout import (
        checkout_workspace_items,
    )

    return await checkout_workspace_items(
        items=list(items or []),
        artifact_root=workspace_artifact_root(workspace),
        current_turn_id=str(turn_id or "").strip(),
        tenant=str(tenant or ""),
        project=str(project or ""),
        user_id=str(user_id or ""),
        conversation_id=str(conversation_id or ""),
        storage_path=storage_path or None,
        source_resolver=source_resolver,
    )


class WorkspacePublishError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _publish_source_path(*, workspace: Path, turn_id: str, value: str) -> tuple[Path, str]:
    raw = str(value or "").strip().replace("\\", "/").strip("/")
    if not raw:
        raise WorkspacePublishError("publish_path_missing", "publish requires a files/... path")
    prefixes = (
        "files/",
        f"{turn_id}/files/",
        f"{WORKSPACE_SUBDIR}/{turn_id}/files/",
    )
    relative = next((raw[len(prefix) :] for prefix in prefixes if raw.startswith(prefix)), "")
    parts = Path(relative).parts
    if not relative or any(part in {"", ".", ".."} for part in parts):
        raise WorkspacePublishError(
            "publish_path_outside_files",
            "publish accepts only current-turn files/... paths",
        )
    artifact_root = workspace_artifact_root(workspace)
    files_root = artifact_root / turn_id / "files"
    source = files_root.joinpath(*parts)
    cursor = artifact_root
    for part in (turn_id, "files", *parts):
        cursor = cursor / part
        if cursor.is_symlink():
            raise WorkspacePublishError(
                "publish_symlink_not_allowed",
                f"publish path crosses a symlink: {raw}",
            )
    try:
        source.resolve().relative_to(files_root.resolve())
    except ValueError as error:
        raise WorkspacePublishError(
            "publish_path_escape",
            f"publish path escapes current-turn files: {raw}",
        ) from error
    if not source.is_file():
        raise WorkspacePublishError("publish_file_missing", f"publish file does not exist: {raw}")
    return source, Path(*parts).as_posix()


async def publish_workspace_files(
    paths: List[str],
    *,
    workspace: Path | str,
    turn_id: str,
    hosting_service: Any,
    tenant: str,
    project: str,
    user_id: str,
    user_type: str,
    conversation_id: str,
    request_id: str = "",
    state: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Host selected current-turn derivatives and surface them to the user.

    Only files below this turn's ``files/`` area are eligible. The agent chooses
    paths; the trusted parent binds identity and conversation hosting.
    """
    turn = str(turn_id or "").strip()
    if not turn:
        raise WorkspacePublishError("publish_turn_missing", "publish requires the current turn id")
    if hosting_service is None:
        raise WorkspacePublishError("publisher_unavailable", "conversation file hosting is unavailable")
    root = Path(workspace)
    selected: List[tuple[Path, str]] = []
    seen: set[str] = set()
    for value in paths or []:
        source, relative = _publish_source_path(workspace=root, turn_id=turn, value=value)
        if relative not in seen:
            selected.append((source, relative))
            seen.add(relative)
    if not selected:
        raise WorkspacePublishError("publish_paths_missing", "publish requires at least one files/... path")

    runtime_root = Path(tempfile.mkdtemp(prefix="kdcube-workspace-publish-"))
    artifact_root = runtime_root / "workdir"
    artifacts: List[Dict[str, Any]] = []
    try:
        for source, relative in selected:
            physical = Path(turn) / "files" / relative
            staged = artifact_root / physical
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, staged)
            mime = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
            artifacts.append({
                "type": "file",
                "output": {"type": "file", "path": physical.as_posix(), "mime": mime},
                "mime": mime,
                "resource_id": relative,
                "slot": relative,
                "tool_id": "workspace.publish",
                "description": "Published by the hosted agent",
            })
        hosted = await hosting_service.host_files_to_conversation(
            rid=str(request_id or ""),
            files=artifacts,
            outdir=runtime_root,
            tenant=str(tenant or ""),
            project=str(project or ""),
            user=str(user_id or ""),
            conversation_id=str(conversation_id or ""),
            user_type=str(user_type or "registered"),
            turn_id=turn,
        )
        rows = [dict(row) for row in (hosted or []) if isinstance(row, Mapping)]
        if rows:
            await hosting_service.emit_solver_artifacts(files=rows, citations=[])
            if state is not None:
                existing = state.setdefault("hosted_files", [])
                if not isinstance(existing, list):
                    existing = []
                    state["hosted_files"] = existing
                known = {
                    str(row.get("logical_path") or row.get("hosted_uri") or "")
                    for row in existing
                    if isinstance(row, Mapping)
                }
                existing.extend(
                    row
                    for row in rows
                    if str(row.get("logical_path") or row.get("hosted_uri") or "") not in known
                )
        return rows
    finally:
        shutil.rmtree(runtime_root, ignore_errors=True)


def pull_report_text(rows: List[Mapping[str, Any]], *, workspace: Path | str) -> str:
    """The tool's answer: what landed, where, and why anything did not.

    Paths are reported relative to the agent's working directory, because that is
    what it will type into its own file tools."""
    if not rows:
        return "No refs were given, so nothing was pulled."
    base = Path(workspace)
    lines: List[str] = []
    for row in rows:
        ref = str(row.get("ref") or "")
        if not row.get("ok"):
            lines.append(f"- {ref} — NOT pulled: {row.get('error') or 'could not be resolved'}")
            continue
        path = Path(str(row.get("path") or ""))
        try:
            shown = path.relative_to(base)
        except Exception:
            shown = path
        size = row.get("size")
        mime = str(row.get("mime") or "").strip()
        detail = " · ".join(part for part in (f"{size:,} bytes" if isinstance(size, int) else "", mime) if part)
        lines.append(f"- {ref} → `{shown}`" + (f" ({detail})" if detail else ""))
        # The platform's own time-limited download link for the same bytes.
        # Binaries do not travel through a tool result; this is how one is
        # handed to a person, or to anything that takes a URL, without
        # copying it anywhere.
        link = str(row.get("download_url") or "").strip()
        if link:
            lines.append(f"  link: {link}")
    return "\n".join(lines)
