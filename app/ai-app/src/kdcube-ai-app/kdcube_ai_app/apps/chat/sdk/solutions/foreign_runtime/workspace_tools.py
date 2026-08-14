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

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

__all__ = [
    "WORKSPACE_MCP_SERVER_ID",
    "WORKSPACE_PULL_TOOL",
    "workspace_mcp_server",
    "workspace_pull_dir",
    "pull_into_workspace",
]

#: The server id and tool name the runtime's permission rules name.
#: NOT "workspace": Claude Code reserves that name and refuses the server with
#: `reserved_name`, which the lane sees only as tools that never arrive.
WORKSPACE_MCP_SERVER_ID = "turn_workspace"
WORKSPACE_PULL_TOOL = f"mcp__{WORKSPACE_MCP_SERVER_ID}__pull"

#: Where pulled bytes land, relative to the agent's working directory. A single
#: predictable folder, so the agent can list it and the operator can find it.
PULL_SUBDIR = "pulled"


def workspace_pull_dir(workspace: Path | str) -> Path:
    return Path(workspace) / PULL_SUBDIR


def workspace_mcp_server(
    *,
    workspace: Path | str,
    tenant: str,
    project: str,
    user_id: str,
    conversation_id: str = "",
    storage_path: str = "",
    python_executable: str = "",
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
    }
    if storage_path:
        env["KDCUBE_WS_STORAGE"] = str(storage_path)
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
) -> List[Dict[str, Any]]:
    """Materialize `refs` under the workspace's pull directory.

    One report row per ref, successes and failures alike — a ref that cannot be
    resolved never aborts the batch, because an agent asking for three files and
    getting two plus a reason can still work."""
    from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.pull import (
        pull_refs_into_dir,
    )

    dest = workspace_pull_dir(workspace)
    dest.mkdir(parents=True, exist_ok=True)
    return await pull_refs_into_dir(
        refs=[str(ref).strip() for ref in refs or [] if str(ref).strip()],
        dest_dir=dest,
        tenant=str(tenant or ""),
        project=str(project or ""),
        user_id=str(user_id or ""),
        conversation_id=str(conversation_id or ""),
        storage_path=storage_path or None,
    )


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
