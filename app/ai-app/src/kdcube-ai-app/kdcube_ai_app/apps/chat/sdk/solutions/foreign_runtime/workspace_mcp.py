# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
#
# ── workspace_mcp.py ── the turn's workspace, as a local stdio MCP server ──
#
# Run as `python -m …foreign_runtime.workspace_mcp`, spawned by the hosted agent
# itself (the `.mcp.json` entry `workspace_tools.workspace_mcp_server` writes).
# It speaks stdio to its parent and to nothing else: no URL, no bearer, no grant.
#
# It exists because a CLI runtime has one door for tools — MCP — while an
# in-process runtime binds the same function directly (lg-react's `pull_files`).
# Same contract, different binding step.
#
# The turn is an EVENT STREAM: files and objects arrive as events naming refs,
# and what the agent does about them is the agent's decision. This server is one
# of the tools that decision needs — the one that turns a ref into bytes on
# disk, when and only when the agent asks.
#
# Identity comes from the environment the parent set, never from tool arguments:
# an agent cannot pull as somebody else by naming them.

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, List

LOGGER = logging.getLogger("kdcube.workspace_mcp")


def _mcp_server(name: str) -> Any:
    """FastMCP under either mcp SDK generation.

    The runtime image ships mcp 2.x (`FastMCP` renamed `MCPServer`) while an
    operator venv may still run 1.x. One factory imports whichever is present,
    rather than forking the module."""
    try:  # mcp >= 2
        from mcp.server import MCPServer  # type: ignore

        return MCPServer(name)
    except Exception:
        from mcp.server.fastmcp import FastMCP  # type: ignore

        return FastMCP(name)


def _identity() -> dict:
    return {
        "workspace": os.environ.get("KDCUBE_WS_WORKSPACE") or os.getcwd(),
        "tenant": os.environ.get("KDCUBE_WS_TENANT") or "",
        "project": os.environ.get("KDCUBE_WS_PROJECT") or "",
        "user_id": os.environ.get("KDCUBE_WS_USER") or "",
        "conversation_id": os.environ.get("KDCUBE_WS_CONVERSATION") or "",
        "storage_path": os.environ.get("KDCUBE_WS_STORAGE") or "",
    }


def build_app() -> Any:
    from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.workspace_tools import (
        pull_into_workspace,
        pull_report_text,
        workspace_pull_dir,
    )

    app = _mcp_server("KDCube turn workspace")
    ident = _identity()

    @app.tool()
    async def pull(refs: List[str]) -> str:
        """Bring files this conversation carries into your working directory.

        A message tells you a file or an object is attached and names its ref
        (`conv:fi:…`); nothing is on disk until you ask for it here. Pass the
        refs exactly as they appear, then open what you get back with your
        ordinary file tools.

        Works whatever else is switched on or off: this is your own workspace,
        not a service. Refs that cannot be resolved are reported and the rest
        still arrive."""
        rows = await pull_into_workspace(
            list(refs or []),
            workspace=ident["workspace"],
            tenant=ident["tenant"],
            project=ident["project"],
            user_id=ident["user_id"],
            conversation_id=ident["conversation_id"],
            storage_path=ident["storage_path"] or None,
        )
        return pull_report_text(rows, workspace=ident["workspace"])

    @app.tool()
    async def pulled() -> str:
        """List what you have already pulled this turn."""
        dest = workspace_pull_dir(ident["workspace"])
        if not dest.is_dir():
            return "Nothing pulled yet."
        names = sorted(p.name for p in dest.iterdir() if p.is_file())
        if not names:
            return "Nothing pulled yet."
        return "\n".join(f"- `{Path(dest).name}/{name}`" for name in names)

    return app


def main() -> None:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    app = build_app()
    run = getattr(app, "run", None)
    if run is None:  # pragma: no cover - defensive across SDK generations
        raise SystemExit("mcp server object exposes no run()")
    try:
        run(transport="stdio")
    except TypeError:
        # mcp 2.x runs stdio by default and takes no transport argument.
        result = run()
        if asyncio.iscoroutine(result):
            asyncio.run(result)


if __name__ == "__main__":  # pragma: no cover
    main()
