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
        "turn_id": os.environ.get("KDCUBE_WS_TURN") or "",
        "storage_path": os.environ.get("KDCUBE_WS_STORAGE") or "",
        "broker_socket": os.environ.get("KDCUBE_WS_BROKER_SOCKET") or "",
        "broker_token": os.environ.get("KDCUBE_WS_BROKER_TOKEN") or "",
    }


def build_app() -> Any:
    from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.workspace_tools import (
        pull_into_workspace,
        pull_report_text,
        checkout_into_workspace,
        workspace_artifact_root,
    )
    from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.workspace_broker import (
        broker_source_resolver,
        request_workspace_broker,
    )

    app = _mcp_server("KDCube turn workspace")
    ident = _identity()
    source_resolver = (
        broker_source_resolver(
            socket_path=ident["broker_socket"],
            token=ident["broker_token"],
        )
        if ident["broker_socket"] and ident["broker_token"]
        else None
    )

    @app.tool()
    async def pull(refs: List[str]) -> str:
        """Materialize object refs as read-only, source-scoped workspace data.

        Pass a `conv:fi:` file ref, or an owner object_ref supported by this
        adapter, exactly as it appears. `conv:ev:` and other timeline refs are
        records: read the event first and pull its object_ref. Nothing is on
        disk until you ask. Open the returned path with your ordinary file
        tools; use checkout before editing it.

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
            source_resolver=source_resolver,
        )
        return pull_report_text(rows, workspace=ident["workspace"])

    @app.tool()
    async def checkout(items: List[dict]) -> str:
        """Create or reset editable files from durable object refs.

        Each item has exactly `from`, `to`, and `strategy`. `from` is the
        durable object locator. `to` is a current-workspace path under
        `git/projects/...` (versioned project state) or `files/...` (an
        individual derivative). `strategy` is `replace`, or `overlay` for a
        directory. Repeating replace is the reset operation. All items are
        validated before any destination changes.
        """
        try:
            result = await checkout_into_workspace(
                list(items or []),
                workspace=ident["workspace"],
                turn_id=ident["turn_id"],
                tenant=ident["tenant"],
                project=ident["project"],
                user_id=ident["user_id"],
                conversation_id=ident["conversation_id"],
                storage_path=ident["storage_path"] or None,
                source_resolver=source_resolver,
            )
        except Exception as error:
            code = getattr(error, "code", "checkout_failed")
            details = getattr(error, "details", None)
            suffix = f" Details: {details}" if details else ""
            return f"Checkout failed ({code}): {error}.{suffix}"
        lines = ["Editable checkout complete:"]
        for row in result.get("items") or []:
            path = Path(ident["workspace"]) / ".kdcube/turn-workspace" / str(row.get("physical_path") or "")
            try:
                shown = path.relative_to(Path(ident["workspace"]))
            except ValueError:
                shown = path
            lines.append(
                f"- {row.get('from')} -> `{shown}` ({row.get('strategy')}, {row.get('kind')})"
            )
        return "\n".join(lines)

    @app.tool()
    async def publish(paths: List[str]) -> str:
        """Publish selected current-turn files to the conversation.

        Pass a stable `files/...` path or the current-turn workspace-relative
        path returned by checkout/used by your native file tools. The trusted
        parent binds the user, conversation, and hosting service. The result
        returns durable `conv:fi:` refs; credentials never enter this process or
        the model context.
        """
        if not ident["broker_socket"] or not ident["broker_token"]:
            return "Publish failed (publisher_unavailable): this hosted runtime has no trusted publisher."
        try:
            rows = await request_workspace_broker(
                socket_path=ident["broker_socket"],
                token=ident["broker_token"],
                operation="publish",
                payload={"paths": list(paths or [])},
            )
        except Exception as error:
            return f"Publish failed ({getattr(error, 'code', 'publish_failed')}): {error}"
        lines = ["Published to the conversation:"]
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"- {row.get('filename') or 'file'} -> "
                f"{row.get('logical_path') or row.get('hosted_uri') or '(hosted)'}"
            )
        return "\n".join(lines) if len(lines) > 1 else "No files were published."

    @app.tool()
    async def record_turn_summary(
        summary: str,
        refs: List[str] | None = None,
        phrases: List[str] | None = None,
        entities: List[str] | None = None,
    ) -> str:
        """Stage one searchable summary of this turn's reusable result.

        State the outcome, durable facts, decisions, and relevant object/file
        refs. ``phrases`` are exact ways a person may search for it; ``entities``
        are names or identifiers. Call again to replace the earlier draft. The
        trusted parent persists only the final draft, and only if the turn
        completes successfully. This does not alter the agent's private session.
        """
        if not ident["broker_socket"] or not ident["broker_token"]:
            return (
                "Turn summary failed (turn_summary_unavailable): this hosted runtime "
                "has no trusted summary binding."
            )
        try:
            receipt = await request_workspace_broker(
                socket_path=ident["broker_socket"],
                token=ident["broker_token"],
                operation="record_turn_summary",
                payload={
                    "summary": summary,
                    "refs": list(refs or []),
                    "phrases": list(phrases or []),
                    "entities": list(entities or []),
                },
            )
        except Exception as error:
            return (
                f"Turn summary failed ({getattr(error, 'code', 'turn_summary_failed')}): "
                f"{error}"
            )
        action = "replaced" if bool((receipt or {}).get("replaced")) else "staged"
        return (
            f"Turn summary {action}; it becomes durable and searchable only after "
            "this turn completes successfully."
        )

    @app.tool()
    async def pulled() -> str:
        """List what you have already pulled this turn."""
        root = workspace_artifact_root(ident["workspace"])
        if not root.is_dir():
            return "Nothing pulled yet."
        names = sorted(
            path.relative_to(Path(ident["workspace"])).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        )
        if not names:
            return "Nothing pulled yet."
        return "\n".join(f"- `{name}`" for name in names)

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
