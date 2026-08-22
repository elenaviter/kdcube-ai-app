# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""Reusable Agent Harness Workspace binding for hosted Claude Code turns."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.context_resolver import (
    resolve_context_workspace_source,
)
from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.materialization import (
    WorkspaceSourceResolver,
)
from kdcube_ai_app.apps.chat.sdk.runtime.harness.timeline.contributions import (
    stage_turn_summary,
)
from kdcube_ai_app.apps.chat.sdk.solutions.claude_code.types import (
    ClaudeCodeWorkspaceConfig,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.publication import (
    WorkspacePublicationPolicy,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.workspace_broker import (
    WorkspaceBroker,
    start_workspace_broker,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.workspace_tools import (
    WORKSPACE_MCP_SERVER_ID,
    WORKSPACE_PUBLISH_TOOL,
    WORKSPACE_TURN_SUMMARY_TOOL,
    build_workspace_hosting_service,
    publish_workspace_files,
    workspace_mcp_server,
)


CLAUDE_CODE_TURN_WORKSPACE_PERMISSION = f"mcp__{WORKSPACE_MCP_SERVER_ID}"
_WORKSPACE_INSTRUCTION_MARKER = "<!-- kdcube-agent-harness-workspace -->"
_AUTO_HOSTING_SERVICE = object()


def _deduplicate(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return tuple(result)


def _workspace_instruction_block(
    *, publication_available: bool, turn_summary_available: bool, turn_id: str
) -> str:
    current_root = f".kdcube/turn-workspace/{turn_id}"
    publication = (
        "- Use `publish(paths=[...])` only for selected current-turn file outputs. "
        "Pass the same workspace-relative path you read or wrote, or its stable `files/...` form. "
        "Publish only outputs intended for the person. The request is policy-checked and returns "
        "durable `conv:fi:` refs; a local edit is not delivered until this succeeds."
        if publication_available
        else "- Conversation-file publication is unavailable in this binding. Keep outputs local and say so."
    )
    lines = [
        _WORKSPACE_INSTRUCTION_MARKER,
        "## KDCube turn workspace",
        "",
        "Events and objects are separate. A `conv:ev:` value identifies an event record; "
        "use its separate `object_ref` when bytes are available.",
        "",
        f"The active physical turn root is `{current_root}`. Native Read, Grep, Edit, Write, "
        "and Bash use paths below that root. Workspace tool arguments use stable `files/...` "
        "or `git/projects/...` destinations and return the physical workspace-relative path.",
        "",
        "- Use `pull(refs=[...])` for a read-only local copy. Open the returned "
        "workspace-relative path with Read, Grep, or Bash.",
        "- Use `checkout(items=[{from,to,strategy}])` to create or reset editable "
        "state. `to` is below `files/...` or `git/projects/...`; `replace` resets "
        "a file or directory and `overlay` merges a directory.",
        "- Edit checked-out paths with ordinary Claude Code file tools. Repeating "
        "checkout from the durable source is reset.",
        publication,
    ]
    if turn_summary_available:
        from kdcube_ai_app.apps.chat.sdk.skills.instructions.workspace_agent_instructions import (
            turn_summary_contribution_guide,
        )

        lines.extend(["", turn_summary_contribution_guide()])
    return "\n".join(lines)


def _append_workspace_instructions(
    current: Optional[str],
    *,
    publication_available: bool,
    turn_summary_available: bool,
    turn_id: str,
) -> str:
    base = str(current or "").rstrip()
    if _WORKSPACE_INSTRUCTION_MARKER in base:
        return base + "\n"
    block = _workspace_instruction_block(
        publication_available=publication_available,
        turn_summary_available=turn_summary_available,
        turn_id=turn_id,
    )
    return f"{base}\n\n{block}\n" if base else f"{block}\n"


@dataclass
class ClaudeCodeTurnWorkspaceBinding:
    """One live broker plus its model-facing Claude Code configuration seam."""

    workspace: Path
    turn_id: str
    broker: WorkspaceBroker
    server_spec: Mapping[str, Any]
    publication_available: bool
    turn_summary_available: bool
    _owner_task: Optional[asyncio.Task[Any]] = field(default=None, repr=False)
    _owner_callback: Any = field(default=None, repr=False)

    def merge_mcp_servers(
        self,
        servers: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        merged = {
            str(server_id): dict(spec)
            for server_id, spec in dict(servers or {}).items()
        }
        existing = merged.get(WORKSPACE_MCP_SERVER_ID)
        expected = dict(self.server_spec)
        if existing is not None and existing != expected:
            raise ValueError(
                f"Claude Code MCP server id {WORKSPACE_MCP_SERVER_ID!r} is reserved "
                "for the Agent Harness Workspace binding"
            )
        merged[WORKSPACE_MCP_SERVER_ID] = expected
        return merged

    def apply_workspace_config(
        self,
        config: ClaudeCodeWorkspaceConfig,
    ) -> ClaudeCodeWorkspaceConfig:
        """Merge server, permission, enablement, and teaching into one config."""
        servers = self.merge_mcp_servers(config.mcp_servers)
        enabled = (
            tuple(servers.keys())
            if config.enabled_mcp_servers is None
            else _deduplicate(
                list(config.enabled_mcp_servers) + [WORKSPACE_MCP_SERVER_ID]
            )
        )
        allowed = _deduplicate(
            list(config.allowed_tools) + [CLAUDE_CODE_TURN_WORKSPACE_PERMISSION]
        )
        denied_values = list(config.denied_tools)
        if not self.publication_available:
            denied_values.append(WORKSPACE_PUBLISH_TOOL)
        if not self.turn_summary_available:
            denied_values.append(WORKSPACE_TURN_SUMMARY_TOOL)
        denied = _deduplicate(denied_values)
        return replace(
            config,
            mcp_servers=servers,
            enabled_mcp_servers=enabled,
            allowed_tools=allowed,
            denied_tools=denied,
            instructions_markdown=_append_workspace_instructions(
                config.instructions_markdown,
                publication_available=self.publication_available,
                turn_summary_available=self.turn_summary_available,
                turn_id=self.turn_id,
            ),
        )

    @property
    def closed(self) -> bool:
        return self.broker.closed

    async def close(self) -> None:
        if self._owner_task is not None and self._owner_callback is not None:
            self._owner_task.remove_done_callback(self._owner_callback)
            self._owner_callback = None
        await self.broker.close()

    async def __aenter__(self) -> "ClaudeCodeTurnWorkspaceBinding":
        return self

    async def __aexit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        await self.close()


async def bind_claude_code_turn_workspace(
    *,
    workspace: Path | str,
    tenant: str,
    project: str,
    user_id: str,
    conversation_id: str,
    turn_id: str,
    entrypoint: Any = None,
    state: Optional[Dict[str, Any]] = None,
    ctx_browser: Any = None,
    outdir: Path | str | None = None,
    source_resolver: Optional[WorkspaceSourceResolver] = None,
    hosting_service: Any = _AUTO_HOSTING_SERVICE,
    publication_policy: Optional[WorkspacePublicationPolicy] = None,
    turn_summary_enabled: bool = False,
    user_type: str = "",
    request_id: str = "",
    storage_path: str = "",
    python_executable: str = "",
    logger: Any = None,
    tool_id: str = "workspace.pull",
    tool_call_id: str = "claude-code-turn-workspace",
) -> ClaudeCodeTurnWorkspaceBinding:
    """Bind shared workspace and optional turn-context tools to one Claude turn.

    Identity, resolvers, hosting, and product approval stay in the trusted
    parent. The child receives only bound non-secret identity, an authenticated
    local broker endpoint, and model-supplied refs/paths.
    """
    root = Path(workspace)
    root.mkdir(parents=True, exist_ok=True)
    turn_state = state if isinstance(state, dict) else {}
    log = logger or getattr(entrypoint, "logger", None)

    browser = ctx_browser if ctx_browser is not None else getattr(entrypoint, "ctx_browser", None)
    browser_runtime = getattr(browser, "runtime_ctx", None)
    resolved_outdir = str(outdir or getattr(browser_runtime, "outdir", "") or "").strip()
    resolver = source_resolver
    if resolver is None and browser is not None and resolved_outdir:
        async def _resolve(*, ref: str, staging_dir: Path) -> Any:
            return await resolve_context_workspace_source(
                ref=ref,
                staging_dir=staging_dir,
                ctx_browser=browser,
                outdir=Path(resolved_outdir),
                state=turn_state,
                tool_id=tool_id,
                tool_call_id=tool_call_id,
            )

        resolver = _resolve

    host = hosting_service
    if host is _AUTO_HOSTING_SERVICE:
        host = None
        if entrypoint is not None:
            try:
                host = build_workspace_hosting_service(entrypoint)
            except Exception:
                if log is not None:
                    log.warning(
                        "Claude Code turn-workspace publisher unavailable",
                        exc_info=True,
                    )

    comm_service = getattr(getattr(host, "comm", None), "service", None) or {}
    resolved_user_type = str(
        user_type
        or (comm_service.get("user_type") if isinstance(comm_service, dict) else "")
        or turn_state.get("user_type")
        or "registered"
    )
    resolved_request_id = str(
        request_id
        or (comm_service.get("request_id") if isinstance(comm_service, dict) else "")
        or f"claude-code:{conversation_id}:{turn_id}"
    )

    async def _publish(*, paths: list[str]) -> list[dict[str, Any]]:
        return await publish_workspace_files(
            paths,
            workspace=root,
            turn_id=turn_id,
            hosting_service=host,
            tenant=tenant,
            project=project,
            user_id=user_id,
            user_type=resolved_user_type,
            conversation_id=conversation_id,
            request_id=resolved_request_id,
            state=turn_state,
            policy=publication_policy,
        )

    turn_summary_available = bool(turn_summary_enabled and isinstance(state, dict))

    async def _record_turn_summary(
        *,
        summary: str,
        refs: Any = None,
        phrases: Any = None,
        entities: Any = None,
    ) -> Dict[str, Any]:
        return stage_turn_summary(
            turn_state,
            summary=summary,
            refs=refs,
            phrases=phrases,
            entities=entities,
            contributor="claude_code",
        )

    broker = await start_workspace_broker(
        source_resolver=resolver,
        publisher=_publish if host is not None else None,
        summary_contributor=_record_turn_summary if turn_summary_available else None,
    )
    server_spec = workspace_mcp_server(
        workspace=root,
        tenant=tenant,
        project=project,
        user_id=user_id,
        conversation_id=conversation_id,
        turn_id=turn_id,
        storage_path=storage_path,
        python_executable=python_executable,
        broker_socket=str(broker.socket_path),
        broker_token=broker.token,
    )
    owner_task = asyncio.current_task()
    binding = ClaudeCodeTurnWorkspaceBinding(
        workspace=root,
        turn_id=str(turn_id),
        broker=broker,
        server_spec=server_spec,
        publication_available=host is not None,
        turn_summary_available=turn_summary_available,
        _owner_task=owner_task,
    )
    if owner_task is not None:
        loop = asyncio.get_running_loop()

        def _close_on_owner_done(_task: Any) -> None:
            if not binding.closed and not loop.is_closed():
                loop.create_task(binding.broker.close())

        binding._owner_callback = _close_on_owner_done
        owner_task.add_done_callback(_close_on_owner_done)
    return binding


__all__ = [
    "CLAUDE_CODE_TURN_WORKSPACE_PERMISSION",
    "ClaudeCodeTurnWorkspaceBinding",
    "bind_claude_code_turn_workspace",
]
