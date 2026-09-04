# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""One hosted Claude Code turn, composed from reusable Agent Harness seams."""

from __future__ import annotations

import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable

from kdcube_ai_app.apps.chat.sdk.config import get_secret
from kdcube_ai_app.apps.chat.sdk.protocol import external_events_texts
from kdcube_ai_app.apps.chat.sdk.runtime import comm_ctx
from kdcube_ai_app.apps.chat.sdk.solutions.claude_code import (
    ClaudeCodeAgent,
    ClaudeCodeAgentConfig,
    ClaudeCodeBinding,
    ClaudeCodeSessionStoreConfig,
    ClaudeCodeWorkspaceConfig,
    bind_claude_code_turn_workspace,
    prepare_claude_code_workspace,
    run_claude_code_turn,
)
from kdcube_ai_app.apps.chat.sdk.solutions.claude_code.live_control import (
    delivered_message_ids,
    publish_arrivals,
    seed_live_control,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.external_events import (
    fold_turn_external_events,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.live_watch import (
    LANE_MESSAGE_ID_KEY,
    LiveLaneWatch,
    event_is_steer,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.mcp_bridge import (
    current_turn_user_sub,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.turn_record import (
    conversation_is_new,
    emit_turn_timing,
    finalize_conversation_title,
)
from kdcube_ai_app.apps.chat.sdk.util import _now_ms

from .artifact_writer import PDF_NAME, XLSX_NAME


LOGGER = logging.getLogger("harness_claude_demo.turn")
BUNDLE_ID = "harness-claude-demo@1-0"
AGENT_ID = "claude"
TITLE_ROLE = "harness.claude.title"
DEFAULT_ALLOWED_TOOLS = (
    "Read",
    "Grep",
    "Glob",
    "Edit",
    "Write",
    "Bash",
    "WebSearch",
    "WebFetch",
)
ALLOWED_CREDENTIAL_ENVS = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_KEY",
}
INSTRUCTIONS = f"""# KDCube Agent Harness Claude demonstration

You are the Claude Code agent hosted by the KDCube Agent Harness. Work through
real tools and report their actual outcomes. Do not claim that a tool ran when
it did not.

For release research, use WebSearch and WebFetch and prefer primary python.org
pages. Preserve source URLs in your answer and in `research.json`.

When asked to produce the demonstration files:

1. Write `research.json` in the workspace with keys `title`, `version`,
   `release_date`, `highlights` (an array of strings), and `sources` (an array
   of objects with `label` and `url`).
2. Read the active physical turn root from the KDCube turn-workspace block in
   this file. Run `python3 artifact_writer.py --input research.json
   --output-dir <active-turn-root>/files`.
3. Confirm that `{PDF_NAME}` and `{XLSX_NAME}` exist.
4. Call the turn workspace `publish` tool once with the stable paths
   `files/{PDF_NAME}` and `files/{XLSX_NAME}`. A local file is not delivered
   until publication succeeds.

The platform owns caller identity, conversation durability, publication policy,
economics, and the communicator timeline. Never copy credentials into files or
output.
"""


def _agent_prop(entrypoint: Any, key: str, default: Any = None) -> Any:
    try:
        return entrypoint.bundle_prop(f"agent.{key}", default)
    except Exception:
        return default


def _safe_segment(value: Any, *, fallback: str) -> str:
    raw = str(value or "").strip()
    safe = "".join(char if char.isalnum() or char in "-_." else "-" for char in raw).strip("-_.")
    return safe or fallback


def _string_sequence(value: Any, default: Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, Iterable):
        values = value
    else:
        values = default
    normalized = tuple(str(item).strip() for item in values if str(item).strip())
    return normalized or tuple(default)


def _turn_user(entrypoint: Any, state: Dict[str, Any]) -> str:
    identity = entrypoint.runtime_identity() or {}
    return str(
        current_turn_user_sub(entrypoint)
        or state.get("economics_user")
        or state.get("authority_user")
        or state.get("actor_user")
        or state.get("user")
        or identity.get("user")
        or identity.get("fingerprint")
        or "anonymous"
    ).strip() or "anonymous"


def _copy_support_files(workspace: Path) -> None:
    source = Path(__file__).with_name("artifact_writer.py")
    target = workspace / "artifact_writer.py"
    if not target.exists() or target.read_bytes() != source.read_bytes():
        shutil.copy2(source, target)


async def _credential_env(entrypoint: Any) -> Dict[str, str]:
    ref = str(_agent_prop(entrypoint, "credential_ref", "b:agent.claude_code_key") or "").strip()
    env_name = str(_agent_prop(entrypoint, "credential_env", "CLAUDE_CODE_KEY") or "").strip()
    if env_name not in ALLOWED_CREDENTIAL_ENVS:
        raise ValueError(
            "agent.credential_env must be one of: " + ", ".join(sorted(ALLOWED_CREDENTIAL_ENVS))
        )
    value = str(await get_secret(ref, bundle_id=BUNDLE_ID) or "").strip() if ref else ""
    if not value or (value.startswith("<") and value.endswith(">")):
        raise ValueError(
            f"No Claude Code credential resolved from {ref!r}. Configure the secret reference in bundles.yaml "
            "and its value in bundles.secrets.yaml or the deployment secrets provider."
        )
    return {env_name: value}


async def _answer_in_band(state: Dict[str, Any], text: str) -> Dict[str, Any]:
    try:
        await comm_ctx.delta(text=text, index=0, marker="answer")
        await comm_ctx.delta(text="", index=1, marker="answer", completed=True)
    except Exception:
        LOGGER.warning("Could not stream the setup failure", exc_info=True)
    state["final_answer"] = text
    return state


def _prompt(question: str, *, turn_id: str) -> str:
    return (
        f"[KDCube turn {turn_id}]\n\n"
        f"{question.strip()}\n\n"
        "Use the KDCube instructions in CLAUDE.md. Work from observed tool results, keep source URLs, "
        "and publish only the files explicitly requested by the user."
    )


async def run_claude_demo_turn(
    entrypoint: Any,
    *,
    state: Dict[str, Any],
    thread_id: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    del params
    started_ms = _now_ms()
    started = time.perf_counter()
    events = await fold_turn_external_events(entrypoint, state)
    state["external_events"] = events
    question = "\n\n".join(external_events_texts(events)).strip()
    if not question:
        return await _answer_in_band(state, "This turn did not contain a text request.")

    conversation_id = str(thread_id or state.get("conversation_id") or state.get("session_id") or "").strip()
    turn_id = str(state.get("turn_id") or uuid.uuid4()).strip()
    if not conversation_id:
        return await _answer_in_band(state, "The runtime did not bind a conversation id to this turn.")

    storage_root = entrypoint.bundle_storage_root()
    if storage_root is None:
        return await _answer_in_band(
            state,
            "The app has no bundle storage for its Claude Code conversation workspace.",
        )

    identity = entrypoint.runtime_identity() or {}
    tenant = str(identity.get("tenant") or getattr(entrypoint.settings, "TENANT", "") or "default")
    project = str(identity.get("project") or getattr(entrypoint.settings, "PROJECT", "") or "default")
    user_id = _turn_user(entrypoint, state)
    workspace = (
        Path(storage_root)
        / "agent_workspaces"
        / _safe_segment(user_id, fallback="anonymous")
        / _safe_segment(conversation_id, fallback="conversation")
    )
    workspace.mkdir(parents=True, exist_ok=True)
    _copy_support_files(workspace)

    is_new = await conversation_is_new(entrypoint, state, conversation_id=conversation_id)
    await finalize_conversation_title(
        entrypoint,
        state,
        conversation_id=conversation_id,
        question=question,
        title_role=TITLE_ROLE,
    )

    try:
        env = await _credential_env(entrypoint)
    except ValueError as exc:
        return await _answer_in_band(state, str(exc))

    turn_workspace = await bind_claude_code_turn_workspace(
        workspace=workspace,
        tenant=tenant,
        project=project,
        user_id=user_id,
        conversation_id=conversation_id,
        turn_id=turn_id,
        entrypoint=entrypoint,
        state=state,
        turn_summary_enabled=True,
        request_id=f"harness-claude-demo:{conversation_id}:{turn_id}",
        logger=LOGGER,
        tool_call_id="harness-claude-demo-workspace",
    )
    allowed_tools = _string_sequence(
        _agent_prop(entrypoint, "allowed_tools", DEFAULT_ALLOWED_TOOLS),
        DEFAULT_ALLOWED_TOOLS,
    )
    workspace_config = turn_workspace.apply_workspace_config(
        ClaudeCodeWorkspaceConfig(
            allowed_tools=allowed_tools,
            instructions_markdown=INSTRUCTIONS,
        )
    )

    def _prepare_workspace() -> None:
        _copy_support_files(workspace)
        prepare_claude_code_workspace(workspace, workspace_config)

    _prepare_workspace()
    settings = entrypoint.settings
    implementation = str(
        getattr(settings, "CLAUDE_CODE_SESSION_STORE_IMPLEMENTATION", "local") or "local"
    ).strip().lower()
    git_repo = str(getattr(settings, "CLAUDE_CODE_SESSION_GIT_REPO", "") or "").strip() or None
    session_store = ClaudeCodeSessionStoreConfig(
        implementation=implementation,
        local_root=workspace / ".claude",
        tenant=tenant,
        project=project,
        user_id=user_id,
        conversation_id=conversation_id,
        agent_name=AGENT_ID,
        git_repo=git_repo,
    )
    binding = ClaudeCodeBinding(
        user_id=user_id,
        conversation_id=conversation_id,
        session_id=str(state.get("session_id") or conversation_id),
        claude_session_id=str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"kdcube/harness-claude-demo/{tenant}/{project}/{user_id}/{conversation_id}",
            )
        ),
    )
    control_args = (
        seed_live_control(workspace, turn_id=turn_id)
        if bool(_agent_prop(entrypoint, "live_control", True))
        else []
    )
    runner_logger = (
        entrypoint.logger
        if isinstance(getattr(entrypoint, "logger", None), logging.Logger)
        else None
    )
    agent = ClaudeCodeAgent(
        config=ClaudeCodeAgentConfig(
            agent_name=AGENT_ID,
            workspace_path=workspace,
            model=str(_agent_prop(entrypoint, "model", "claude-sonnet-4-6") or "").strip() or None,
            allowed_tools=tuple(workspace_config.allowed_tools),
            extra_args=tuple(control_args),
            env=env,
            command=str(_agent_prop(entrypoint, "command", "claude") or "claude"),
            permission_mode="acceptEdits",
            timeout_seconds=float(_agent_prop(entrypoint, "timeout_seconds", 900) or 900),
            step_name="harness.claude",
            workspace_config=workspace_config,
            log_stream_output=True,
        ),
        binding=binding,
        logger=runner_logger,
    )

    def _deliver(arrivals: list[Dict[str, Any]]) -> None:
        publish_arrivals(
            workspace,
            [
                {
                    "text": "\n\n".join(external_events_texts([body])),
                    "message_id": str(body.get(LANE_MESSAGE_ID_KEY) or ""),
                }
                for body in arrivals
                if external_events_texts([body])
            ],
            stop=any(event_is_steer(body) for body in arrivals),
            turn_id=turn_id,
        )

    try:
        async with LiveLaneWatch(entrypoint, state, on_arrival=_deliver) as watch:
            result = await run_claude_code_turn(
                agent=agent,
                prompt=_prompt(question, turn_id=turn_id),
                kind="regular",
                resume_existing=not is_new,
                session_store=session_store,
                refresh_support_files=_prepare_workspace,
                logger=runner_logger,
            )
        delivered = delivered_message_ids(workspace)
        if delivered:
            await watch.consume_delivered(delivered)
    finally:
        await turn_workspace.close()

    final_text = str(getattr(result, "final_text", "") or "").strip()
    if getattr(result, "status", "") == "completed" and final_text:
        state["final_answer"] = final_text
    else:
        detail = str(
            getattr(result, "error_message", None)
            or getattr(result, "failure_diagnostics", None)
            or f"status={getattr(result, 'status', 'unknown')}, exit_code={getattr(result, 'exit_code', None)}"
        )
        await _answer_in_band(
            state,
            "The Claude Code run did not produce an answer. "
            f"{detail}. The platform did not silently retry it.",
        )

    await emit_turn_timing(
        entrypoint,
        started_ms=started_ms,
        total_ms=int((time.perf_counter() - started) * 1000),
    )
    return state
