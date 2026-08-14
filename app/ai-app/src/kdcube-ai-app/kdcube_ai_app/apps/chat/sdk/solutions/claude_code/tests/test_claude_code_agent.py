# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from kdcube_ai_app.apps.chat.emitters import ChatCommunicator
from kdcube_ai_app.apps.chat.sdk.protocol import (
    ExternalEventActor,
    ExternalEventPayload,
    ExternalEventRequest,
    ExternalEventRouting,
    ExternalEventUser,
)
from kdcube_ai_app.apps.chat.sdk.runtime.comm_ctx import bind_current_request_context
from kdcube_ai_app.apps.chat.sdk.skills.skills_registry import set_skills_descriptor
from kdcube_ai_app.apps.chat.sdk.solutions.claude_code import (
    ClaudeCodeAgent,
    ClaudeCodeWorkspaceConfig,
    prepare_claude_code_workspace,
)
from kdcube_ai_app.apps.chat.sdk.solutions.claude_code.types import ClaudeCodeAgentConfig, ClaudeCodeBinding
from kdcube_ai_app.infra.accounting import AccountingSystem, clear_context


class _RecordingEmitter:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def emit(self, *, event: str, data: dict, **kwargs) -> None:
        del kwargs
        self.events.append((event, data))


class _FakeProcess:
    def __init__(
        self,
        *,
        stdout_lines: list[str],
        stderr_lines: list[str],
        returncode: int,
        wait_delay: float = 0.0,
    ):
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.returncode = None
        self._stdout_lines = list(stdout_lines)
        self._stderr_lines = list(stderr_lines)
        self._planned_returncode = returncode
        self._wait_delay = wait_delay
        self._done = asyncio.Event()
        self._task = asyncio.create_task(self._feed())

    async def _feed(self) -> None:
        for line in self._stdout_lines:
            self.stdout.feed_data(line.encode("utf-8"))
            await asyncio.sleep(0)
        self.stdout.feed_eof()
        for line in self._stderr_lines:
            self.stderr.feed_data(line.encode("utf-8"))
            await asyncio.sleep(0)
        self.stderr.feed_eof()
        if self._wait_delay > 0:
            try:
                await asyncio.wait_for(self._done.wait(), timeout=self._wait_delay)
            except asyncio.TimeoutError:
                pass
        if self.returncode is None:
            self.returncode = self._planned_returncode
        self._done.set()

    async def wait(self) -> int:
        await asyncio.shield(self._task)
        await self._done.wait()
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        if self.returncode is None:
            self.returncode = -15
        self._done.set()

    def kill(self) -> None:
        if self.returncode is None:
            self.returncode = -9
        self._done.set()


class _RecordingAccountingBackend:
    def __init__(self) -> None:
        self.writes: list[tuple[str, str]] = []

    async def write_text_a(self, path: str, content: str) -> None:
        self.writes.append((path, content))


def _ctx() -> ExternalEventPayload:
    return ExternalEventPayload(
        request=ExternalEventRequest(request_id="req-claude-code"),
        routing=ExternalEventRouting(
            session_id="sid-claude",
            conversation_id="conv-claude",
            turn_id="turn-claude",
            bundle_id="bundle.claude",
        ),
        actor=ExternalEventActor(
            tenant_id="demo-tenant",
            project_id="demo-project",
        ),
        user=ExternalEventUser(
            user_type="privileged",
            user_id="admin-user-1",
            fingerprint="fingerprint-1",
            username="admin",
            roles=["kdcube:role:super-admin"],
            permissions=["kdcube:*:chat:*;read;write;delete"],
            timezone="UTC",
        ),
    )


def _make_comm() -> tuple[ChatCommunicator, _RecordingEmitter]:
    emitter = _RecordingEmitter()
    comm = ChatCommunicator(
        emitter=emitter,
        tenant="demo-tenant",
        project="demo-project",
        user_id="admin-user-1",
        user_type="privileged",
        service={
            "request_id": "req-claude-code",
            "tenant": "demo-tenant",
            "project": "demo-project",
            "user": "admin-user-1",
        },
        conversation={
            "session_id": "sid-claude",
            "conversation_id": "conv-claude",
            "turn_id": "turn-claude",
        },
    )
    return comm, emitter


def _config(workspace_path: Path) -> ClaudeCodeAgentConfig:
    return ClaudeCodeAgentConfig(
        agent_name="kb-writer",
        workspace_path=workspace_path,
        allowed_tools=("Read", "Grep", "WebSearch"),
        additional_directories=(workspace_path / "repos" / "output",),
        extra_args=("--append-system-prompt", "Stay concise."),
        env={"EXTRA_ENV": "yes"},
    )


@pytest.mark.asyncio
async def test_from_current_context_derives_deterministic_binding(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    comm, _ = _make_comm()
    ctx = _ctx()

    with bind_current_request_context(ctx, comm=comm):
        first = ClaudeCodeAgent.from_current_context(
            agent_name="kb-writer",
            workspace_path=workspace,
        )
        second = ClaudeCodeAgent.from_current_context(
            agent_name="kb-writer",
            workspace_path=workspace,
        )

    assert first.binding == second.binding
    assert first.binding.user_id == "admin-user-1"
    assert first.binding.conversation_id == "conv-claude"
    assert first.binding.session_id == "sid-claude"
    assert first.binding.claude_session_id


def test_build_args_includes_session_allowed_tools_and_agent(tmp_path: Path):
    workspace = tmp_path / "workspace"
    (workspace / "repos" / "output").mkdir(parents=True)
    # --agent is only passed when the named agent is defined in the workspace.
    agents_dir = workspace / ".claude" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "kb-writer.md").write_text("# Agent\n", encoding="utf-8")
    binding = ClaudeCodeBinding(
        user_id="admin-user-1",
        conversation_id="conv-claude",
        session_id="sid-claude",
        claude_session_id="claude-session-1",
    )
    agent = ClaudeCodeAgent(config=_config(workspace), binding=binding, comm=None)

    args = agent.build_args("Explain the repo")
    resume_args = agent.build_args("Explain the repo", resume_existing=True)

    assert "--allowedTools" in args
    assert "Read,Grep,WebSearch" in args
    assert "--permission-mode" in args
    assert "acceptEdits" in args
    assert "--add-dir" in args
    assert str(workspace / "repos" / "output") in args
    assert "--session-id" in args
    assert "claude-session-1" in args
    assert "--resume" in resume_args
    assert "claude-session-1" in resume_args
    assert "--session-id" not in resume_args
    assert "--agent" in args
    assert "kb-writer" in args
    assert args[-1] == "Explain the repo"


def test_build_args_omits_agent_when_no_definition(tmp_path: Path):
    """agent_name may be used purely as a label. With no
    .claude/agents/<name>.md definition the CLI rejects --agent, so the runner
    omits it and runs the default agent with the seeded instructions."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    binding = ClaudeCodeBinding(
        user_id="admin-user-1",
        conversation_id="conv-claude",
        session_id="sid-claude",
        claude_session_id="claude-session-1",
    )
    agent = ClaudeCodeAgent(config=_config(workspace), binding=binding, comm=None)

    args = agent.build_args("Explain the repo")

    assert "--agent" not in args
    assert "kb-writer" not in args
    assert args[-1] == "Explain the repo"


@pytest.mark.asyncio
async def test_run_turn_closes_stdin_when_prompt_is_passed_as_argument(monkeypatch, tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    comm, _ = _make_comm()
    ctx = _ctx()
    captured_kwargs: list[dict] = []

    async def _fake_create_subprocess_exec(*args, **kwargs):
        del args
        captured_kwargs.append(kwargs)
        return _FakeProcess(stdout_lines=[], stderr_lines=[], returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    with bind_current_request_context(ctx, comm=comm):
        agent = ClaudeCodeAgent.from_current_context(
            agent_name="kb-writer",
            workspace_path=workspace,
        )
        await agent.run_turn("Explain the repo")

    assert captured_kwargs
    assert captured_kwargs[0]["stdin"] is asyncio.subprocess.DEVNULL


def test_build_args_includes_selected_model(tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    binding = ClaudeCodeBinding(
        user_id="admin-user-1",
        conversation_id="conv-claude",
        session_id="sid-claude",
        claude_session_id="claude-session-1",
    )
    config = ClaudeCodeAgentConfig(
        agent_name="kb-writer",
        workspace_path=workspace,
        model="claude-opus-4-6",
    )
    agent = ClaudeCodeAgent(config=config, binding=binding, comm=None)

    args = agent.build_args("Explain the repo")

    assert "--model" in args
    assert "claude-opus-4-6" in args


def test_prepare_claude_code_workspace_writes_mcp_settings_and_instructions(tmp_path: Path):
    workspace = tmp_path / "workspace"

    prepared = prepare_claude_code_workspace(
        workspace,
        ClaudeCodeWorkspaceConfig(
            mcp_servers={
                "scoped_data": {
                    "type": "http",
                    "url": "http://127.0.0.1:8020/mcp",
                    "headers": {"X-Test-Token": "token"},
                }
            },
            allowed_tools=("mcp__scoped_data__task_context",),
            denied_tools=("Bash", "Read"),
            instructions_markdown="# Scoped Data\nUse only MCP tools.\n",
        ),
    )

    assert prepared["mcp_servers"] == ["scoped_data"]
    assert (workspace / ".mcp.json").exists()
    assert (workspace / ".claude" / "settings.local.json").exists()
    assert (workspace / "CLAUDE.md").read_text(encoding="utf-8") == "# Scoped Data\nUse only MCP tools.\n"

    mcp_config = json.loads((workspace / ".mcp.json").read_text(encoding="utf-8"))
    assert mcp_config["mcpServers"]["scoped_data"]["headers"]["X-Test-Token"] == "token"

    settings = json.loads((workspace / ".claude" / "settings.local.json").read_text(encoding="utf-8"))
    assert settings["enableAllProjectMcpServers"] is False
    assert settings["enabledMcpjsonServers"] == ["scoped_data"]
    assert settings["permissions"]["allow"] == ["mcp__scoped_data__task_context"]
    assert settings["permissions"]["deny"] == ["Bash", "Read"]


def test_prepare_claude_code_workspace_materializes_kdcube_skills(tmp_path: Path):
    skills_root = tmp_path / "bundle" / "skills"
    skill_dir = skills_root / "product" / "email-analysis"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: email-analysis",
                "description: Analyze scoped email candidates.",
                "namespace: product",
                "when_to_use:",
                "  - Classifying email messages",
                "---",
                "",
                "# Email Analysis",
                "",
                "Use the scoped email MCP tools only.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (skill_dir / "reference.md").write_text("Reference notes", encoding="utf-8")
    workspace = tmp_path / "workspace"

    try:
        set_skills_descriptor({"CUSTOM_SKILLS_ROOT": str(skills_root)})
        prepared = prepare_claude_code_workspace(
            workspace,
            ClaudeCodeWorkspaceConfig(
                skill_ids=("product.email-analysis",),
                skill_allowed_tools={
                    "product.email-analysis": (
                        "mcp__email__task_context",
                        "mcp__email__record_processing_result",
                    )
                },
            ),
        )
    finally:
        set_skills_descriptor(None)

    skill_path = workspace / ".claude" / "skills" / "product-email-analysis" / "SKILL.md"
    support_path = workspace / ".claude" / "skills" / "product-email-analysis" / "reference.md"
    skill_text = skill_path.read_text(encoding="utf-8")

    assert prepared["materialized_skill_ids"] == ["product.email-analysis"]
    assert skill_path.exists()
    assert support_path.read_text(encoding="utf-8") == "Reference notes"
    assert "name: email-analysis" in skill_text
    assert "Analyze scoped email candidates." in skill_text
    assert "Classifying email messages" in skill_text
    assert "allowed-tools: mcp__email__task_context, mcp__email__record_processing_result" in skill_text
    assert "Use the scoped email MCP tools only." in skill_text


@pytest.mark.asyncio
async def test_run_turn_streams_incremental_deltas(monkeypatch, tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    comm, emitter = _make_comm()
    ctx = _ctx()

    outputs = [
        json.dumps({"message": {"content": [{"type": "text", "text": "Hello"}]}}) + "\n",
        json.dumps({"message": {"content": [{"type": "text", "text": "Hello world"}]}}) + "\n",
    ]

    async def _fake_create_subprocess_exec(*args, **kwargs):
        del args, kwargs
        return _FakeProcess(stdout_lines=outputs, stderr_lines=[], returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    with bind_current_request_context(ctx, comm=comm):
        agent = ClaudeCodeAgent.from_current_context(
            agent_name="kb-writer",
            workspace_path=workspace,
            allowed_tools=("Read", "WebSearch"),
        )
        result = await agent.run_turn("Summarize repo")

    deltas = [
        envelope["delta"]["text"]
        for _, envelope in emitter.events
        if envelope.get("type") == "chat.delta"
    ]
    steps = [
        envelope
        for _, envelope in emitter.events
        if envelope.get("type") == "chat.step"
    ]

    assert result.status == "completed"
    assert result.final_text == "Hello world"
    assert result.delta_count == 2
    assert deltas == ["Hello", " world"]
    assert steps[0]["event"]["status"] == "started"
    assert steps[0]["data"]["turn_kind"] == "regular"
    assert steps[-1]["event"]["status"] == "completed"


@pytest.mark.asyncio
async def test_run_turn_handles_large_single_stdout_json_line(monkeypatch, tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    comm, emitter = _make_comm()
    ctx = _ctx()
    large_text = "x" * (96 * 1024)
    outputs = [
        json.dumps({"message": {"content": [{"type": "text", "text": large_text}]}}) + "\n",
    ]

    async def _fake_create_subprocess_exec(*args, **kwargs):
        del args, kwargs
        return _FakeProcess(stdout_lines=outputs, stderr_lines=[], returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    with bind_current_request_context(ctx, comm=comm):
        agent = ClaudeCodeAgent.from_current_context(
            agent_name="kb-writer",
            workspace_path=workspace,
            allowed_tools=("Read",),
        )
        result = await agent.run_turn("Summarize repo")

    deltas = [
        envelope["delta"]["text"]
        for _, envelope in emitter.events
        if envelope.get("type") == "chat.delta"
    ]

    assert result.status == "completed"
    assert result.final_text == large_text
    assert result.delta_count == 1
    assert deltas == [large_text]


@pytest.mark.asyncio
async def test_run_turn_preserves_multiple_distinct_claude_messages(monkeypatch, tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    comm, emitter = _make_comm()
    ctx = _ctx()

    outputs = [
        json.dumps({"message": {"content": [{"type": "text", "text": "First message"}]}}) + "\n",
        json.dumps({"message": {"content": [{"type": "text", "text": "Second message"}]}}) + "\n",
    ]

    async def _fake_create_subprocess_exec(*args, **kwargs):
        del args, kwargs
        return _FakeProcess(stdout_lines=outputs, stderr_lines=[], returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    with bind_current_request_context(ctx, comm=comm):
        agent = ClaudeCodeAgent.from_current_context(
            agent_name="kb-writer",
            workspace_path=workspace,
            allowed_tools=("Read",),
        )
        result = await agent.run_turn("Summarize repo")

    deltas = [
        envelope["delta"]["text"]
        for _, envelope in emitter.events
        if envelope.get("type") == "chat.delta"
    ]

    assert result.status == "completed"
    assert result.final_text == "First message\n\nSecond message"
    assert deltas == ["First message", "\n\nSecond message"]


@pytest.mark.asyncio
async def test_run_turn_emits_stderr_and_failure_step(monkeypatch, tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    comm, emitter = _make_comm()
    ctx = _ctx()

    outputs = [
        json.dumps(
            {
                "type": "result",
                "subtype": "error",
                "duration_ms": 800,
                "total_cost_usd": 0.001,
            }
        )
        + "\n",
    ]

    async def _fake_create_subprocess_exec(*args, **kwargs):
        del args, kwargs
        return _FakeProcess(stdout_lines=outputs, stderr_lines=["fatal: boom\n"], returncode=1)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    with bind_current_request_context(ctx, comm=comm):
        agent = ClaudeCodeAgent.from_current_context(
            agent_name="kb-writer",
            workspace_path=workspace,
        )
        result = await agent.run_followup("Continue")

    steps = [
        envelope
        for _, envelope in emitter.events
        if envelope.get("type") == "chat.step"
    ]

    assert result.status == "failed"
    assert result.turn_kind == "followup"
    assert any(step["event"]["step"] == "claude_code.agent.stderr" for step in steps)
    assert steps[-1]["event"]["status"] == "error"
    assert steps[-1]["data"]["error"] == "fatal: boom"
    assert steps[-1]["data"]["last_stderr_line"] == "fatal: boom"
    assert steps[-1]["data"]["raw_result_event"] == {
        "type": "result",
        "subtype": "error",
        "duration_ms": 800,
        "total_cost_usd": 0.001,
    }


@pytest.mark.asyncio
async def test_run_turn_collects_structured_events_from_streamed_chunks(monkeypatch, tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    comm, emitter = _make_comm()
    ctx = _ctx()
    observed: list[dict] = []

    outputs = [
        json.dumps({"message": {"content": [{"type": "text", "text": "NEWS_PIPELINE_EVENT {\"type\":\"phase\""}]}}) + "\n",
        json.dumps({"message": {"content": [{"type": "text", "text": "NEWS_PIPELINE_EVENT {\"type\":\"phase\",\"phase\":\"draft_markdown\",\"status\":\"started\"}\nDone"}]}}) + "\n",
    ]

    async def _fake_create_subprocess_exec(*args, **kwargs):
        del args, kwargs
        return _FakeProcess(stdout_lines=outputs, stderr_lines=[], returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    with bind_current_request_context(ctx, comm=comm):
        agent = ClaudeCodeAgent.from_current_context(
            agent_name="kb-writer",
            workspace_path=workspace,
            structured_output_prefixes=("NEWS_PIPELINE_EVENT",),
            on_structured_output=lambda event: observed.append(event),
        )
        result = await agent.run_turn("Summarize repo")

    completed_steps = [
        envelope
        for _, envelope in emitter.events
        if envelope.get("type") == "chat.step" and envelope.get("event", {}).get("status") == "completed"
    ]

    assert result.status == "completed"
    assert result.structured_events == [
        {
            "prefix": "NEWS_PIPELINE_EVENT",
            "payload": {"type": "phase", "phase": "draft_markdown", "status": "started"},
            "raw_line": "NEWS_PIPELINE_EVENT {\"type\":\"phase\",\"phase\":\"draft_markdown\",\"status\":\"started\"}",
        }
    ]
    assert observed == result.structured_events
    assert completed_steps[-1]["data"]["structured_events"] == result.structured_events


@pytest.mark.asyncio
async def test_run_turn_collects_executive_journal_from_standard_prefix(monkeypatch, tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    comm, emitter = _make_comm()
    ctx = _ctx()

    outputs = [
        json.dumps({"message": {"content": [{"type": "text", "text": "EXECUTIVE_JOURNAL searched scoped messages; found two candidates\nStill working"}]}}) + "\n",
        json.dumps({"message": {"content": [{"type": "text", "text": "EXECUTIVE_JOURNAL_CODE print('checkpoint')\nDone"}]}}) + "\n",
    ]

    async def _fake_create_subprocess_exec(*args, **kwargs):
        del args, kwargs
        return _FakeProcess(stdout_lines=outputs, stderr_lines=[], returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    with bind_current_request_context(ctx, comm=comm):
        agent = ClaudeCodeAgent.from_current_context(
            agent_name="kb-writer",
            workspace_path=workspace,
        )
        result = await agent.run_turn("Summarize repo")

    completed_steps = [
        envelope
        for _, envelope in emitter.events
        if envelope.get("type") == "chat.step" and envelope.get("event", {}).get("status") == "completed"
    ]

    assert result.status == "completed"
    assert len(result.executive_journal) == 2
    assert result.executive_journal[0]["prefix"] == "EXECUTIVE_JOURNAL"
    assert result.executive_journal[0]["channel"] == "note"
    assert result.executive_journal[0]["text"] == "searched scoped messages; found two candidates"
    assert result.executive_journal[0]["captured_at"]
    assert result.executive_journal[1]["prefix"] == "EXECUTIVE_JOURNAL_CODE"
    assert result.executive_journal[1]["channel"] == "code"
    assert result.executive_journal[1]["code"] == "print('checkpoint')"
    assert completed_steps[-1]["data"]["executive_journal"] == result.executive_journal


@pytest.mark.asyncio
async def test_run_turn_timeout_marks_failure_and_emits_timeout_metadata(monkeypatch, tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    comm, emitter = _make_comm()
    ctx = _ctx()

    async def _fake_create_subprocess_exec(*args, **kwargs):
        del args, kwargs
        return _FakeProcess(
            stdout_lines=["working on the mailbox\n"],
            stderr_lines=["warning: still running\n"],
            returncode=0,
            wait_delay=5.0,
        )

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    with bind_current_request_context(ctx, comm=comm):
        agent = ClaudeCodeAgent.from_current_context(
            agent_name="kb-writer",
            workspace_path=workspace,
            timeout_seconds=0.01,
        )
        result = await agent.run_turn("Summarize repo")

    error_steps = [
        envelope
        for _, envelope in emitter.events
        if envelope.get("type") == "chat.step" and envelope.get("event", {}).get("status") == "error"
    ]

    assert result.status == "failed"
    assert result.timed_out is True
    assert result.timeout_seconds == pytest.approx(0.01)
    assert result.duration_ms is not None
    assert result.duration_ms >= 0
    assert "timeout" in (result.error_message or "").lower()
    assert result.failure_diagnostics["reason"] == "timeout_waiting_for_process_result"
    assert result.failure_diagnostics["stdout_tail"][-1] == "working on the mailbox"
    assert result.failure_diagnostics["stderr_tail"][-1] == "warning: still running"
    assert result.failure_diagnostics["raw_result_event_seen"] is False
    assert error_steps[-1]["data"]["timed_out"] is True
    assert error_steps[-1]["data"]["timeout_seconds"] == pytest.approx(0.01)
    assert error_steps[-1]["data"]["failure_diagnostics"]["reason"] == "timeout_waiting_for_process_result"


@pytest.mark.asyncio
async def test_followup_and_steer_resume_same_claude_session_id(monkeypatch, tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    comm, emitter = _make_comm()
    ctx = _ctx()
    captured: list[tuple] = []

    async def _fake_create_subprocess_exec(*args, **kwargs):
        captured.append(args)
        del kwargs
        return _FakeProcess(stdout_lines=[], stderr_lines=[], returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    with bind_current_request_context(ctx, comm=comm):
        agent = ClaudeCodeAgent.from_current_context(
            agent_name="kb-writer",
            workspace_path=workspace,
        )
        first = await agent.run_followup("Continue")
        second = await agent.run_steer("Change direction")

    assert first.session_id == second.session_id
    assert captured
    session_ids = []
    for args in captured:
        idx = args.index("--resume")
        session_ids.append(args[idx + 1])
    assert len(set(session_ids)) == 1

    started_steps = [
        envelope["data"]["turn_kind"]
        for _, envelope in emitter.events
        if envelope.get("type") == "chat.step" and envelope.get("event", {}).get("status") == "started"
    ]
    assert started_steps == ["followup", "steer"]


@pytest.mark.asyncio
async def test_run_turn_emits_accounting_event_with_usage(monkeypatch, tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    comm, _ = _make_comm()
    ctx = _ctx()
    backend = _RecordingAccountingBackend()

    outputs = [
        json.dumps({"type": "system", "subtype": "init", "model": "sonnet"}) + "\n",
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "model": "claude-sonnet-4-5-20250929",
                    "usage": {
                        "input_tokens": 120,
                        "output_tokens": 30,
                        "cache_creation_input_tokens": 40,
                        "cache_read_input_tokens": 5,
                    },
                    "content": [{"type": "text", "text": "Done"}],
                },
            }
        )
        + "\n",
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "duration_ms": 1500,
                "duration_api_ms": 1200,
                "total_cost_usd": 0.0123,
            }
        )
        + "\n",
    ]

    async def _fake_create_subprocess_exec(*args, **kwargs):
        del args, kwargs
        return _FakeProcess(stdout_lines=outputs, stderr_lines=[], returncode=0)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    AccountingSystem.init_storage(
        backend,
        enabled=True,
        cache_in_memory=False,
        redis_turn_cache=False,
    )
    AccountingSystem.set_context(
        user_id="admin-user-1",
        session_id="sid-claude",
        tenant_id="demo-tenant",
        project_id="demo-project",
        request_id="req-claude-code",
        app_bundle_id="bundle.claude",
        component="bundle.claude",
    )

    try:
        with bind_current_request_context(ctx, comm=comm):
            agent = ClaudeCodeAgent.from_current_context(
                agent_name="kb-writer",
                workspace_path=workspace,
            )
            result = await agent.run_turn("Summarize repo")
    finally:
        clear_context()
        AccountingSystem.init_storage(None, enabled=False)

    assert result.status == "completed"
    assert result.model == "claude-sonnet-4-5-20250929"
    assert result.cost_usd == pytest.approx(0.0123)
    assert backend.writes, "Claude Code run should emit an accounting event"

    _, content = backend.writes[-1]
    event = json.loads(content)
    assert event["service_type"] == "llm"
    assert event["provider"] == "anthropic"
    assert event["model_or_service"] == "claude-sonnet-4-5-20250929"
    assert event["success"] is True
    assert event["usage"]["input_tokens"] == 120
    assert event["usage"]["output_tokens"] == 30
    assert event["usage"]["cache_creation_tokens"] == 40
    assert event["usage"]["cache_read_tokens"] == 5
    assert event["usage"]["cost_usd"] == pytest.approx(0.0123)
    assert event["metadata"]["agent"] == "kb-writer"
    assert event["metadata"]["agent_name"] == "kb-writer"
    assert event["metadata"]["runtime"] == "claude_code"
    assert event["metadata"]["turn_kind"] == "regular"


@pytest.mark.asyncio
async def test_failed_run_marks_accounting_event_unsuccessful(monkeypatch, tmp_path: Path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    comm, _ = _make_comm()
    ctx = _ctx()
    backend = _RecordingAccountingBackend()

    outputs = [
        json.dumps({"type": "system", "subtype": "init", "model": "haiku"}) + "\n",
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "model": "claude-haiku-4-5-20251001",
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 10,
                    },
                },
            }
        )
        + "\n",
        json.dumps(
            {
                "type": "result",
                "subtype": "error",
                "duration_ms": 800,
                "total_cost_usd": 0.001,
            }
        )
        + "\n",
    ]

    async def _fake_create_subprocess_exec(*args, **kwargs):
        del args, kwargs
        return _FakeProcess(stdout_lines=outputs, stderr_lines=["fatal: boom\n"], returncode=1)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_create_subprocess_exec)

    AccountingSystem.init_storage(
        backend,
        enabled=True,
        cache_in_memory=False,
        redis_turn_cache=False,
    )
    AccountingSystem.set_context(
        user_id="admin-user-1",
        session_id="sid-claude",
        tenant_id="demo-tenant",
        project_id="demo-project",
        request_id="req-claude-code",
        app_bundle_id="bundle.claude",
        component="bundle.claude",
    )

    try:
        with bind_current_request_context(ctx, comm=comm):
            agent = ClaudeCodeAgent.from_current_context(
                agent_name="kb-writer",
                workspace_path=workspace,
            )
            result = await agent.run_turn("Continue")
    finally:
        clear_context()
        AccountingSystem.init_storage(None, enabled=False)

    assert result.status == "failed"
    assert backend.writes, "Failed Claude Code run should still emit an accounting event"

    _, content = backend.writes[-1]
    event = json.loads(content)
    assert event["service_type"] == "llm"
    assert event["provider"] == "anthropic"
    assert event["success"] is False
    assert event["error_message"] == "fatal: boom"
    assert event["usage"]["input_tokens"] == 20
    assert event["usage"]["output_tokens"] == 10
    assert event["usage"]["cost_usd"] == pytest.approx(0.001)


# ── the reader gets the answer, not the workings ──────────────────────────────

def test_a_tool_result_never_reaches_the_answer():
    """REGRESSION (live): Claude Code reports every tool result back into its own
    conversation as a `user` event whose content is what the tool returned — a
    file read comes back line-numbered. The generic extractor streamed that into
    the chat, so a whole README arrived as the assistant's answer with line
    numbers down the side."""
    from kdcube_ai_app.apps.chat.sdk.solutions.claude_code.streaming import (
        extract_text_from_claude_event,
        extract_tool_uses_from_claude_event,
    )

    tool_result = {
        "type": "user",
        "message": {"role": "user", "content": [
            {"type": "tool_result", "content": "1\t# LinkedIn publications\n2\t\n3\tThe canonical store"},
        ]},
    }
    assert extract_text_from_claude_event(tool_result) == ""

    # the same shape nested one level deeper (an assistant echoing a result block)
    assert extract_text_from_claude_event(
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_result", "content": "1\tsecret file body"},
        ]}}
    ) == ""


def test_a_tool_call_is_visible_as_activity_not_as_text():
    from kdcube_ai_app.apps.chat.sdk.solutions.claude_code.streaming import (
        extract_text_from_claude_event,
        extract_tool_uses_from_claude_event,
    )

    event = {
        "type": "assistant",
        "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Read", "input": {"file_path": "/store/README.md"}},
        ]},
    }
    assert extract_text_from_claude_event(event) == ""      # not answer text
    calls = extract_tool_uses_from_claude_event(event)      # but visible as a step
    assert calls == [{"id": "", "name": "Read", "input": {"file_path": "/store/README.md"}}]


def test_the_agent_s_own_words_still_stream():
    from kdcube_ai_app.apps.chat.sdk.solutions.claude_code.streaming import (
        extract_text_from_claude_event,
    )
    assert extract_text_from_claude_event(
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "The test account is restricted."},
        ]}}
    ) == "The test account is restricted."
    # a bare result event (no type) keeps working — that is the final answer path
    assert extract_text_from_claude_event({"result": "done"}) == "done"


def test_a_tool_result_is_shown_as_activity_with_its_size():
    """Elena's rule: show the workings, do not swallow them. The result becomes
    an activity row carrying the head of the output and how much it stood for —
    which is also what makes a bad turn debuggable after the fact."""
    from kdcube_ai_app.apps.chat.sdk.solutions.claude_code.streaming import (
        TOOL_RESULT_PREVIEW_CHARS,
        extract_tool_results_from_claude_event,
    )

    body = "x" * (TOOL_RESULT_PREVIEW_CHARS + 500)
    results = extract_tool_results_from_claude_event({
        "type": "user",
        "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": body},
        ]},
    })
    assert len(results) == 1
    row = results[0]
    assert row["tool_use_id"] == "call_1"
    assert row["total_chars"] == len(body)
    assert row["truncated"] is True
    assert len(row["text"]) == TOOL_RESULT_PREVIEW_CHARS
    assert row["is_error"] is False

    # an errored tool is marked, not hidden
    err = extract_tool_results_from_claude_event({
        "type": "user",
        "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "c2", "content": "permission denied", "is_error": True},
        ]},
    })
    assert err[0]["is_error"] is True and err[0]["text"] == "permission denied"


def test_a_waiting_tool_is_visible_while_it_waits():
    """LIVE: an MCP call whose response was lost in transit sat at `running` for
    two minutes with nothing on screen and nothing in the log, and the user read
    it as "no response". The CLI knew — it heartbeats the pending call — so the
    heartbeat is what makes waiting legible, and it points at the PARENT call,
    not at its own synthetic heartbeat id."""
    from kdcube_ai_app.apps.chat.sdk.solutions.claude_code.streaming import (
        tool_progress_from_claude_event,
    )

    waiting = tool_progress_from_claude_event({
        "type": "tool_progress",
        "tool_use_id": "toolu_ABC-heartbeat-2",
        "parent_tool_use_id": "toolu_ABC",
        "tool_name": "mcp__press__search",
        "elapsed_time_seconds": 90,
        "heartbeat": True,
    })
    assert waiting == {
        "tool_use_id": "toolu_ABC",
        "tool_name": "mcp__press__search",
        "elapsed_seconds": 90,
    }
    assert tool_progress_from_claude_event({"type": "assistant"}) is None


def test_an_activity_row_says_what_the_agent_did():
    """LIVE: the Steps list read `Bash(command=<252 chars>, descripti…` — an
    argument dump, truncated before the part that matters. A row should read
    like a sentence: the tool, then the thing it acted on."""
    from kdcube_ai_app.apps.chat.sdk.solutions.claude_code.streaming import (
        claude_tool_activity_title,
    )

    # With a human description present, that is the row's subject; the command
    # itself is in the row body.
    assert claude_tool_activity_title(
        "Bash", {"command": "git status --porcelain", "description": "Show working tree status"}
    ) == "Bash · Show working tree status"
    assert claude_tool_activity_title(
        "Bash", {"command": "git status --porcelain"}
    ) == "Bash · git status --porcelain"
    assert claude_tool_activity_title(
        "Read", {"file_path": "/store/publications/README.md"}
    ) == "Read · /store/publications/README.md"
    # an MCP tool keeps its server: two services may publish `search`
    assert claude_tool_activity_title(
        "mcp__press__search", {"query": "keep agent"}
    ) == "press · search · keep agent"
    # no arguments at all still names the tool
    assert claude_tool_activity_title("TodoWrite", {}) == "TodoWrite"
    # a long command is cut at the end, not in the middle of the verb
    long_cmd = "python3 " + "x" * 300
    title = claude_tool_activity_title("Bash", {"command": long_cmd})
    assert title.startswith("Bash · python3 ") and title.endswith("…")


def test_a_long_subject_keeps_the_part_that_identifies_it():
    """LIVE: the row read `Bash · ls -a | head -20; echo "=== pkg ro…` — cut
    before anything that says what the agent did. A row is scanned, not read."""
    from kdcube_ai_app.apps.chat.sdk.solutions.claude_code.streaming import (
        TOOL_TITLE_SUBJECT_CHARS,
        claude_tool_activity_title,
    )

    # Bash carries a human description beside the command: prefer it.
    assert claude_tool_activity_title(
        "Bash",
        {"command": 'ls -a | head -20; echo "=== pkg root ==="; cat AGENTS.md',
         "description": "List the store and read the router"},
    ) == "Bash · List the store and read the router"

    # A path keeps its TAIL — the mount point identifies nothing.
    title = claude_tool_activity_title(
        "Read", {"file_path": "/bundles/kdcube/applications/kdcube-docs/procedures/"
                              "synthesis/press.linkedin@2026-08-13/publications/README.md"}
    )
    assert title.endswith("publications/README.md")
    assert len(title) <= len("Read · ") + TOOL_TITLE_SUBJECT_CHARS


def test_the_cli_is_pointed_at_the_turn_s_mcp_config(tmp_path):
    """LIVE: the agent reported — accurately — that the MCP tools its
    instructions describe were not in the session, while `.mcp.json` sat in its
    workspace with both servers in it. A project-scoped server is subject to
    approval, and a lane has nobody to approve; naming the file removes that
    step, and `--strict-mcp-config` keeps a machine-level config from adding
    servers this turn never declared."""
    from kdcube_ai_app.apps.chat.sdk.solutions.claude_code.agent import (
        ClaudeCodeAgent,
        ClaudeCodeAgentConfig,
        ClaudeCodeBinding,
    )

    workspace = tmp_path / "ws"
    workspace.mkdir()
    config = ClaudeCodeAgentConfig(
        agent_name="press",
        workspace_path=workspace,
        allowed_tools=("Read", "mcp__press"),
    )
    agent = ClaudeCodeAgent(
        config=config,
        binding=ClaudeCodeBinding(
            user_id="u", conversation_id="c", session_id="s", claude_session_id="cs",
        ),
        comm=None,
    )

    # no config file yet: nothing to point at
    assert "--mcp-config" not in agent.build_args("hello")

    (workspace / ".mcp.json").write_text('{"mcpServers": {}}', encoding="utf-8")
    args = agent.build_args("hello")
    assert "--mcp-config" in args
    assert str(workspace / ".mcp.json") in args
    assert "--strict-mcp-config" in args


def test_a_step_body_travels_as_markdown_not_inside_data():
    """LIVE: activity rows appeared with titles and nothing to expand. The chat
    renders a step's body from the composed markdown the comm contract builds;
    a body hand-placed in `data` never becomes that."""
    import asyncio

    from kdcube_ai_app.apps.chat.sdk.solutions.claude_code.agent import (
        ClaudeCodeAgent,
        ClaudeCodeAgentConfig,
        ClaudeCodeBinding,
    )

    seen: list[dict] = []

    class _Comm:
        async def step(self, **kwargs):
            seen.append(kwargs)

    agent = ClaudeCodeAgent(
        config=ClaudeCodeAgentConfig(agent_name="press", workspace_path=Path(".")),
        binding=ClaudeCodeBinding(
            user_id="u", conversation_id="c", session_id="s", claude_session_id="cs",
        ),
        comm=_Comm(),
    )
    asyncio.run(agent._emit_step(
        step="tool.1", status="completed", title="Read · x.md",
        data={"chars": 12}, markdown="**Read · x.md** — 12 chars",
    ))
    assert seen and seen[0]["markdown"] == "**Read · x.md** — 12 chars"
    assert "markdown" not in seen[0]["data"]

    # no body: the argument is omitted rather than sent empty
    seen.clear()
    asyncio.run(agent._emit_step(step="tool.2", status="running", title="t", data={}))
    assert "markdown" not in seen[0]


def test_the_workspace_is_trusted_so_its_permissions_are_read(tmp_path):
    """LIVE: the CLI logged "Ignoring 8 permissions.allow entries … this
    workspace has not been trusted" and ran with none of them — the per-turn
    allow list, MCP servers included, silently discarded. Trust is an
    interactive dialog nobody is there to click; the workspace is ours, so the
    lane states the fact."""
    import json as _json
    from kdcube_ai_app.apps.chat.sdk.solutions.claude_code.workspace import (
        ClaudeCodeWorkspaceConfig,
        prepare_claude_code_workspace,
    )

    workspace = tmp_path / "turn"
    workspace.mkdir()
    prepare_claude_code_workspace(
        workspace,
        ClaudeCodeWorkspaceConfig(
            mcp_servers={"turn_workspace": {"type": "stdio", "command": "python3"}},
            allowed_tools=["Read", "mcp__turn_workspace"],
        ),
    )
    config = _json.loads((workspace / ".claude" / ".claude.json").read_text(encoding="utf-8"))
    entry = config["projects"][str(workspace.resolve())]
    assert entry["hasTrustDialogAccepted"] is True

    # an existing config is merged, never replaced
    (workspace / ".claude" / ".claude.json").write_text(
        _json.dumps({"projects": {"/other": {"hasTrustDialogAccepted": True}}, "keep": 1}),
        encoding="utf-8",
    )
    prepare_claude_code_workspace(
        workspace,
        ClaudeCodeWorkspaceConfig(allowed_tools=["Read"]),
    )
    config = _json.loads((workspace / ".claude" / ".claude.json").read_text(encoding="utf-8"))
    assert config["keep"] == 1
    assert "/other" in config["projects"]
    assert config["projects"][str(workspace.resolve())]["hasTrustDialogAccepted"] is True
