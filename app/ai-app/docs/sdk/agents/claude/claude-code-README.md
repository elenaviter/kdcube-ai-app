---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/claude/claude-code-README.md
title: "Claude Code Agent"
summary: "Native Python SDK runner for Claude Code with deterministic user and conversation binding, workspace-scoped execution, communicator-backed streaming, framed structured-output parsing, timeout control, and correct session resume semantics."
tags: ["sdk", "agents", "claude", "claude-code", "streaming", "communicator", "workspace"]
keywords: ["ClaudeCodeAgent", "run_followup", "run_steer", "allowedTools", "session-id", "resume", "add-dir", "permission-mode", "stream-json", "ChatCommunicator", "timeout_seconds", "structured_output_prefixes", "mcp-config", "strict-mcp-config", "workspace trust", "turn_workspace", "activity rows", "per-conversation workspace", "prompt cache"]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/apps/app-with-resident-coding-agent-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/bundle-agent-integration-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/bundle-runtime-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/streaming/channeled-streamer-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/tools/tool-subsystem-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/claude/claude-code-accounting-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/claude/claude-code-workspace-bootstrap-README.md
---
# Claude Code Agent

This page documents the native Python Claude Code runner added under:

- [src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/solutions/claude_code/agent.py](../../../../src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/solutions/claude_code/agent.py)
- [src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/solutions/claude_code/runtime.py](../../../../src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/solutions/claude_code/runtime.py)
- [src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/solutions/claude_code/types.py](../../../../src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/solutions/claude_code/types.py)
- [src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/solutions/claude_code/streaming.py](../../../../src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/solutions/claude_code/streaming.py)

Use this when a bundle or SDK component wants to run `claude` directly from Python without introducing a Node bridge or bundle-local subprocess glue.

For bundle-level wiring with React, bundle-served MCP endpoints, generated
`.mcp.json`, and deployment reachability requirements, read
[Bundle Agent Integration](../../bundle/bundle-agent-integration-README.md).

## What it gives you

The SDK surface is:

- `ClaudeCodeAgent`
- `ClaudeCodeAgentConfig`
- `ClaudeCodeBinding`
- `ClaudeCodeRunResult`
- `ClaudeCodeTurnKind = "regular" | "followup" | "steer"`
- `ClaudeCodeSessionStoreConfig`
- `run_claude_code_turn(...)`

Main features:

- native Python subprocess execution of `claude`
- deterministic Claude session binding from current KDCube user + conversation + agent name
- explicit caller-supplied workspace path
- optional SDK writing of standard Claude workspace support files:
  `.mcp.json`, `.claude/settings.local.json`, `CLAUDE.md`, and native
  `.claude/skills/...` project Skills materialized from KDCube skill ids
- explicit caller-supplied allowed tools
- explicit additional writable / accessible directories via `--add-dir`
- explicit Claude permission mode such as `acceptEdits`
- incremental `chat.delta` emission through `ChatCommunicator`
- optional framed structured-output extraction from streamed assistant text
- optional per-turn timeout
- separate stderr step emission
- bounded failure diagnostics with stdout/stderr tails
- optional bounded stdout stream logging for debugging
- support for `regular`, `followup`, and `steer` turns

## Mental model

This runner is not a long-lived PTY session.

Each turn starts a fresh `claude -p` subprocess, but reuses a stable Claude
session identity so Claude Code can keep its own continuity across turns.

That means:

- first turn uses `--session-id <stable-uuid>` to create the Claude session
- continued turns use `--resume <stable-uuid>` to continue that same session
- `run_followup(...)` and `run_steer(...)` always resume
- `run_turn(..., resume_existing=True)` is available when the caller wants a
  normal prompt shape but is continuing an already existing conversation

All of these reuse the same workspace and deterministic Claude session id.

## Binding model

The runner binds itself from the current request context in:

- [src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/runtime/comm_ctx.py](../../../../src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/runtime/comm_ctx.py)

It reads:

- `request_context.user.user_id`
- `request_context.user.fingerprint` as fallback
- `request_context.routing.conversation_id`
- `request_context.routing.session_id` as fallback

The deterministic Claude session id is derived as:

```python
uuid.uuid5(
    uuid.NAMESPACE_URL,
    f"kdcube/claude-code/{user_id}/{conversation_id}/{agent_name}",
)
```

So the effective session identity is:

- current user
- current conversation
- Claude agent name

This avoids cross-user session collisions while still allowing one user to run multiple Claude sessions by using different conversations or different agent names.

Important distinction:

- `ClaudeCodeBinding.session_id` is the current KDCube request/session correlation id
- `ClaudeCodeBinding.claude_session_id` is the stable Claude resume identity

So browser session expiry or multi-device login changes do not break Claude Code session continuity. Continuity is anchored to `user_id + conversation_id + agent_name`, not to the transient KDCube session id.

### Background, cron, and service-owned bindings

`ClaudeCodeAgent.from_current_context(...)` is the right default inside a live
chat/API request. Cron jobs, background jobs, and service-owned pipelines often
do not have a meaningful end-user request context. Those callers should build
the binding explicitly and keep the same continuity boundary every run.

Minimal pattern:

```python
import uuid
from pathlib import Path

from kdcube_ai_app.apps.chat.sdk.solutions.claude_code import (
    ClaudeCodeAgent,
    ClaudeCodeAgentConfig,
    ClaudeCodeBinding,
    ClaudeCodeSessionStoreConfig,
    run_claude_code_turn,
)

conversation_id = "tenant/project/bundle/news/claude-code-session"
agent_name = "marketing-news-pipeline"
claude_session_id = str(
    uuid.uuid5(
        uuid.NAMESPACE_URL,
        f"kdcube/claude-code/news-pipeline/{conversation_id}/{agent_name}",
    )
)

agent = ClaudeCodeAgent(
    config=ClaudeCodeAgentConfig(
        agent_name=agent_name,
        workspace_path=Path("/srv/work/news-workspace"),
        command="claude",
        model="claude-sonnet-4-6",
        allowed_tools=("Read", "Grep", "Bash", "WebFetch", "WebSearch"),
        additional_directories=(Path("/srv/work/news-pipeline"),),
        env={"ANTHROPIC_API_KEY": "..."},
        permission_mode="acceptEdits",
        timeout_seconds=900,
        structured_output_prefixes=("NEWS_PIPELINE_RESULT",),
    ),
    binding=ClaudeCodeBinding(
        user_id="news-pipeline",
        conversation_id=conversation_id,
        session_id=conversation_id,
        claude_session_id=claude_session_id,
    ),
    comm=bound_comm_or_none,
)

session_store = ClaudeCodeSessionStoreConfig(
    implementation="local",  # or "git"
    local_root=Path("/srv/work/news-workspace/.claude"),
    tenant="tenant",
    project="project",
    user_id="news-pipeline",
    conversation_id=conversation_id,
    agent_name=agent_name,
    git_repo=None,
)

result = await run_claude_code_turn(
    agent=agent,
    prompt=prompt,
    kind="regular",
    resume_existing=previous_run_initialized,
    session_store=session_store,
)
```

For these service-owned flows, make the following choices explicit:

- `ClaudeCodeBinding.user_id`: use a stable service identity, not an arbitrary request user.
- `conversation_id`: use a durable pipeline/conversation id that represents the logical job stream.
- `claude_session_id`: derive it deterministically from that service identity, conversation, and agent.
- `ClaudeCodeAgentConfig.command`: pass the configured Claude binary when it is not simply `claude`.
- `structured_output_prefixes`: set every line-framed result prefix the pipeline expects to consume.
- `session_store.local_root`: use a deterministic local root for the same continuity boundary.

Do not rely on final prose parsing for machine contracts when a structured
prefix is available. The runner only populates
`ClaudeCodeRunResult.structured_events` for prefixes listed in
`structured_output_prefixes`.

## Public API

Typical usage:

```python
from pathlib import Path

from kdcube_ai_app.apps.chat.sdk.solutions.claude_code import ClaudeCodeAgent

agent = ClaudeCodeAgent.from_current_context(
    agent_name="kb-writer",
    workspace_path=Path("/workspace/docs"),
    model="claude-sonnet-4-6",
    allowed_tools=["Read", "Grep", "Bash", "WebFetch", "WebSearch"],
    additional_directories=[
        Path("/workspace/output-repo"),
        Path("/workspace/source-repo"),
    ],
    permission_mode="acceptEdits",
    timeout_seconds=900,
    structured_output_prefixes=("CLAUDE_EVENT",),
)

result = await agent.run_turn(
    "Review the connected repos and propose the wiki structure."
)
```

Follow-up:

```python
followup = await agent.run_followup(
    "Continue, but focus only on installation and deployment sections."
)
```

Steer:

```python
steer = await agent.run_steer(
    "Change direction. Stop editing source repos and only prepare the output wiki repo."
)
```

Continuing an existing conversation with a regular turn:

```python
result = await agent.run_turn(
    "Now push the prepared wiki branch.",
    resume_existing=True,
)
```

## Output diagnostics

Claude Code emits `stream-json` stdout. The runner always stores capped raw
stdout lines in `ClaudeCodeRunResult.raw_output_lines`. Failed runs also include
`failure_diagnostics` with:

- failure reason and interpretation
- timeout/exit details
- stdout/stderr tails
- final text tail
- result-event presence
- structured-output and executive-journal tails
- usage/model snapshot when available

Full stream logging is opt-in because stdout can contain private user data or
large intermediate payloads:

```python
agent = ClaudeCodeAgent.from_current_context(
    agent_name="kb-writer",
    workspace_path=Path("/workspace/docs"),
    log_stream_output=True,
    log_stream_output_max_chars=1200,
)
```

When enabled, each stdout event is written as one bounded log line with the
event type, raw size, and a tail preview.

## CLI invocation model

The runner executes Claude Code in print mode with stream-json:

```text
claude -p --verbose --output-format stream-json --include-partial-messages ...
```

Important current flags:

- `-p`
- `--verbose`
- `--output-format stream-json`
- `--include-partial-messages`
- `--mcp-config <workspace>/.mcp.json --strict-mcp-config` when that file exists
- `--model <alias|name>` when configured
- `--allowedTools ...` when configured
- `--permission-mode <mode>` when configured
- `--add-dir <path>` for each configured additional directory
- `--agent <agent_name>` **only when that agent is defined in the workspace** at `.claude/agents/<agent_name>.md`
- `--session-id <stable-uuid>` for first turn
- `--resume <stable-uuid>` for continued turns

The CLI command is configurable through `ClaudeCodeAgentConfig.command`, but defaults to `claude`.
Anything else a lane needs on the command line rides `ClaudeCodeAgentConfig.extra_args`
— that is how live control passes `--settings` (below).

### Reaching a run that is already going (live control)

A hosted lane can hand a message to a run in flight, and can stop one, without
killing the process. Both go through a `PreToolUse` hook, seeded by
`solutions/claude_code/live_control.py` into the workspace's `.kdcube-live/`:

- `allow` + a reason — the model reads what the person said and keeps working
- `deny` + a reason — the tool call does not happen, so the model answers with
  what it has

Two measurements decided this shape, against CLI 2.1.232:

- **streaming stdin does not deliver into a live turn.** With
  `--input-format stream-json --replay-user-messages`, a message written between
  two tool calls was replayed back on stdout in ~20 ms — ingested immediately —
  and the running turn never acted on it. The run finished normally and no
  further turn started.
- **the hook does.** With `allow` + `additionalContext`, the model's next line
  after a tool call reported the out-of-band message and it carried on; with
  `deny`, it stopped calling tools and answered (`turns=2` where the task needed
  eight).

The seeded files are `kdcube-live-events.json` (the buffer), `kdcube-live-hook.py`
and `kdcube-live-settings.json`, and the hook is registered for **every** tool
(`matcher: "*"`, MCP tools included) — a matcher naming a few leaves the run
reachable through the ones it forgot.

**The buffer is stamped with its turn, and the hook is invoked with that turn id
on its command line.** The workspace is per CONVERSATION, so the buffer outlives
the turn that wrote it: without the stamp, a stop written in one turn denied
every tool call of every later turn, and the agent — seeing its commands refused
repeatedly — reasonably concluded that a permission policy was blocking the
path. A stop belongs to one run.

The stamp only holds because these files sit outside the session store's
checkout; seeded inside it they were restored in matching stale pairs, which the
stamp cannot detect (workspace bootstrap doc, "Live control files in the
workspace").

Two limits, both real:

- a run inside one long tool call reaches no hook until that call returns, and a
  run that has stopped calling tools reaches none at all
- what no hook delivered is not claimed as delivered
  (`delivered_message_ids`), so the caller can leave it for the next turn

### Resume after a kill

Measured, because the timeout path already kills mid-run: a run **SIGKILLed** two
tool calls into a loop resumed on the same `--session-id` with exit 0 and its
memory from before the kill intact — transcript readable, unfinished assistant
turn harmless, work not redone. `--fork-session` is available but not needed.

### When `--agent` is passed

`agent_name` is always a stable label (it keys the session/journal/accounting
paths), but it is passed to the CLI as `--agent` **only when a matching agent
definition exists** in the workspace: `.claude/agents/<agent_name>.md`. The CLI
runs with `cwd` set to the workspace and validates `--agent` against its
registered/defined agents, so passing a name with no definition fails the run
with `--agent '<name>' not found`. When no definition is present, the runner
omits `--agent` and the default agent runs with the seeded `CLAUDE.md`
instructions — the same effective behavior, without the failure.

A bundle that wants a genuinely custom agent seeds its definition into the
workspace `.claude/` (for example through the workspace's `.claude` git seed —
see [workspace bootstrap](claude-code-workspace-bootstrap-README.md)); a bundle
that uses `agent_name` purely as a label seeds nothing and runs the default
agent.

### What it takes for the CLI to actually have its MCP tools

Writing `.mcp.json` is necessary and not sufficient. A hosted lane that looks
correct from the outside can still start a session with zero MCP tools, and the
agent then reports — accurately — that the tools its instructions describe are
not in its session. Every one of these is required:

- **The config is named on the command line.** A project `.mcp.json` the CLI
  discovers by itself is *approval-scoped*, and a lane has nobody to approve it.
  The runner passes `--mcp-config <workspace>/.mcp.json --strict-mcp-config`
  whenever the file exists; strict mode also keeps a user- or machine-level
  config from adding servers this turn never declared.
- **The workspace is trusted.** Until a project is trusted the CLI ignores
  *all* of its `permissions.allow` entries — the whole list, MCP tools
  included — and says so in its own log: `Ignoring N permissions.allow entries
  … this workspace has not been trusted`. `prepare_claude_code_workspace(...)`
  records the trust as a fact (`.claude/.claude.json` →
  `projects["<abs workspace>"].hasTrustDialogAccepted = true`), because the
  workspace is the platform's own directory.
- **No server carries a reserved name.** `workspace` is reserved by the CLI; a
  server declared under it is refused by name. The SDK's local server is
  `turn_workspace` (`WORKSPACE_MCP_SERVER_ID`).
- **`--allowedTools` is a permission list, not an availability filter.**
  Omitting a tool there does not remove it from the session, and listing one
  does not add a server that was never configured.
- **A remote KDCube surface answers the bearer.** An app's own `@mcp` surface
  must be declared so a delegated bearer authenticates on it (`route: "public"`
  plus a managed `auth_config` with `selected_tool_grants: true`) — and a public
  route changes the served path, so the connection URL and the Connection Hub
  catalog's resource pattern must follow it. Details and probes:
  [Bundle Agent Integration §8](../../bundle/bundle-agent-integration-README.md#8-mcp-url-reachability).

When tools are missing, read the CLI's own `system` init event first: it lists
the session's servers, their connection status, and the ignored-permissions
warning. It names the cause; guessing from the app side does not.

### The turn's own workspace server

A CLI runtime has exactly one door for tools — MCP — so the platform's
pull-by-ref primitive reaches it as a **local stdio server** rather than an
in-process binding:
`sdk/solutions/foreign_runtime/workspace_mcp.py`, configured by
`workspace_tools.workspace_mcp_server(...)`. It offers `pull(refs)` and
`pulled()` over the same `pull_refs_into_dir` the native agent uses, takes its
identity from the child's environment (never from a tool argument), and returns
each pulled file as a local path plus a time-limited download link.

It is deliberately **not** a named service: namespaces come and go with an
administrator's inventory, a user's pick, or a lapsed grant, and none of that
may take away an agent's ability to open a file its own conversation carries.
Nothing is materialized until the agent asks.

## Streaming behavior

Claude Code stream-json output is not token-by-token.

In practice the CLI often emits cumulative partial message snapshots. The SDK runner converts those snapshots into incremental suffix chunks before calling `self.comm.delta(...)`.

That logic lives in:

- [src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/solutions/claude_code/streaming.py](../../../../src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/solutions/claude_code/streaming.py)

Communicator behavior:

- `chat.step` with `status="started"` when the turn begins
- `chat.delta` for each incremental Claude chunk
- `chat.step` on stderr lines when `emit_stderr_steps=True`
- `chat.step` with `status="completed"` or `status="error"` at the end

The runner does not call `chat.complete` itself. That remains the responsibility of the surrounding bundle or workflow turn handling.

### Tool activity is not answer text

Claude Code reports every tool result back into its own conversation as a
`user` event, and a tool call as a `tool_use` block on an assistant event. A
generic text extractor takes both, so a file the agent read arrives in chat **as
the agent's answer**, line numbers and all.

The contract the runner enforces:

- `tool_result`, `tool_use`, `thinking`, and `redacted_thinking` blocks, and
  whole `user` / `system` events, are **never** answer text
  (`_NON_ANSWER_BLOCK_TYPES`, `_NON_ANSWER_EVENT_TYPES` in `streaming.py`);
- they are not swallowed either. Each call and each result becomes an
  **activity row** — a `chat.step` carrying the title, the arguments, a bounded
  output head (`TOOL_RESULT_PREVIEW_CHARS`), the output size, and an error flag
  — plus one compact line in the thinking lane;
- **one step key per call** (`tool.1`, `tool.2`, …). A shared key makes the UI
  show a single row rewriting itself: three tools, one flickering line;
- **the row's body travels as the `markdown` argument** of the step, because
  the comm contract composes it into the block shape the chat renders. A body
  placed in `data` produces a row with nothing to expand;
- **a row reads as a sentence** (`claude_tool_activity_title`):
  `Bash · Show working tree status`, `Read · …/publications/README.md`,
  `press · search · keep agent`. A Bash call prefers its human `description`
  over its command line; a path keeps its **tail**, not its head.

Extraction helpers: `extract_tool_uses_from_claude_event`,
`extract_tool_results_from_claude_event`, `claude_tool_activity_title`.

## Structured streamed output

Some callers need more than raw `final_text`. For that case the runner can parse
framed JSON records directly from streamed assistant text.

Configure:

- `ClaudeCodeAgentConfig.structured_output_prefixes`
- `ClaudeCodeAgentConfig.on_structured_output`
- `ClaudeCodeAgentConfig.on_text_chunk`

The intended contract is line-framed output, for example:

```text
CLAUDE_EVENT {"type":"phase","phase":"analysis","status":"started"}
CLAUDE_EVENT {"type":"warning","message":"fallback path activated"}
```

The prefix is caller-defined. The platform only enforces that parsing is
prefix-based; it does not reserve an application-specific event name.

The runner does not try to parse arbitrary JSON from normal prose. It only
parses lines beginning with one of the configured prefixes.

Parsed records are returned in `ClaudeCodeRunResult.structured_events` as:

```python
{
    "prefix": "CLAUDE_EVENT",
    "payload": {"type": "phase", "phase": "analysis", "status": "started"},
    "raw_line": 'CLAUDE_EVENT {"type":"phase","phase":"analysis","status":"started"}',
}
```

This is meant for workflows that need semantic progress while the turn is still
running, while still ending with one final result payload in `final_text`.

## Executive Journal

The runner also has a standard structured-output checkpoint channel named
`executive_journal`. It uses the reserved prefix:

```text
EXECUTIVE_JOURNAL Searched scoped emails and found 3 candidates.
EXECUTIVE_JOURNAL {"channel":"struct","candidate_count":3,"note":"Scoped search completed"}
EXECUTIVE_JOURNAL_CODE print("small recoverable snippet")
```

Claude Code callers can instruct the subprocess to emit these one-line JSON
or text checkpoints after substantial progress. The SDK captures them into
`ClaudeCodeRunResult.executive_journal` even if the Claude process later fails
or times out. Entries are intentionally loose: plain text is captured as
`channel="note"`, JSON as `channel="struct"` unless it declares a different
`channel`, and `EXECUTIVE_JOURNAL_CODE` as `channel="code"`.

Configure:

- `ClaudeCodeAgentConfig.executive_journal_prefixes`
- `ClaudeCodeAgentConfig.executive_journal_max_entries`

The default prefix is `EXECUTIVE_JOURNAL` and the default retained entry count
is 100. These entries are intended for compact recoverable progress, not full
transcripts or large artifacts.

## Result object

`ClaudeCodeRunResult` returns:

- `status`
- `session_id`
- `final_text`
- `delta_count`
- `exit_code`
- `stderr_lines`
- `raw_output_lines`
- `turn_kind`
- `agent_name`
- `provider`
- `requested_model`
- `model`
- `usage`
- `cost_usd`
- `duration_ms`
- `api_duration_ms`
- `raw_result_event`
- `error_message`
- `timed_out`
- `timeout_seconds`
- `structured_events`
- `executive_journal`

This is meant for bundle logic and diagnostics, not only UI streaming.

`requested_model` is what the caller asked Claude Code to use. `model` is what the CLI stream actually reported for the run. When aliases like `sonnet` or `opus` are used, this distinction is useful for observability and accounting.

## Model selection

`ClaudeCodeAgentConfig.model` is optional.

- if omitted or `"default"`, the runner starts Claude Code without `--model`
- if set, the runner forwards it via `claude --model <alias|name>`

This makes it possible for a bundle to persist a user-selected Claude model and reuse it across turns while still keeping the actual resolved model visible in the result object and accounting events.

## Accounting

Claude Code runs are accounted as normal `service_type=llm` usage events with:

- `provider="anthropic"`
- `metadata.runtime="claude_code"`
- resolved usage from the `stream-json` result stream

See [claude-code-accounting-README.md](claude-code-accounting-README.md).

## Workspace model

The caller must provide `workspace_path`.

By default, the low-level runner does not:

- clone repos
- isolate concurrent worktrees
- publish or push changes

That is intentional. Workspace orchestration belongs to the caller or a higher-level SDK abstraction.

If the Claude run needs access outside the main workspace root, the caller
should pass `additional_directories`. These are forwarded to Claude Code as
`--add-dir` entries.

Important distinction:

- `workspace_path` controls the subprocess working directory
- `additional_directories` controls extra paths passed to Claude through
  `--add-dir`
- neither one is a security sandbox

Plain-language boundary summary:

- `workspace_path` means "run Claude from this directory."
- `additional_directories` means "also pass these paths via `--add-dir`."
- That is workspace scoping, but not security isolation. Claude is still a
  subprocess in the same OS/container security boundary. It is not a sandbox,
  chroot, container, or per-user filesystem jail.
- Repo bootstrap/publish means hydrating/persisting Claude's own
  session/workspace files, for example via git-backed session store. That
  remains handled by the higher-level runtime, not the low-level subprocess
  runner.
- Secret injection policy means the runner should not decide which secrets are
  safe to resolve/write. The caller must pass resolved short-lived tokens or env
  values deliberately.

The caller must choose a per-user/per-conversation/per-agent workspace path when
concurrent or cross-user isolation is required.

### Make it per conversation, never per turn

A workspace path that changes between turns costs the whole prompt cache. The
working directory is part of the CLI's runtime system prompt, so a new path per
turn re-creates that prefix: the stream reports
`cache_miss_reason: system_changed` and the turn pays for it in latency (~7s to
first token, of which the model was ~2s, on a measured turn of ~13k re-created
tokens).

Use one directory per continuity boundary — `…/agent_workspaces/<conversation>`
— the same boundary as `claude_session_id` and the session-store branch. Then
the cache holds, the session store materializes once per conversation instead of
once per turn, and the CLI's transcript stays where it put it.

## Workspace support files

The SDK includes a helper for standard Claude Code workspace files:

- `ClaudeCodeWorkspaceConfig`
- `prepare_claude_code_workspace(...)`

Use it when the bundle wants the SDK to write `.mcp.json`,
`.claude/settings.local.json`, `CLAUDE.md`, and native Claude Code project
Skills before the Claude subprocess starts.

Example:

```python
from kdcube_ai_app.apps.chat.sdk.solutions.claude_code import (
    ClaudeCodeAgent,
    ClaudeCodeAgentConfig,
    ClaudeCodeWorkspaceConfig,
)

workspace_config = ClaudeCodeWorkspaceConfig(
    mcp_servers={
        "scoped_data": {
            "type": "http",
            "url": mcp_url,
            "headers": {"X-Example-MCP-Token": short_lived_token},
        }
    },
    allowed_tools=[
        "mcp__scoped_data__task_context",
        "mcp__scoped_data__list_items",
        "mcp__scoped_data__record_result",
    ],
    skill_ids=[
        "product.scoped-data-processing",
    ],
    skill_allowed_tools={
        "product.scoped-data-processing": [
            "mcp__scoped_data__task_context",
            "mcp__scoped_data__list_items",
            "mcp__scoped_data__record_result",
        ],
    },
    denied_tools=["Bash", "Read", "Edit", "Write", "WebFetch", "WebSearch"],
    instructions_markdown=(
        "# Scoped Data Processor\n\n"
        "Use only the configured scoped_data MCP tools.\n"
        "Call task_context first and record_result before the final answer.\n"
    ),
)

agent = ClaudeCodeAgent(
    config=ClaudeCodeAgentConfig(
        agent_name="scoped-data-processor",
        workspace_path=workspace_path,
        workspace_config=workspace_config,
        allowed_tools=list(workspace_config.allowed_tools),
    ),
    binding=binding,
    comm=comm,
)
```

When `workspace_config` is set, `ClaudeCodeAgent.run_turn(...)` prepares the
workspace before checking that `workspace_path` exists.

Besides those files the helper records the workspace as **trusted** in the
CLI's own config (`.claude/.claude.json`), without which the CLI discards every
`permissions.allow` entry it just wrote — see
[What it takes for the CLI to actually have its MCP tools](#what-it-takes-for-the-cli-to-actually-have-its-mcp-tools).

`skill_ids` are KDCube skill ids known to the active skills subsystem, for
example `public.pdf-press` or `product.email-analysis`. The SDK expands skill
imports, writes each resolved skill as a native Claude Code project Skill under
`<workspace_path>/.claude/skills/<skill-name>/SKILL.md`, and copies support
files next to the source KDCube `SKILL.md`.

The SDK does not infer Claude tool permissions from KDCube skill `tools.yaml`.
React tool ids and Claude MCP tool names are different surfaces. Configure MCP
servers and `allowed_tools` explicitly; use `skill_allowed_tools` only when the
generated Claude Skill should also declare skill-local Claude tool hints.

The helper does not resolve secrets. The caller must pass already-resolved
short-lived tokens, headers, or non-secret MCP URLs.

## Session-store bootstrap

Claude workspace/session continuity is now handled by a separate runtime layer,
not by the low-level runner itself.

Use:

- `run_claude_code_turn(...)`
- `ClaudeCodeSessionStoreConfig`

when the caller wants:

- a bundle-controlled local Claude root
- optional git bootstrap before a regular turn
- optional publish after the turn

That layer supports:

- `CLAUDE_CODE_SESSION_STORE_IMPLEMENTATION=local|git`
- `CLAUDE_CODE_SESSION_GIT_REPO=<repo>`

See [claude-code-workspace-bootstrap-README.md](claude-code-workspace-bootstrap-README.md).

## Allowed tools

Allowed Claude Code tools are fully caller-controlled.

Example:

```python
agent = ClaudeCodeAgent.from_current_context(
    agent_name="repo-curator",
    workspace_path=workspace_path,
    allowed_tools=["Read", "Grep", "Bash", "WebFetch", "WebSearch"],
)
```

If `allowed_tools` is empty, the runner simply omits `--allowedTools`.

## Permission mode

The runner exposes Claude Code permission mode through
`ClaudeCodeAgentConfig.permission_mode` and
`ClaudeCodeAgent.from_current_context(..., permission_mode=...)`.

Current default:

- `acceptEdits`

This is useful for managed workspaces where the caller wants Claude to edit
within the allowed workspace / `--add-dir` scope without stopping on each file
write.

## Error behavior

Current behavior:

- invalid or missing workspace path raises before subprocess execution
- subprocess start failure emits an error step and re-raises
- non-zero Claude exit code returns `ClaudeCodeRunResult(status="failed", ...)`
- per-turn timeout marks the run as failed and terminates the Claude subprocess
- stderr lines are captured separately and also included in the final error step payload
- final error step payload includes:
  - `last_stderr_line`
  - `raw_result_event`
  - `timed_out`
  - `timeout_seconds`
  - `failure_diagnostics`

`failure_diagnostics` is a compact debugging snapshot for failed runs. It
includes the failure reason (`timeout_waiting_for_process_result`,
`timeout_after_result_event`, `stream_reader_failed`, or `nonzero_exit`), a
short interpretation, stdout/stderr tails, final text tail, delta count,
structured-output counts/tails, executive-journal tail, usage snapshot, model
resolution, and whether a Claude result event was seen. It is intentionally a
diagnostic view; it does not change cache accounting or retry behavior.

The runner is designed so failures are visible both:

- in Python control flow
- in SSE / communicator diagnostics

## Current limitations

This first cut does not provide:

- PTY-backed interactive stdin sessions
- security-grade workspace isolation or sandboxing
- automatic secret injection or secret-resolution policy
- bundle UI integration

It can write standard workspace support files when `workspace_config` is
provided, but the caller still owns the policy for which MCP servers, headers,
instructions, skills, and permissions are safe to write.

Stream and watchdog behavior:

- Claude Code `stream-json` stdout/stderr is read in chunks and assembled into
  lines by the SDK, so large single-line JSON events do not trip Python's
  default `StreamReader.readline()` limit.
- While the subprocess is alive, the runner marks processor task activity
  internally. This keeps long Claude turns from being treated as idle without
  emitting fake user-facing events.
- The processor hard wall-time cap still wins over ongoing activity.

The generic runner still does not itself own repo bootstrap/publish policy. That
is handled by the higher-level Claude workspace/session-store runtime layer.

Those belong to higher-level bundle integrations.

## Tests

Focused tests live in:

- [src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/solutions/claude_code/tests/test_claude_code_agent.py](../../../../src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/solutions/claude_code/tests/test_claude_code_agent.py)

Covered cases:

- deterministic binding from current request context
- argument construction
- incremental snapshot-to-delta conversion
- stderr emission
- failure reporting
- large single-line `stream-json` events
- internal processor activity touch for long-running subprocesses
- first-turn `--session-id` vs resumed-turn `--resume`
- session reuse across `followup` and `steer`
- git-backed session bootstrap/publish through `run_claude_code_turn(...)`

## Intended Bundle Use

A bundle can use this SDK runner to:

- bind Claude Code execution to the current admin user
- keep conversation continuity across turns
- point Claude at caller-managed repo workspaces
- stream Claude output through the standard communicator path
- optionally persist Claude's own session substrate through the git-backed session store
