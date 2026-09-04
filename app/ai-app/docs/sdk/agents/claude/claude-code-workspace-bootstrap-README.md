---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/claude/claude-code-workspace-bootstrap-README.md
title: "Claude Code Workspace Management"
summary: "How KDCube manages Claude Code session continuity and binds shared pull, checkout, publication, and optional turn-summary capabilities to a hosted Claude turn."
tags: ["sdk", "agents", "claude", "claude-code", "workspace", "git", "bootstrap"]
updated_at: 2026-09-04
keywords:
  [
    "Claude Code workspace",
    "Claude session store",
    "claude_session_id",
    "git-backed Claude session",
    "bundle-controlled workspace root",
    "run_claude_code_turn",
    "bind_claude_code_turn_workspace",
    "workspace publication policy",
    "record_turn_summary",
    "Claude Code turn summary contribution",
  ]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/claude/claude-code-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/claude/claude-code-accounting-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/configuration/service-runtime-configuration-mapping-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/configuration/assembly-descriptor-README.md
  - repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/solutions/claude_code/runtime.py
  - repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/solutions/claude_code/agent.py
  - repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/solutions/claude_code/harness_workspace.py
  - repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/solutions/foreign_runtime/publication.py
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/timeline/turn-summary-contributions-README.md
---
# Claude Code Workspace Management

KDCube separates two concerns:

- `ClaudeCodeAgent` is the generic runner
- the caller or bundle owns workspace/bootstrap policy

That split matters because Claude continuity is not restored from KDCube's own
conversation JSON. Continuity comes from Claude's own local session files plus
the stable `claude_session_id`.

The current runtime support for that lives in:

- [src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/solutions/claude_code/agent.py](../../../../src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/solutions/claude_code/agent.py)
- [src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/solutions/claude_code/runtime.py](../../../../src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/solutions/claude_code/runtime.py)
- [src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/solutions/claude_code/types.py](../../../../src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/solutions/claude_code/types.py)

## Integration profiles

App authors choose the Claude Code integration that matches the product lane.
Both profiles use the same policy-neutral `ClaudeCodeAgent` and
`run_claude_code_turn(...)` SDK contract:

- **Direct bundle-owned execution.** The app supplies its own workspace,
  session lineage, allowed tools, inputs, validation, and product persistence.
  This fits off-turn pipelines and jobs whose output becomes app-domain state
  rather than a chat file. No Agent Harness turn-workspace binding is required.
  The News app is the reference: its issue-date workspace, read-only source
  repositories, resumed Claude session, accounting scope, validators, and news
  archive are all owned by the app.
- **Harness-bound conversational execution.** The app additionally calls
  `bind_claude_code_turn_workspace(...)` when Claude must resolve conversation
  or owner refs, create/reset editable turn files, explicitly deliver files
  into a conversation, or contribute a searchable turn summary. The Press app
  is the reference.

The harness binding is additive and opt-in. The generic runner does not create
a broker, expose conversation refs, or acquire file-hosting authority by
default.

Neither profile turns the Claude CLI into a per-user sandbox. A
per-conversation path separates continuity and prevents accidental collisions;
it does not stop `Bash` from reaching other paths, environment values, or
network destinations available to the processor. Use direct execution for
trusted app-owned pipelines or trusted admin/insider lanes. A hostile
multi-user lane requires a separate OS-enforced per-user process/filesystem
profile. The full boundary is owned by
[Claude Code Agent: Subprocess trust boundary](claude-code-README.md#subprocess-trust-boundary).

## Core model

Claude Code workspace management is defined by these rules:

- the bundle chooses the local root Claude should use
- the agent runner does not hardcode that root
- Claude continuity is anchored to:
  - `user_id`
  - `conversation_id`
  - `agent_name`
- KDCube may optionally bootstrap the chosen local Claude root from git before a turn
- after the turn, KDCube may publish the mutated Claude root back to git

So the authoritative continuity substrate is:

- the Claude-created files under the bundle-chosen local Claude root

not:

- KDCube conversation message history
- the final assistant transcript stored by the bundle
- accounting events

## Harness turn workspace binding

Claude's session store and the Agent Harness turn workspace are separate
contracts. The session store preserves Claude's private conversation/session
continuity. The harness workspace gives an adapter explicit access to durable
conversation files and owner objects, editable current-turn copies, and
conversation publication.

`ClaudeCodeAgent` remains a policy-neutral runner. A hosted wrapper opts into
the harness workspace once with `bind_claude_code_turn_workspace(...)`; the
binding owns the local stdio MCP server, trusted parent broker, permission rule,
server enablement, and common instruction block:

```text
accepted event
  event_ref:  conv:ev:conv_c42.turn_t9.events/canvas/context/evt_17
  object_ref: cnv:users/u7/canvases/main/objects/card_17/v000004.md
                         |
                         v
Claude Code local MCP: pull | checkout | publish | record_turn_summary | pulled
                         |
              Unix socket, per-turn token
                         |
trusted parent: request identity + ContextBrowser + hosting service
```

```python
turn_workspace = await bind_claude_code_turn_workspace(
    workspace=workspace,
    tenant=tenant,
    project=project,
    user_id=user_id,
    conversation_id=conversation_id,
    turn_id=turn_id,
    entrypoint=entrypoint,
    state=state,
    publication_policy=product_policy,
    turn_summary_enabled=declared_python_tool_enabled(
        declared_connections,
        disabled_tools,
        "record_turn_summary",
    ),
)
servers = turn_workspace.merge_mcp_servers(product_servers)
workspace_config = turn_workspace.apply_workspace_config(
    ClaudeCodeWorkspaceConfig(
        mcp_servers=servers,
        instructions_markdown=product_instructions,
    )
)
try:
    result = await run_claude_code_turn(...)
finally:
    await turn_workspace.close()
```

- `pull` materializes a `conv:fi:` ref or an authorized owner `object_ref` into
  a collision-safe readonly path; Claude uses native Read, Grep, or Bash on the
  returned workspace-relative path.
- `checkout(items=[{from,to,strategy}])` resolves the source directly into an
  editable path below `git/projects/...` or `files/...`. Repeating `replace`
  resets the target from the durable source; `overlay` is directory-only.
- Claude edits the checked-out path with its native file tools.
- `publish(paths=[...])` explicitly hosts selected current-turn `files/...`
  outputs into the conversation and returns durable `conv:fi:` refs. Claude may
  pass the same current-turn workspace-relative path it read/wrote or the stable
  `files/...` form, so publication adds no path-translation step. Before
  hosting, the shared gate enforces count, per-file size, aggregate size, and
  supported MIME families. A product callback may narrow or deny the concrete
  request under trusted runtime identity, but cannot widen the shared ceiling.
- `record_turn_summary(summary, refs, phrases, entities)` stages one
  replaceable semantic contribution for the current turn. The trusted
  successful-turn recorder persists the final draft as a searchable
  `conv.working.summary` block; it never reads or exports Claude's private
  session summary. The wrapper enables this operation only when its ordinary
  `kind: python` tool declaration survives the admin ceiling and the user's
  per-conversation narrowing.

The matching descriptor inventory entry is ordinary agent configuration:

```yaml
tools:
  - name: turn_context
    kind: python
    alias: turn_context
    allowed: [record_turn_summary]
```

Omit the entry to keep the capability outside the app's ceiling. A user may
switch it off for one conversation like any other declared Python tool. The
common capability instruction joins only when the operation is actually
available to Claude.

The MCP child receives operation arguments and bound non-secret identity, not
provider credentials or a hosting service. Owner resolution, publication, and
summary staging cross the authenticated local broker to the trusted parent.
The Press Claude Code wrapper is the worked consumer of the reusable binding. A new wrapper
chooses its workspace, session lineage, product instructions, and publication
approval policy; it does not reimplement resolver, broker, hosting, or Claude
configuration glue. This mechanism does not by itself change the process and
filesystem isolation profile chosen for Claude Code, and it does not govern
direct filesystem or network actions performed through `Bash`.

## Local vs git session store

Claude session storage supports two implementations:

- `local`
- `git`

The standard `assembly.yaml` fields are the authoring contract:

```yaml
storage:
  claude_code_session:
    type: local   # local | git
    repo: ""      # local path, HTTPS URL, or SSH URL when type is git
```

A direct SDK host points `get_settings()` at that descriptor set. The runnable
contract is [Run the Agent Harness from Python](../../../recipes/quickstart/run-agent-harness-from-python-README.md).

Meaning:

- `local`
  - continuity depends on local disk persistence
  - no bootstrap or publish is performed

- `git`
  - each Claude conversation/agent gets its own remote branch
  - the local Claude root is bootstrapped from that branch before a regular turn
  - changes are published back after the regular turn

## Bundle-controlled local root

The runner remains generic:

- `ClaudeCodeAgent.from_current_context(..., workspace_path=...)`
- `ClaudeCodeAgentConfig.workspace_path`

So the SDK does not declare one global Claude root such as `/var/lib/claude/...`.

Instead:

- the bundle decides the local path
- the runtime bootstrap layer hydrates that exact path
- the publish layer persists that same path

Example valid choices:

- `<workspace_root>/.claude`
- `<workspace_root>/runtime/claude`
- a whole caller-owned Claude workspace root

The important rule is determinism: the bundle must consistently point Claude at
the same logical local root for the same continuity boundary.

### One workspace per conversation, not per turn

The same determinism rule applies to `workspace_path`, and it is not only about
continuity — it is about cost. The working directory is part of the CLI's
runtime system prompt, so a workspace created per turn re-creates that prefix
every turn (`cache_miss_reason: system_changed` in the stream) and the user
waits through it before the first token.

Use one directory per conversation — `<bundle-storage-root>/agent_workspaces/
<conversation_id>` — matching the boundary `claude_session_id` and the
session-store branch already use. Consequences worth knowing:

- the session store bootstraps **once per conversation** instead of once per
  turn, so a turn that resumes pays no materialization;
- the CLI's transcript stays under the same `projects/<cwd-slug>` key across
  turns. The cwd-retarget on bootstrap then only serves what it was meant for:
  lineage restored on a **different node**, whose recorded cwd differs;
- workspace files (`CLAUDE.md`, `.mcp.json`, settings) are refreshed in place
  per turn, which is what `refresh_support_files` is for.

### Trust is part of preparing the workspace

Claude Code honors a project's `permissions.allow` only after the project has
been trusted through an interactive dialog. A hosted lane has nobody to click
it, so the CLI logs `Ignoring N permissions.allow entries … this workspace has
not been trusted` and runs with **none** of them — including MCP servers.

`prepare_claude_code_workspace(...)` therefore writes the trust record itself:

```text
<CLAUDE_CONFIG_DIR>/.claude.json
  projects:
    "<absolute workspace path>":
      hasTrustDialogAccepted: true
```

The workspace is the platform's own directory — created by it, configured by
it, living in bundle storage — so this states a fact rather than delegating a
security decision to the agent. `CLAUDE_CONFIG_DIR` is set to the session
store's `local_root` in git mode (see below), which is what makes the trust
record travel with the session lineage.

For cron/background pipelines this root should be tied to the service-owned
identity and logical conversation, for example:

```text
<bundle-storage-root>/_news/claude-code-session
```

The same service-owned identity should also be used in
`ClaudeCodeBinding.user_id` and `ClaudeCodeSessionStoreConfig.user_id`. This is
what keeps a scheduled pipeline from accidentally sharing Claude session state
with an interactive user conversation.

## Branch identity

The git-backed session store uses one branch per:

- tenant
- project
- user
- conversation
- agent name

Shape:

```text
refs/heads/kdcube/claude/<tenant>/<project>/<user_id>/<conversation_id>/<agent_name>
```

This matches the same continuity boundary used for `claude_session_id`.

## Live control files in the workspace

A lane that wants to reach its run mid-flight seeds three files into
`.kdcube-live/` — a directory of its own beside `.claude/`, never inside it (see
`solutions/claude_code/live_control.py`):

| File | What it is |
| --- | --- |
| `kdcube-live-events.json` | the buffer: this turn's id, a stop flag, and what the person said |
| `kdcube-live-hook.py` | the `PreToolUse` hook, self-contained (the CLI runs it as a bare subprocess with no path back to the SDK) |
| `kdcube-live-settings.json` | registers the hook for every tool; passed as `--settings` through `extra_args` |

**These are per TURN inside a per CONVERSATION directory**, which is the trap:
the buffer outlives the turn that wrote it. Each turn reseeds it, and the seed
stamps the turn id — which the settings also put on the hook's command line, so
a buffer another turn wrote is ignored even before the reseed. A turn that
returns early (a refusal, a connect-required answer) never reaches the seed, so
the stamp is what makes the leftover harmless rather than the reset.

**Why they are not in `.claude/`.** That directory is the session store's
checkout (above): every turn it is emptied except for `.git` and reset to the
previous turn's snapshot. Files seeded there before the run are deleted a moment
later and REPLACED by the previous turn's copies — buffer and settings together,
so the stale stop arrives with the stale turn id that matches it and the stamp
above cannot tell. The symptom was total: after one stop, every later turn had
every tool call refused, while the first turn of a conversation always worked
because there was nothing to restore yet.

A store lineage that already carries these files sheds them at the next
bootstrap, so a conversation from before this heals on its next turn.

The buffer does not accumulate: it holds one turn's messages and is overwritten
whole. It is a few dozen bytes, and workspace cleanup takes it with everything
else.

## Turn lifecycle

The high-level runtime entry is:

```python
result = await run_claude_code_turn(
    agent=agent,
    prompt=prompt,
    kind="regular",
    resume_existing=False,
    session_store=ClaudeCodeSessionStoreConfig(...),
    refresh_support_files=refresh_fn,
)
```

Behavior:

### Regular turn

If `storage.claude_code_session.type` is `git`:

1. bootstrap the local Claude root from the conversation branch
2. optionally refresh bundle-owned support files inside that root
3. run Claude
4. publish the mutated Claude root back to the same branch

Bootstrap is rerun-safe:

- the local Claude session checkout can already exist
- the workspace branch can already be checked out there
- bootstrap refreshes that dedicated local checkout from the stored lineage branch

If Claude still fails with a stale local-session error such as "Session ID ...
is already in use", the runtime resets that dedicated local checkout and
retries the turn once in resume mode.

### Followup / steer

Current default behavior:

- no git bootstrap
- no git publish
- the local root is reused as-is inside the current live environment

This keeps followup/steer cheap for the active runtime. If a future product flow
needs cold-start followup/steer on another node, the caller can widen the
bootstrap/publish turn-kind policy.

## What goes into the git branch

Only the Claude continuity substrate should be published there.

Good contents:

- Claude session files needed by `--resume`
- minimal KDCube companion files inside that same root, if the bundle requires them

Do not put these there:

- the full product conversation JSON
- accounting events
- unrelated bundle storage
- general project output artifacts

The purpose of this branch is:

- preserve Claude continuity

not:

- replace KDCube conversation storage

## Isolation requirements

The Claude session git store follows the same security principle as the ReAct Agent's
git-backed workspace storage:

- local runtime should only see the assigned conversation branch
- it must not expose a broad shared repo view with other users' branches

So the local bootstrapped root visible to the Claude run should correspond only
to its own:

- tenant
- project
- user
- conversation
- agent

## Assembly descriptor support

Claude session storage is configured in `assembly.yaml`:

```yaml
storage:
  claude_code_session:
    type: local   # local | git
    repo: ""      # used only when type=git
```

HTTPS credentials use `services.git.http_token` and
`services.git.http_user` in `secrets.yaml`. SSH transport uses the
`services.git.git_ssh_key_path`, `git_ssh_known_hosts`, and
`git_ssh_strict_host_key_checking` fields in `assembly.yaml`. The shared Git
helper resolves this standard descriptor-backed configuration for bootstrap
and publish.

This keeps Claude session-store policy at the same platform configuration
layer as the ReAct agent's Git-backed workspace settings.

## Relationship to the core Claude runner

The runner and the workspace/bootstrap layer are intentionally separate.

`ClaudeCodeAgent` is responsible for:

- building Claude CLI args
- binding deterministic `claude_session_id`
- running the subprocess
- streaming deltas and steps
- optionally extracting framed structured JSON events from streamed assistant text
- enforcing optional per-turn timeout
- returning structured usage/model/cost/failure results

The runtime/bootstrap layer is responsible for:

- local vs git session-store policy
- branch naming
- bootstrap before the turn
- publish after the turn
- self-healing refresh of the dedicated local Claude session checkout

That separation keeps bundles flexible while still giving the platform a
standard continuity mechanism.

## Custom agent definitions

The runner passes `--agent <agent_name>` only when the workspace contains a
matching definition at `.claude/agents/<agent_name>.md`. Seed that file the same
way as other workspace `.claude` content — for example through the session-store
git seed — to run a genuinely custom agent. With no definition the runner omits
`--agent` and the default agent runs with the seeded `CLAUDE.md`, so an
`agent_name` used purely as a continuity/accounting label needs nothing seeded.
See [claude-code-README.md](claude-code-README.md) for the runner-side rule.

## Related docs

- session identity and runner behavior:
  - [claude-code-README.md](claude-code-README.md)
- Claude accounting:
  - [claude-code-accounting-README.md](claude-code-accounting-README.md)
- service env reference:
  - [docs/configuration/service-runtime-configuration-mapping-README.md](../../../configuration/service-runtime-configuration-mapping-README.md)
- assembly schema:
  - [docs/configuration/assembly-descriptor-README.md](../../../configuration/assembly-descriptor-README.md)
