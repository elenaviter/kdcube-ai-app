---
id: repo:kdcube-ai-app/app/ai-app/docs/recipes/apps/app-with-resident-coding-agent-README.md
title: "Build An App With A Resident Coding Agent"
summary: "Builder recipe for hosting a CLI coding agent (Claude Code) inside a KDCube app so it works a git-backed content store the same way an engineer does on a laptop: per-conversation workspace, session store on a git branch, the app's own MCP surface reached through Connection Hub grants and dialed locally, a local stdio workspace server for pull-by-ref, tool activity as chat steps, and the machine-local toolchain the agent maintains itself."
status: active
tags: ["recipes", "app", "claude-code", "coding-agent", "mcp", "git", "workspace"]
updated_at: 2026-08-14
keywords: ["resident coding agent", "claude code in an app", "per-conversation workspace", "turn_workspace pull", "self_hosted mcp", "delegable catalog", "activity rows", "git session store", "AGENTS.md one rulebook", "toolchain cache"]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/apps/app-with-agents-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/claude/claude-code-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/claude/claude-code-workspace-bootstrap-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/bundle-agent-integration-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/conversation/hosted-agent-conversation-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/create-delegated-automation-access-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/build/how-to-avoid-common-bundle-integration-failures-README.md
---

# Build An App With A Resident Coding Agent

Use this recipe when the work your app governs is **files in a repository** —
content, configuration, datasets, templates — and you want the people who own
that work to talk to an agent that edits it, instead of only clicking a form.

The agent is a **coding agent** (this recipe uses the Claude Code CLI): it
reads, greps, edits, runs commands, and commits. The app makes it a citizen of
the platform: identity, consent, accounting, a conversation that survives a
reload, and a desk beside the chat.

The outcome is one store with two doors:

```text
                    ONE GIT-BACKED STORE (the files that matter)
                    /                                        \
        laptop door                                        app door
   an engineer + their own CLI agent            a colleague in the browser
   working the checkout directly                talking to the RESIDENT agent
        \                                        /
         `----------- same files, same runbook, same git ----------'
```

Everything below is what it takes for the right-hand door to behave like the
left-hand one.

KDCube still uses **bundle** in literal identifiers (`bundles.yaml`,
`bundle_id`, `@bundle_entrypoint`). In prose, app and bundle mean the same
deployable unit.

## What you will wire

```text
   BROWSER                          RUNTIME (proc)                     OUTSIDE
   ───────                          ──────────────                     ───────
  chat  ──prompt+refs──►  turn lane ──► execute_core
   ▲                                      │
   │  activity rows                       ├─► workspace prepared (per CONVERSATION)
   │  answer deltas                       │      .mcp.json · settings · CLAUDE.md · trust
   │                                      │
  desk ◄─ops (@api)──►  services  ◄──────┤
   │                       │              ├─► claude CLI subprocess
   │                       ▼              │      │
   │                  GIT STORE ◄─────────┘      ├── Read/Edit/Bash on the store
   │                  (working tree)             ├── mcp__<app>__*  (app's own surface)
   │                                             └── mcp__turn_workspace__pull
   │                                                      │
   └───────── conversation record ◄── recording ──────────┘
                                                    session transcript ──► git branch
```

Six seams, in the order they bite:

| # | Seam | Gets wrong as |
| --- | --- | --- |
| 1 | the store the agent edits | a venv in the working tree; edits lost on re-materialization |
| 2 | the workspace + session | a new workspace per turn; prompt cache re-created every turn |
| 3 | instructions | two rulebooks that drift |
| 4 | the app's own MCP surface | 401/403/404 nobody can read; response lost in transit |
| 5 | the turn's objects | pre-materialized bytes nobody opens, or refs nothing can fetch |
| 6 | the chat surface | tool output published as the agent's answer |

## OP 10 · Decide where the files live, and who may write them

A resident coding agent needs a **working tree on a filesystem**. Two facts
decide the layout:

- the app re-materializes its package from source on refresh — anything the
  agent created inside the package tree is thrown away;
- the store is what the review desk commits from — anything noisy in it (build
  outputs, virtualenvs, caches) shows up in every `git status` an operator reads.

```text
<bundle storage root>/
  store/                        ← the git checkout: content the agent edits
    <content tree>/…            ← files, per your domain
  agent_workspaces/
    <conversation_id>/          ← ONE per conversation (see OP 20)
      .claude/                  ← CLI config dir + session transcript
      .mcp.json                 ← generated per turn
      CLAUDE.md                 ← generated per turn
      _pulled/                  ← objects the agent pulled this turn

<machine-local cache, NOT bundle storage>/
  <toolchain>-<sha256(requirements.txt)[:16]>/    ← the operator venv (OP 60)
```

Rules that come out of that:

- **the store is a checkout, not a mirror.** The agent's `git status`, the
  desk's "uncommitted changes" strip, and a colleague's `git pull` must all be
  reading one tree.
- **nothing generated goes in it.** Workspaces, venvs and caches live beside it.
- **writes are the app's operations, not raw file writes from the browser.**
  The desk calls guarded `@api` operations; the agent uses its own file tools
  inside the tree it was given. Both land in the same working tree, so the same
  commit lane publishes them.

## OP 20 · One workspace per conversation, and a session store in git

The CLI keeps its own continuity: a deterministic session id plus the transcript
files it writes. KDCube makes that durable across workers with the git-backed
session store — one branch per continuity boundary.

```text
   workspace path         = <storage>/agent_workspaces/<conversation_id>
   claude_session_id      = uuid5(user + conversation + agent_name)
   session store branch   = kdcube/claude/<tenant>/<project>/<user>/<conversation>/<agent>
   CLAUDE_CONFIG_DIR      = <workspace>/.claude        (git mode sets this for you)
```

```python
result = await run_claude_code_turn(
    agent=agent, prompt=prompt, kind="regular",
    resume_existing=previous_turn_ran,
    session_store=ClaudeCodeSessionStoreConfig(
        implementation="git",                 # or "local" for a single-node dev box
        local_root=workspace / ".claude",
        tenant=tenant, project=project,
        user_id=user_id, conversation_id=conversation_id,
        agent_name=agent_name, git_repo=repo_url,
    ),
)
```

**Per conversation, never per turn.** The working directory is part of the
runtime's system prompt. A workspace path that changes between turns re-creates
that prefix every turn — the stream says `cache_miss_reason: system_changed` —
and the user waits through a rebuild before the first token, every time.

```text
   per-turn workspace                    per-conversation workspace
   ──────────────────                    ──────────────────────────
   turn 1  cwd=/…/turn_a  ▉▉▉▉▉ 13k      turn 1  cwd=/…/conv_x  ▉▉▉▉▉ 13k
   turn 2  cwd=/…/turn_b  ▉▉▉▉▉ 13k      turn 2  cwd=/…/conv_x  ·  cached
   turn 3  cwd=/…/turn_c  ▉▉▉▉▉ 13k      turn 3  cwd=/…/conv_x  ·  cached
           ↑ system prompt re-created            ↑ prefix holds
```

Read: [Claude Code Workspace Management](../../sdk/agents/claude/claude-code-workspace-bootstrap-README.md).

## OP 30 · One rulebook

If the store's package ships a runbook — `AGENTS.md`, `README.md`, procedures —
the resident agent follows **that**, the same document an engineer on a laptop
follows. A second, lane-only instruction set drifts from the first within a
week and doubles every future edit.

```text
   package/AGENTS.md  ────────────────┬───────────────►  laptop agent
     roles, procedures, file layout   │
                                      └──►  CLAUDE.md (generated per turn)
                                              "You are the resident agent.
                                               Follow AGENTS.md at <path>.
                                               You are <identity>. Your
                                               surfaces are <…>."
```

The lane's own instructions carry only what is true of *this* agent: who it is,
what it may publish, which conversation it is in, and — generated fresh each
turn — the vocabulary of whatever capabilities survived the user's pick.

Two consequences people miss:

- **the package must travel with the store** in the agent's working
  directories, or its links cannot be opened and the runbook is decoration;
- **the commands the runbook demands must be permitted.** A lane has nobody to
  answer a permission prompt, so a runbook that says "run `git status`" needs
  `Bash` allowed. Keep the runtime's toolset a deployment choice
  (`agent.tools.allow` / `agent.tools.deny`, empty by default) rather than a
  constant in code: `--allowedTools` grants permission, it does not remove what
  the CLI ships.

## OP 40 · The app's own MCP surface, and the grant that opens it

A coding agent can read the store with file tools. What it cannot do with file
tools is ask the app a *question* — search, resolve, validate, commit through
the guarded lane. That is the app's own MCP surface, and the agent reaches it as
the signed-in user.

Four declarations have to agree. Miss one and the failure names something else:

```text
 1. THE SURFACE                @mcp(alias="ops", route="public",
    (app entrypoint)                transport="streamable-http",
                                    auth_config="surfaces.as_provider.mcp.ops.auth")
                                          │
 2. THE AUTH BLOCK             surfaces.as_provider.mcp.ops.auth:
    (app descriptor)             mode: managed
                                 authority_id: delegated_client
                                 selected_tool_grants: true
                                          │
 3. THE CONNECTION             surfaces.as_consumer.agents.<agent>.tools:
    (app descriptor)             - kind: mcp   server_id: ops   delegated: true
                                   url: https://<public host>/…/public/mcp/ops
                                   resource: "*/…/public/mcp/ops*"
                                   self_hosted: true
                                   scopes: [<app>:read, <app>:write, …]
                                   allowed: [<tool>, <tool>, …]
                                          │
 4. THE CATALOG                connection-hub@1-0 →
    (deployment descriptor)      connections.delegated_credentials.oauth:
                                   resources:    <resource> → tool → claim
                                   capabilities: <claim> → who may delegate it
```

Notes that cost a day each if you learn them from a stack trace:

- **"public" names the ROUTE, not the access.** No token → 404; a bad or
  ungranted token → 403; the managed guard checks the grant **per tool**. On the
  operations route the same surface answers 401 to a bearer, and a multi-tool
  surface reached through the REST path answers `403 ambiguous operation
  catalog`.
- **a public route changes the served path** to `…/<bundle>/public/mcp/<alias>`;
  the connection URL and the catalog's resource pattern must follow it.
- **the catalog is the deployment's decision.** An app *offers* endpoints; the
  Connection Hub catalog is what the deployment *allows* — always a subset. An
  endpoint absent from it is not grantable, and the hub answers
  `delegated_access_grants_not_delegable`. Both halves are needed: the endpoint
  under `resources`, and the claim under `capabilities` with who may delegate it.
- **name claims after the app and the consequence**, not after the transport:
  `<app>:read` / `:write` / `:delete` / `:commit`, so the consent card reads as
  a decision about outcomes. See
  [Create Delegated Automation Access](../connections/create-delegated-automation-access-README.md).
- **enumerate `allowed` tools** rather than `*`: the picker gets per-tool rows,
  and a partial opt-out can be expressed in the CLI's permission grammar, which
  has no wildcard between `mcp__ops` and `mcp__ops__search`.
- **`self_hosted: true`** when this deployment serves the surface: the declared
  public URL stays (grants and catalog patterns are written against it) and the
  call is dialed on the runtime's own loopback instead of leaving the machine and
  coming back through a tunnel or load balancer.

### What the CLI needs before any of that matters

Writing `.mcp.json` is necessary and not sufficient. Every one of these is
required, and each fails silently:

```text
   ✔ the config is NAMED on the command line     --mcp-config <ws>/.mcp.json --strict-mcp-config
       (a discovered project config is approval-scoped; a lane has nobody to approve)
   ✔ the workspace is TRUSTED                    .claude/.claude.json →
       (untrusted ⇒ every permissions.allow      projects["<abs ws>"].hasTrustDialogAccepted = true
        entry is ignored, MCP included)
   ✔ no server uses a RESERVED name              `workspace` is taken; the SDK's own
                                                  local server is `turn_workspace`
   ✔ the surface answers JSON, not a stream      the platform builds bundle MCP sub-apps
       (a stream is a shape every hop can          with json_response=True — a hop that
        break; the client then reports the         delivers every byte and closes the
        response as LOST while the server          stream badly costs you the response
        logged a clean 200)
```

When tools are missing, read the CLI's own `system` init event first: it lists
the session's servers, their connection status, and any
`Ignoring N permissions.allow entries` warning. It names the cause; the app side
cannot see it.

## OP 50 · Give the turn's objects a pull, not a copy

A message carries more than text: uploaded files, pinned objects, refs from
other apps. None of it should be written into the workspace before the agent
asks — most turns never open it, and a binary copied per turn is paid for per
turn.

The platform's pull-by-ref primitive reaches a CLI runtime as a **local stdio
MCP server** — the same contract an in-process runtime binds as a function:

```text
   message ──► turn events ──► the prompt says:  "attached: conv:fi:…/brief.pdf"
                                          │
                        agent decides it needs it
                                          │
                                          ▼
        mcp__turn_workspace__pull(refs=["conv:fi:…/brief.pdf"])
                                          │
             ┌────────────────────────────┴─────────────────────────┐
             │  resolves through the platform's own resolver         │
             │  writes bytes into <workspace>/_pulled/               │
             │  answers: local path  +  time-limited download link   │
             └───────────────────────────────────────────────────────┘
                                          │
                    agent opens it with its ordinary file tools
```

```python
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime import (
    WORKSPACE_MCP_SERVER_ID, workspace_mcp_server,
)

servers = claude_code_mcp_servers(server_map)          # the app's own + others
servers[WORKSPACE_MCP_SERVER_ID] = workspace_mcp_server(
    workspace=workspace_path, tenant=tenant, project=project,
    user_id=user_id, conversation_id=conversation_id,
)
```

Two boundaries worth stating out loud:

- **identity travels in the child's environment**, never in a tool argument, so
  an agent cannot pull as somebody else by naming them;
- **it is not a service capability.** Namespaces come and go with an
  administrator's inventory, a user's pick, or a lapsed grant. None of that may
  take away an agent's ability to open a file its own conversation carries — an
  agent with every capability switched off still reads, edits, and answers.

## OP 60 · The toolchain: the app names the path, the agent keeps it

If the store's procedures run scripts, they need an interpreter with
dependencies. Three placements, two of them wrong:

```text
   ✗ inside the package tree      the desk commits from it; re-materialization deletes it
   ✗ on shared storage            thousands of small files over a network filesystem;
                                  two hosts installing at once corrupt each other
   ✓ machine-local disk           keyed by sha256(requirements.txt)[:16], built in a
                                  staging directory and moved into place, so a
                                  half-finished install is never taken for an interpreter
```

And the ownership rule: **the agent maintains it, on demand, in the
conversation.** The app tells the agent the path and whether it exists; the
agent reads `requirements.txt`, inspects the venv, and builds or refreshes it
with its own shell. An app that rebuilds in the background pays on turns that
never touch the tooling and hides failures in a log nobody reads.

The turn text the app generates says one of two things:

```text
   installed at <path> — check it still matches requirements.txt before a long job
   not installed on this machine yet. Build it at <path> …  never make a venv
   inside the package
```

## OP 70 · The chat surface: workings are rows, the answer is the answer

A CLI runtime reports every tool result back into its own conversation as a
`user` event. A generic text extractor takes it, and a file the agent read
arrives in chat **as the agent's answer**, line numbers and all.

```text
   CLI stdout events                     what the chat shows
   ─────────────────                     ───────────────────
   assistant · tool_use    ────────────►  ▸ step  tool.1  "Bash · Show working tree status"   [running]
   tool_progress heartbeat ────────────►  ▸ step  tool.1  "… still running, 90s"
   user · tool_result      ────────────►  ▸ step  tool.1  ✓ 1,432 chars + output head          [completed]
   assistant · text        ────────────►  the answer (deltas)
   result                  ────────────►  usage, cost, timing
```

Rules:

- `tool_result`, `tool_use`, `thinking` blocks and whole `user` / `system`
  events are **never** answer text;
- **one step key per call** (`tool.1`, `tool.2`, …) — a shared key makes the UI
  show a single row rewriting itself;
- **the row's body travels as the step's `markdown` argument**, not in `data`,
  or the row has nothing to expand;
- **a row reads as a sentence**: prefer a command's human `description`; keep a
  path's tail, not its head;
- **surface the waiting**: the CLI heartbeats a pending tool call — put the
  elapsed time on the row, and log it. A call that will never return looks
  exactly like a slow one until you do.

## OP 80 · Do not lose the platform's turn

An app hosting an agent almost always overrides a lifecycle hook — to bring the
store current, to prepare a workspace. **Call the base first:**

```python
async def pre_run_hook(self, *, state, econ_ctx: dict | None = None):
    await super().pre_run_hook(state=state, econ_ctx=econ_ctx or {})   # recording
    await self._ensure_store(reason="pre_run_hook")                    # then your work
```

The base `pre_run_hook` starts the turn's event recording, and that recording is
what a reopened conversation is rebuilt from. The failure is silent: cost and
elapsed time appear live and vanish on reload. `econ_ctx` is required by the
economics base — a `super()` call that omits it kills every turn before the
agent starts.

Also part of "the turn is not lost":

- **context objects are recorded as their own events**, sharing the message's
  `batch_id`, so a reopened conversation shows a live chip instead of prose
  about one;
- **the conversation title needs a role that resolves to a model.** Declare the
  role in `role_models`, or every conversation lists as "Untitled" and nothing
  raises;
- the seam may leave the title on `state` or return it on the result — the
  recorder reads both.

Contract: [The Conversation For Any Agent](../../sdk/solutions/conversation/hosted-agent-conversation-README.md).

## OP 90 · The desk beside the chat

A coding agent is good at changing files and poor at showing you thirty of
them. Pair the chat with a small app UI over the same store:

```text
   ┌──────────────── app scene ─────────────────┐
   │  desk (widget)            │  chat          │
   │  ── file list ────────────┤                │
   │  ▣ asset.png   NEW  222KB │  ▸ Bash · git  │
   │  ¶ notes.md   EDITED 1.5KB│  ▸ Read · …/n… │
   │  ⚙ meta.yaml         1.8KB│  the answer…   │
   │  [↻ Reload] [⇪ Upload]    │                │
   │  ── uncommitted (3) ──────┤                │
   │  [ note ] [Commit & Push] │                │
   └───────────────────────────┴────────────────┘
        ▲                              ▲
        └── both read the SAME working tree ──┘
```

Design notes that came out of live use:

- the panel is a **reading** of a tree other hands also write (an agent turn, a
  colleague's push, a shell) — so it needs an explicit **reload**, and it should
  say what it is showing and when it read it;
- **the file list is the operator's index**: let them resize it, keep one line
  per file, and keep state marks (`new`, `edited`) smaller than the filenames
  they annotate;
- **a destructive control confirms**, and says what breaks: which references
  would dangle, which receipts would be overwritten;
- **a structured file needs a structured editor**: indent on Tab, keep the
  indentation on Enter, line numbers (errors are reported by line), no soft
  wrap, and a warning for a literal tab in a space-indented format;
- **attach, don't paste**: a file or object row that can be dragged into chat
  (or attached with one control) sends a *ref* — which is exactly what the pull
  tool in OP 50 consumes.

## Verify

Run these in order; each one has failed for real:

1. Ask the agent to list its tools. Expect the app's MCP tools, the workspace
   pull, and the CLI built-ins your deployment allows.
2. Ask for something that needs the app's surface (a search). Watch the row
   complete — not just start.
3. Revoke the grant in Connection Hub, ask again: expect a consent card naming
   the claims, and the same turn's retry to work after approving.
4. Attach a file to a message and ask about it. The agent should pull it, not
   receive it.
5. Ask it to change a file, then look at the desk: the change is there, marked
   uncommitted.
6. Commit from the desk, then `git log` in the store from a shell.
7. **Reopen the conversation**: answer, activity rows, cost, elapsed time,
   title, and any attached object must all still be there.
8. Run two turns and compare time-to-first-token. The second should be faster
   (the prompt cache held).
9. From a shell in the runtime, call the MCP surface with the agent's own
   bearer. Confirm 200 *and* a clean connection close.

## Common failures

| Symptom | Cause | Fix |
| --- | --- | --- |
| the agent reports no MCP tools, config looks right | config not named on the command line; workspace untrusted; reserved server name | read the CLI's `system` init event; fix all three |
| a tool call hangs, then "transport dropped mid-call" | a hop in front of the deployment broke the event stream | JSON framing + `self_hosted: true` on the connection |
| a file the agent read appears as its answer | tool results arrive as `user` events | render them as activity rows, never as text |
| every turn is slow to first token | the workspace path changes per turn | one workspace per conversation |
| `delegated_access_grants_not_delegable` | the endpoint or its claim is missing from the hub catalog | declare both halves: `resources` and `capabilities` |
| cost/time show live and vanish on reload | a lifecycle hook overrode the base without `super()` | call the base first, forwarding `econ_ctx` |
| every conversation is "Untitled" | the title role resolves to no model | declare it in `role_models` |
| the agent asks permission for a command its runbook demands | the tool is not in the allow list | permit it; a lane cannot answer a prompt |
| a `.venv` appears in the store's `git status` | the toolchain was built inside the package | machine-local cache keyed by the requirements hash |

## Done means

- A colleague with a browser and no checkout can ask the resident agent to
  change a file, see the change on the desk, and commit it.
- An engineer with a checkout and their own CLI agent works the same files, the
  same runbook, and the same git history.
- Every capability the agent has is one the signed-in user granted, per agent,
  revocable, and named after its consequence.
- The workings are visible in the chat, the answer is clean, and a reopened
  conversation shows all of it.

## Read next

- [Build An App With Several Agents](app-with-agents-README.md)
- [Claude Code Agent](../../sdk/agents/claude/claude-code-README.md)
- [Bundle Agent Integration](../../sdk/bundle/bundle-agent-integration-README.md)
- [Common Bundle Integration Failures](../../sdk/bundle/build/how-to-avoid-common-bundle-integration-failures-README.md)
