---
id: repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/workspace/workspace-model-README.md
title: "Agent Harness Workspace Model"
summary: "Authoritative contract for the sparse per-turn workspace, project state, produced files, snapshots, attachments, and materialization."
status: active
tags: ["runtime", "harness", "workspace", "pull", "checkout", "artifacts"]
updated_at: 2026-08-21
keywords:
  [
    "sparse workspace",
    "conv:fi",
    "git/projects",
    "files",
    "git/snapshots",
    "materialization",
    "bound user scope",
  ]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/workspace/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/workspace/references-and-paths-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/workspace/workspace-lifecycle-and-distribution-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/events/artifact-resolution-and-materialization-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/react/react-tools-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/react/react-announce-README.md
---
# Agent Harness Workspace Model

The harness workspace is sparse:

```text
each turn receives a fresh physical workspace
durable logical refs survive across turns and workers
bytes appear locally only after ingress, production, pull, or checkout
```

This model is shared. The native ReAct Agent's `[WORKSPACE]` ANNOUNCE and model
tools are one adapter over it; ported agents consume the same path and
materialization contracts through their own bindings.

## Physical Layout

```text
OUTPUT_DIR/
  turn_<current>/
    git/projects/<project_scope>/...      # editable durable project state
    files/<artifact_scope>/...            # produced artifacts/deliverables
    git/snapshots/<snapshot_scope>/...     # story/workflow snapshots
    attachments/<name>                    # current user uploads
    external/<kind>/attachments/<id>/...  # external/domain evidence
  conv_<source_conversation>/
    turn_<source_turn>/...                 # pulled cross-conversation material
```

Only `OUTPUT_DIR`-relative paths are part of the harness contract. Host paths,
container mount prefixes, object-store URIs, and runtime metadata roots are not
agent file paths.

## Logical Refs

The durable identity for a file includes the conversation owner:

```text
conv:fi:conv_<conversation_id>.turn_<turn_id>.git/projects/<scope>/<path>
conv:fi:conv_<conversation_id>.turn_<turn_id>.files/<scope>/<path>
conv:fi:conv_<conversation_id>.turn_<turn_id>.git/snapshots/<scope>/<path>
conv:fi:conv_<conversation_id>.turn_<turn_id>.user.attachments/<name>
conv:fi:conv_<conversation_id>.turn_<turn_id>.external.<kind>.attachments/<id>/<name>
```

See [References And Paths](references-and-paths-README.md) for the full
conversation ref grammar and authority boundary.

## Area Is Meaning

Do not infer artifact meaning from extension or visibility.

| Area | Put here | Do not put here |
| --- | --- | --- |
| `git/projects/` | Source trees, app state, editable durable project files. | One-off reports and downloads. |
| `files/` | PDF/DOCX/PPTX/XLSX/HTML/Markdown reports, archives, diagnostics, render sources, exported deliverables. | Durable project source trees. |
| `git/snapshots/` | Canvas/story/wizard/workflow state snapshots. | User deliverables unless explicitly exported. |
| `attachments/` | User-uploaded bytes. | Assistant-produced output. |
| `external/` | Event attachments and owner-domain evidence. | Direct editable project state. |

Visibility is orthogonal:

```text
git/projects + external  -> visible project artifact
git/projects + internal  -> hidden project state
files        + external  -> user/client deliverable
files        + internal  -> runtime/agent artifact
```

## Records, Locators, And Materialization

An event record ref and its referenced object are distinct:

```text
event_ref  conv:ev:conv_c1.turn_9.events/canvas/evt_17   readable occurrence
object_ref cnv:main@52                                   owner object locator
```

`conv:ar:`, `conv:ev:`, `conv:so:`, `conv:su:`, `conv:tc:`, and
`conv:ws:` identify records. They do not become filesystem bytes through pull
or checkout. An adapter renders or reads the record, then uses a separately
supplied materializable `object_ref` or artifact ref when local bytes are
needed.

```text
logical ref or owner ref
          |
          v
trusted resolver under runtime identity
          |
          v
bytes copied into a collision-safe read-only source path
          |
          v
adapter returns logical_path + physical_path
```

The runtime, not the model, binds tenant, project, actor, user, and authority.
For owner refs such as `mem:`, `task:`, or `cnv:`, the owner provider/rehoster
authorizes and chooses the resulting artifact semantics.

The framework-neutral pull primitive is
`runtime.harness.workspace.pull_refs_into_workspace(...)`. Exact `conv:fi:`
files use the conversation byte resolver. Owner refs use an adapter-supplied
trusted resolver, which must return a pinned `conv:fi:` ref and a local source.
The target preserves source conversation, turn, namespace, and relative path,
so same-basename files cannot overwrite each other. Pulled data is made
read-only.

## Editable Checkout

Checkout resolves its source itself; pull is not a prerequisite:

```json
{
  "items": [
    {
      "from": "conv:fi:conv_c1.turn_7.files/source.pdf",
      "to": "files/pdf-review/working.pdf",
      "strategy": "replace"
    }
  ]
}
```

- `from` is an exact `conv:fi:` ref or an authorized owner locator.
- `to` is relative to the active workspace and lies below `git/projects/...`
  or `files/...`; runtime context supplies conversation and turn identity.
- `replace` makes a file or directory target exactly match the source and is
  also the reset operation.
- `overlay` merges a source directory into a destination directory while
  retaining destination-only entries.

The harness resolves and validates the entire item list before mutation,
rejects overlapping targets and symlinks, then applies the batch with rollback.
Each result reports the pinned source and exact current logical and physical
path.

## Adapter Bindings

| Capability | Native ReAct Agent | Hosted LangGraph example | Hosted Claude Code pattern |
| --- | --- | --- | --- |
| Read event/record | `react.read` | framed event plus `read_file` for supported file refs | event in prompt; wrapper-specific record reader |
| Read-only bytes | `react.pull` | `pull_files` | local MCP `pull` plus native Read/Grep/Bash |
| Editable copy/reset | `react.checkout` | `checkout` | local MCP `checkout` plus native Edit/Write/Bash |
| Produce files | write/render/exec tools | `run_python` | native file/code tools |
| Conversation publication | tool/output hosting contract | `run_python` hosts declared outputs | reusable Claude turn-workspace binding exposes local MCP `publish` through the shared host policy |

A ported agent is not required to expose native ReAct Agent tool names. It can:

1. select refs from its own input/state;
2. call the shared resolver or pull primitive;
3. pull read-only bytes or checkout editable bytes into its turn workspace;
4. inspect or modify the returned physical path using its own facilities;
5. ask a trusted host to publish selected outputs as canonical `conv:fi:` refs.

The worked LangGraph port uses this pattern for event objects, attachments, and
code-exec outputs. The Press Claude Code wrapper uses a turn-scoped local MCP
server plus a trusted parent broker: the child sends refs and `files/...`
paths, while identity, owner resolvers, credentials, and conversation hosting
remain in the parent process.

## ReAct `[WORKSPACE]`

ReAct rebuilds an ANNOUNCE view every round:

```text
LOCAL
  materialized bytes in this worker now

REMOTE git branch
  durable project scopes available from conversation lineage
  but not local until pulled or checked out
```

Critical distributed-runtime invariant:

> **For the current round, only paths listed under ANNOUNCE `[WORKSPACE] LOCAL`
> are known to exist on the worker.**

A logical ref, content preview, or pull completed in a prior turn is durable
evidence that an object exists, not evidence that its bytes are local now.
Before `react.rg`, code, rendering, or another readonly local-bytes operation
uses anything absent from the current `LOCAL` list, call `react.pull` in this
turn. When durable source data must become editable, call `react.checkout`
directly with an explicit current-turn `git/projects/...` or `files/...`
target. Checkout resolves its source; a prior pull is needed only for a
separate readonly comparison. A direct `react.read` preview does not establish
locality; trust the next round's `LOCAL` list.

That presentation is ReAct-specific. The underlying sparse workspace and
logical refs are shared.

## Debug Checklist

1. Is the value a logical ref or an `OUTPUT_DIR`-relative physical path?
2. Does a durable ref include `conv_<conversation_id>`?
3. Are the bytes materialized in this worker?
4. Is the area correct for the object's meaning?
5. Did resolution run under the expected tenant/project/user/authority?
6. For an owner namespace, is its provider/rehoster registered and authorized?
7. Is an adapter confusing project checkout with ordinary byte pull?
