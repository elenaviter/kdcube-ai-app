---
id: repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/workspace/README.md
title: "Agent Harness Workspace"
summary: "Framework-neutral refs, artifacts, paths, change detection, and materialization for distributed turn workspaces."
status: active
tags: ["runtime", "harness", "workspace", "artifacts", "refs", "materialization"]
updated_at: 2026-08-21
keywords:
  [
    "turn workspace",
    "conv:fi",
    "git/projects",
    "files",
    "git/snapshots",
    "OUTPUT_DIR",
    "pull",
    "checkout",
    "event_ref",
    "object_ref",
  ]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/workspace/references-and-paths-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/workspace/workspace-model-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/workspace/artifact-storage-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/workspace/workspace-lifecycle-and-distribution-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/events/artifact-resolution-and-materialization-README.md
---
# Agent Harness Workspace

The harness workspace is a sparse, per-turn physical view over durable
conversation and owner-domain artifacts.

```text
accepted event occurrence
  event_ref = conv:ev:...             read the record; never materialize it
       |
       +-- optional object_ref = cnv:... | task:... | conv:fi:...
                                      |
                                trusted resolution
                                      |
                     +----------------+----------------+
                     |                                 |
                  pull                              checkout
             read-only source view       editable current-turn destination
                                               git/projects/... or files/...
```

The shared workspace layer owns:

- canonical `conv:fi:` refs and `conv_<conversation_id>` ownership segments;
- physical/logical path construction and parsing;
- the semantic distinction between project state, produced files, snapshots,
  attachments, and external evidence;
- artifact records independent of timeline placement;
- artifact-root resolution, snapshots, diffs, and file-item production;
- trusted source resolution under runtime-bound identity;
- collision-safe read-only pull paths;
- transactional checkout with explicit source, destination, and strategy.

It does not own model-facing tool names. The native ReAct Agent exposes
`react.pull`, `react.checkout`, `react.read`, and related operations. The ported
LangGraph example binds `pull_files`, `checkout`, `read_file`, and `run_python`.
A hosted Claude Code wrapper can expose the same pull/checkout contract through
a local MCP server and let Claude use its native file tools on returned paths.
These are adapter bindings over one workspace contract, not aliases for one
agent framework.

Publication is a separate transition. A file in `files/...` is editable
workspace state until an adapter's trusted host stores it as a conversation
file, returns a durable `conv:fi:` ref, and emits or records the file result.

## Area Semantics

| Area | Meaning |
| --- | --- |
| `git/projects/` | Editable durable project/app state. |
| `files/` | Produced artifacts and deliverables. |
| `git/snapshots/` | Story, canvas, wizard, or workflow snapshots. |
| `attachments/` | Current-turn user uploads. |
| `external/` | Materialized external-event or owner-domain evidence. |

Visibility is separate from area. An internal file remains a file; a timeline
message does not become an artifact merely because it is visible.

## Canonical Documents

- [References And Paths](references-and-paths-README.md)
- [Workspace Model](workspace-model-README.md)
- [Artifact Storage](artifact-storage-README.md)
- [Workspace Lifecycle And Distribution](workspace-lifecycle-and-distribution-README.md)
- [Artifact Resolution And Materialization](../events/artifact-resolution-and-materialization-README.md)
