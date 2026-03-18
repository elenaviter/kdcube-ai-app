---
id: ks:docs/sdk/agents/react/agent-workspace-collboration-README.md
title: "Agent Workspace Collaboration"
summary: "How react.read, react.write, react.patch, and react.search_files cooperate over the versioned OUT_DIR workspace."
tags: ["sdk", "agents", "react", "workspace", "artifacts", "files"]
keywords: ["outdir", "workdir", "react.read", "react.search_files", "react.patch", "versioned workspace"]
see_also:
  - ks:docs/sdk/agents/react/react-turn-workspace-README.md
  - ks:docs/sdk/agents/react/artifact-discovery-README.md
  - ks:docs/sdk/agents/react/react-tools-README.md
---
# Agent Workspace Collaboration

This document explains the working model the React agent should use when combining the filesystem tools.

## Core model

The agent does **not** work against one mutable flat directory. It works against a **versioned OUT_DIR**:
- Assistant-created files are namespaced by turn: `turn_<id>/files/<subpath>`
- User attachments are namespaced by turn: `turn_<id>/attachments/<subpath>`
- Runtime-owned folders such as `logs/` also live in OUT_DIR

This gives two properties at once:
- history is preserved in place
- the agent can still reason about a current logical workspace view

The logical workspace view is:
- for `files/<subpath>`: the latest relevant turn version
- for `attachments/<subpath>`: the attached artifact under its original turn namespace
- for non-versioned OUT_DIR folders like `logs/`: the physical OUT_DIR file itself

## Read / search / write responsibilities

`react.read`
- reads by logical path only
- supports the established turn-scoped forms:
  - `fi:<turn_id>.files/<subpath>`
  - `fi:<turn_id>.user.attachments/<subpath>`
- also supports any readable OUT_DIR file via:
  - `fi:<outdir-relative-path>`
  - example: `fi:logs/docker.err.log`

`react.search_files`
- searches under `outdir` or `workdir`
- returns discovery metadata only
- result shape:
  - `root`
  - `hits[]`
- each hit contains:
  - `path`: relative to the searched root
  - `size_bytes`
  - `logical_path` for OUT_DIR hits

`react.write`
- creates or replaces files in the current turn file namespace
- the effective target is always under `<current_turn>/files/...`

`react.patch`
- updates an existing file in the current turn file namespace
- historical `turn_X/files/...` references are treated as source material and patched into the current turn output

## Safe collaboration rules

The intended cooperation pattern is:
1. Use `react.search_files` to discover candidate files.
2. If the hit is under OUT_DIR, take its `logical_path`.
3. Use `react.read` on that `logical_path` to load content into context.
4. If editing is needed, write or patch into `<current_turn>/files/...`.

This keeps:
- discovery separate from loading
- loading separate from mutation
- turn history preserved

## Current limitations

- `react.read` is OUT_DIR-aware, not general workdir-aware.
- workdir hits from `react.search_files` are still discovery-only with the current toolset.
- `react.patch` is for assistant file artifacts, not arbitrary runtime-owned files like `logs/...`.

## Examples

Search logs and read one:
```json
{
  "root": "outdir/logs",
  "hits": [
    {
      "path": "docker.err.log",
      "size_bytes": 18342,
      "logical_path": "fi:logs/docker.err.log"
    }
  ]
}
```

Search turn files and read one:
```json
{
  "root": "outdir",
  "hits": [
    {
      "path": "turn_1773261747483_vfm2tt/files/kdcube-market-comparison.md",
      "size_bytes": 9123,
      "logical_path": "fi:turn_1773261747483_vfm2tt.files/kdcube-market-comparison.md"
    }
  ]
}
```
