---
id: ks:docs/next/copilot-new-namespaces-knowledge-space-index.md
title: "Copilot New Namespaces Knowledge Space Index"
summary: "Proposal for knowledge‑space namespaces (OUT_DIR, knowledge, workspace) for copilot flows."
tags: ["next", "roadmap", "thoughts", "copilot", "namespaces", "knowledge-space", "react"]
keywords: ["OUT_DIR", "knowledge space", "workspace", "react.read", "namespace semantics"]
see_also:
  - ks:docs/sdk/bundle/bundle-dev-README.md
  - ks:docs/README.md
  - ks:docs/sdk/agents/react/artifact-discovery-README.md
---
## 1) Three data spaces (stable semantics)

A. OUT_DIR (existing, per‑turn, RW)

- Used for tool outputs and files created in the current turn.
- Paths: fi:<turn_id>.files/...
- Read via react.read and write via react.write/react.patch.

B. Knowledge Space (new, read‑only)

- System‑prepared reference corpus: docs, indexes, cloned repos, manuals.
- Paths: ks:<relpath>
- Read/search only. No writing/patching/execution here.

C. Conversation Workspace (future, RW)

- Shared project state across turns (true copilot workspace).
- Should be a different namespace later (e.g., wk:).
- Not implemented yet.

This keeps semantics clean: ks is reference, fi is per‑turn, wk is future shared project.

———

## 2) Should the index be shown when a skill is loaded?

No. Skills and knowledge are different surfaces:

- Skills are in the system prompt by design (catalog + instructions).
- Knowledge Space should be discoverable on demand, not always loaded.

Recommended behavior:

- Always include a tiny pointer in the system prompt:
    - “Knowledge Space available via ks:. Use react.search_knowledge(query=...) or
      react.read('ks:index.md').”
- Optionally add an announce block once per conversation:
    - “Knowledge Space index: ks:index.md”.

That keeps token usage under control and avoids cache churn.

———

## 3) How to build the Knowledge Space index

### Best default (file‑based corpus)

At bundle load time:

1. Build ks:index.json with metadata for each doc.
2. Build ks:index.md (human summary: sections + top docs + how to use).

This gives a single stable entry point the agent can load.

### Front‑matter metadata (simple and powerful)

Use YAML front‑matter in each doc:

  ---
id: docs.platform.bundles
title: Bundles & Workflows
summary: How bundles are loaded, configured, and run.
tags: [bundles, runtime, sdk]
see_also:
- ks:docs/platform/react.md
- ks:docs/platform/streaming.md
  ---

Index build simply scans and aggregates these headers.

———

## 4) What about DB/graph knowledge bases?

This is exactly why ks: should be logical, not tied to filesystem.

Define a resolver for ks: paths:

ks:<id>  ->  loader returns {text, mime, links}

### Example resolution strategies

- ks:docs/platform/react → file on disk
- ks:deploy/docker/all_in_one_kdcube/docker-compose.yaml → deployment artifact
- ks:kb/123456 → Postgres/Neo4j lookup
- ks:article/CVE-2024-1234 → KB row

You can plug in resolvers without changing the ReAct protocol.

———

## 5) How to support semantic search

react.search_knowledge is the primary entry for knowledge search.

Return results as:

[{ "path": "ks:docs/platform/react", "snippet": "...", "score": 0.82 }]

Then the agent can call:

react.read(["ks:docs/platform/react"])

This works for files, SQL, graph, or hybrid search.

———

## 6) Recommendation summary

### Index strategy

- Always generate ks:index.json + ks:index.md.
- Keep ks:index.md short (sectioned summary + top entries).

### Presentation strategy

- System prompt: one‑line pointer to Knowledge Space.
- First turn announce (optional): Knowledge Space index: ks:index.md.

### Protocol strategy

- ks: is the only namespace for knowledge.
- File‑based now; resolver‑based later.

———
