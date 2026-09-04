---
id: repo:kdcube-ai-app/agents/integration/claude-bundle/README.md
title: "Run the Hosted Claude Fixture"
summary: "Stage the Claude Code fixture used by optional hosted acceptance, verify its runtime behavior, and change its task."
tags: ["bundle", "claude-code", "agent-harness", "workspace", "economics"]
keywords: ["ClaudeCodeAgent", "turn workspace publish", "LiveLaneWatch", "session lineage"]
updated_at: 2026-09-04
see_also:
  - repo:kdcube-ai-app/agents/integration/README.md
  - repo:kdcube-ai-app/agents/claude/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/agent-in-the-runtimes-README.md
---
# Run the Hosted Claude Fixture

## What it is

This app bundle is the target used by `agents/integration/claude.py`. The
bundle runs Claude Code inside a prepared KDCube deployment; the parent script
drives it through chat and SSE.

## Run it

Follow the [hosted acceptance instructions](../README.md), merge the descriptor
files from `../config/claude/` into the runtime configuration, reload the
bundle, then run from the repository root:

```bash
.venv/bin/python agents/integration/claude.py \
  --workdir ~/.kdcube/kdcube-runtime/<tenant>__<project>
```

The runtime loads this package as `harness-claude-demo@1-0`.

## What the demo shows

The bundle proves that hosted Claude Code can:

- resume one Claude session per conversation;
- stream text and tool activity through the communicator;
- receive live follow-up and steer events;
- run under turn admission and economics accounting; and
- publish a PDF and an XLSX from its governed conversation workspace.

`services/artifact_writer.py` creates the two files from `research.json` after
Claude performs the research.

## Change the demo

- Change model, credentials, timeout, or allowed tools in `entrypoint.py` and
  keep `../config/claude/` aligned.
- Change Claude's task and workspace instructions in `services/turn.py`.
- Change file generation or filenames in `services/artifact_writer.py`.
- Update the parent acceptance assertions when expected outputs change.
- Run `tests/test_artifact_writer.py` after changing file generation.
