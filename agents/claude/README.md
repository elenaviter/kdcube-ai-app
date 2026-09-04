---
id: repo:kdcube-ai-app/agents/claude/README.md
title: "Run the Claude Code Example"
summary: "Run a local Claude Code process through ClaudeCodeAgent, inspect its accounted durable conversation, and change its research-to-files task."
tags: ["agents", "claude-code", "harness", "conversation", "standalone"]
keywords: ["ClaudeCodeAgent", "Claude session", "ChatCommunicator", "accounting"]
updated_at: 2026-09-04
see_also:
  - repo:kdcube-ai-app/agents/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/components/chat-with-claude-code-agent-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/claude/claude-code-workspace-bootstrap-README.md
---
# Run the Claude Code Example

## What it is

This program runs the local Claude Code CLI through `ClaudeCodeAgent`. Claude
uses the direct self-hosted SDK mode defined in the [parent README](../README.md)
and keeps one session and workspace across two turns while the KDCube harness
records events, accounting, and the durable conversation.

## Run it

Install and authenticate Claude Code, then start Redis and Postgres with the
[parent instructions](../README.md). Run:

```bash
cd agents/claude
claude --version
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp config.template.yaml config.local.yaml
set -a
. ../.env
set +a
.venv/bin/python agent.py --config config.local.yaml --check
.venv/bin/python agent.py --config config.local.yaml --infra-check
.venv/bin/python agent.py --config config.local.yaml
```

Claude can use its existing CLI login. For key-based execution, export the
secret named by `claude.api_key_ref` before the final command.

## What the demo shows

Turn one uses Claude's `WebSearch` and writes `research.json`. Turn two resumes
the same Claude session and creates:

```text
output/workspace/research.json
output/workspace/deliverables/research-brief.pdf
output/workspace/deliverables/research-data.xlsx
output/communicator.jsonl
output/conversation-store/
```

A passing run proves streamed Claude events, Redis accounting, Claude session
continuity, two durable Postgres turns, readable storage payloads, and real PDF
and XLSX files. It ends with `demonstration: PASS`.

## Change the demo

- Change `agent.topic`, Claude model, timeout, allowed tools, infrastructure,
  storage, or output in `config.local.yaml`.
- Change `prompts` and workspace instructions in `agent.py`.
- Change `expected_files` in `agent.py` when the task creates different
  outputs.
- Add Python packages used by Claude's Bash commands to `requirements.txt`.

The runner puts this example's virtual-environment `bin` directory on Claude's
`PATH`, so its Bash commands use the packages installed for this example. Bash
and file access operate inside the configured local workspace.
