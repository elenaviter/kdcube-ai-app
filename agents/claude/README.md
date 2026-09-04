---
id: repo:kdcube-ai-app/agents/claude/README.md
title: "Run the Claude Code Example"
summary: "Run Claude Code through ClaudeCodeAgent with a durable Git transcript and a Postgres-backed harness conversation."
tags: ["agents", "claude-code", "harness", "conversation", "standalone"]
keywords: ["ClaudeCodeAgent", "Claude transcript", "Git session store", "ChatCommunicator", "accounting"]
updated_at: 2026-09-04
see_also:
  - repo:kdcube-ai-app/agents/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/components/chat-with-claude-code-agent-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/claude/claude-code-workspace-bootstrap-README.md
---
# Run the Claude Code Example

## What it is

This program runs the local Claude Code CLI through `ClaudeCodeAgent`. The
harness stores its durable conversation in Postgres and configured payload
storage. The SDK stores Claude's resumable JSONL transcript on a dedicated Git
branch selected from tenant, project, user, conversation, and agent identity.

## Run it

Install and authenticate Claude Code. Initialize
`agents/shared/descriptors.local` with
`cd agents/shared && python3 configure.py --provider none`, then start
Redis/Postgres using the [parent instructions](../README.md) and run:

```bash
cd agents/claude
claude --version
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp config.template.yaml config.local.yaml
.venv/bin/python agent.py --config config.local.yaml --descriptors ../shared/descriptors.local --check
.venv/bin/python agent.py --config config.local.yaml --descriptors ../shared/descriptors.local --infra-check
.venv/bin/python agent.py --config config.local.yaml --descriptors ../shared/descriptors.local
```

`--infra-check` pushes an empty bootstrap commit to the configured transcript
branch. This verifies Git read/write transport before Claude uses any tokens.
For key-based CLI execution, set `services.anthropic.claude_code_key` in the
ignored `../shared/descriptors.local/secrets.yaml`.

## What the demo shows

Turn one uses `WebSearch` and writes `research.json`. Turn two resumes the same
Claude session and creates a PDF and XLSX under
`output/runs/claude-demo/workspace/deliverables/`. The SDK bootstraps the
transcript before each turn and publishes it after success or failure. The
harness independently persists both turns and accounting evidence.

A passing run ends with `demonstration: PASS`. Run the command again to prove
that a new process can restore the same `agent.conversation_id` from Git.

## Change the demo

- Change Claude's model, instructions, allowed CLI tools, skills, task, or
  timeout in `config.local.yaml`.
- Change `agent.conversation_id` to start a different transcript branch.
- Change `storage.claude_code_session.repo` in
  `../shared/descriptors.local/assembly.yaml` to use a private remote.
- Put HTTPS Git credentials under `services.git` in
  `../shared/descriptors.local/secrets.yaml`, or configure the standard SSH fields
  under `services.git` in `assembly.yaml`.
- Change `prompts` and `expected_files` in `agent.py` for another scenario.

Workspace files and the Claude JSONL transcript are separate durable surfaces:
the workspace contains task artifacts; the Git session branch carries Claude's
conversation state.
