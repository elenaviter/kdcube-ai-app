---
id: repo:kdcube-ai-app/agents/README.md
title: "Run an Agent from the KDCube SDK"
summary: "Run native ReAct, LangGraph, or Claude Code directly from Python with the KDCube Agent Harness."
tags: ["agents", "harness", "native-react", "langgraph", "claude-code", "quickstart"]
keywords: ["agent examples", "DirectAgentHarness", "Redis", "Postgres", "PDF", "XLSX"]
updated_at: 2026-09-04
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/quick-start-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/README.md
---
# Run an Agent from the KDCube SDK

## What it is

Choose an agent. Each directory is a small runnable program:

| Agent | Core |
| --- | --- |
| [native](native/README.md) | KDCube native ReAct |
| [langgraph](langgraph/README.md) | LangGraph through `KDCubeChatModel` |
| [claude](claude/README.md) | Claude Code through `ClaudeCodeAgent` |

The [shared setup](shared/README.md) contains descriptors and support services.
The separate [hosted-runtime path](advanced/integration/README.md) is advanced
acceptance testing.

> **Shell prerequisite:** these programs run an agent directly from source.
> For a ready runtime with chat UI, authentication, managed tools,
> tool-execution enforcement, isolated workspaces, and app hosting, use
> [KDCube Quick Start](../app/ai-app/docs/quick-start-README.md).

## Run it

Initialize the shared standard descriptors and start Redis/Postgres:

```bash
cd agents/shared
python3 configure.py --provider openai
docker compose --env-file .env -f compose.yaml up -d --wait
```

Then run one agent:

```bash
cd ../native                     # or ../langgraph / ../claude
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp config.template.yaml config.local.yaml
.venv/bin/python agent.py --config config.local.yaml --descriptors ../shared/descriptors.local --check
.venv/bin/python agent.py --config config.local.yaml --descriptors ../shared/descriptors.local --infra-check
.venv/bin/python agent.py --config config.local.yaml --descriptors ../shared/descriptors.local
```

The initializer asks for the provider key without echoing it. For a Claude-only
run authenticated by the Claude CLI, use `--provider none` and run
`claude --version` first.

The complete executable procedure, including isolated code execution, is
[Run the Agent Harness from Python](../app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md).

## What the demo shows

Each agent searches the public web, carries context into another turn, and
creates a real PDF and XLSX file. It prints live communicator events and ends
with `demonstration: PASS` after checking its artifacts and durable records.

All three runners store conversation records in Postgres and payloads in
configured storage. LangGraph also stores graph checkpoints in Postgres.
Claude stores its resumable CLI transcript on a per-conversation Git branch.
Redis holds the per-turn accounting/event mirror.

## Change the demo

Edit `<agent>/config.local.yaml` to change instructions, tools, skills, topic,
limits, or output. Edit `shared/descriptors.local/assembly.yaml` to change the
model, infrastructure, storage, Git transcript store, or executor. Credentials
belong in `shared/descriptors.local/secrets.yaml`; prices and economics policy
belong in `shared/descriptors.local/economics.yaml`.

Stop support services from `agents/shared/`:

```bash
docker compose --env-file .env -f compose.yaml down
```
