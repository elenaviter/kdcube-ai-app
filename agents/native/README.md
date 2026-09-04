---
id: repo:kdcube-ai-app/agents/native/README.md
title: "Run the Native ReAct Example"
summary: "Run ReactSolverV2 directly, inspect its events and durable conversation, and change its research-to-files task."
tags: ["agents", "native-react", "harness", "standalone", "demonstration"]
keywords: ["ReactSolverV2", "DirectAgentHarness", "Postgres conversation", "ChatCommunicator", "accounting"]
updated_at: 2026-09-04
see_also:
  - repo:kdcube-ai-app/agents/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/components/chat-with-react-agent-README.md
---
# Run the Native ReAct Example

## What it is

This program runs KDCube's `ReactSolverV2` in your Python process. The shared
platform descriptors configure its model, secrets, Redis, Postgres, storage,
economics, and isolated executor. `config.local.yaml` selects its instructions,
tools, skills, limits, and task.

## Run it

Initialize `agents/shared/descriptors.local` and start Redis/Postgres using the
[parent instructions](../README.md), then run:

```bash
cd agents/native
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp config.template.yaml config.local.yaml
.venv/bin/python agent.py --config config.local.yaml --descriptors ../shared/descriptors.local --check
.venv/bin/python agent.py --config config.local.yaml --descriptors ../shared/descriptors.local --infra-check
.venv/bin/python agent.py --config config.local.yaml --descriptors ../shared/descriptors.local
```

`--check` constructs the agent before contacting services or a model provider.
`--infra-check` creates the conversation tables and verifies Redis and storage.

## What the demo shows

The first turn searches the web. The second creates a PDF and XLSX. A third
conversation must call `react.memsearch` and recover a source from the first
conversation for the same user. Success ends with `demonstration: PASS`.

Inspect `output/communicator.jsonl`, `output/turn-*`, and the shared
`../shared/output/conversation-store`. Postgres holds the durable conversation index;
Redis holds the per-turn accounted-event mirror.

## Change the demo

- Change instructions, enabled tools, skills, topic, or limits in
  `config.local.yaml`.
- Change the model in `../shared/descriptors.local/assembly.yaml` and put its
  canonical provider key in `../shared/descriptors.local/secrets.yaml`.
- Change local tool implementations in `tools.py` and their supported IDs in
  `configuration.py`.
- Enable `exec_tools.execute_code_python` in `config.local.yaml` after building
  the image configured under `platform.services.proc.exec` in `assembly.yaml`.

The full executor procedure is in
[Run the Agent Harness from Python](../../app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md#8-enable-isolated-python-execution).
