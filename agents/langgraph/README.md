---
id: repo:kdcube-ai-app/agents/langgraph/README.md
title: "Run the LangGraph Example"
summary: "Run LangChain create_agent through KDCubeChatModel, inspect its accounted durable conversation, and change its research-to-files task."
tags: ["agents", "langgraph", "langchain", "harness", "accounting", "standalone"]
keywords: ["KDCubeChatModel", "stream_model_text_tracked", "AsyncPostgresSaver", "ChatCommunicator"]
updated_at: 2026-09-04
see_also:
  - repo:kdcube-ai-app/agents/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/components/chat-with-langgraph-agent-README.md
---
# Run the LangGraph Example

## What it is

This program runs LangChain `create_agent` through KDCube's
`KDCubeChatModel`. The shared platform descriptors configure the model,
secrets, Redis, Postgres, storage, and economics. `config.local.yaml` selects
the graph's instructions, tools, skills, limits, and task.

## Run it

Initialize `agents/shared/descriptors.local` and start Redis/Postgres using the
[parent instructions](../README.md), then run:

```bash
cd agents/langgraph
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp config.template.yaml config.local.yaml
.venv/bin/python agent.py --config config.local.yaml --descriptors ../shared/descriptors.local --check
.venv/bin/python agent.py --config config.local.yaml --descriptors ../shared/descriptors.local --infra-check
.venv/bin/python agent.py --config config.local.yaml --descriptors ../shared/descriptors.local
```

`--check` constructs the graph before contacting services or a model provider.
`--infra-check` creates both the harness conversation tables and LangGraph's
checkpoint tables, then verifies Redis and storage.

## What the demo shows

Turn one searches the web. Turn two resumes the graph thread and creates
`output/runs/langgraph-*/research-brief.pdf` and `research-data.xlsx`. A passing
run proves streamed tool/model events, accounted calls, Postgres checkpoint
continuity, and two durable harness turns. It ends with `demonstration: PASS`.

## Change the demo

- Change instructions, enabled tools, skills, topic, or limits in
  `config.local.yaml`.
- Change the model in `../shared/descriptors.local/assembly.yaml` and put its
  canonical provider key in `../shared/descriptors.local/secrets.yaml`.
- Add a LangChain tool in `tools.py`, add its ID to the supported set in
  `agent.py`, and select it in `config.local.yaml`.
- Change `prompts` and `expected_files` in `agent.py` for another scenario.

Keep `KDCubeChatModel` when model streaming must pass through the harness's
accounting and communicator bridge.
