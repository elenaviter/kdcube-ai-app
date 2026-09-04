---
id: repo:kdcube-ai-app/agents/langgraph/README.md
title: "Run the LangGraph Example"
summary: "Run LangChain create_agent through KDCubeChatModel, inspect its accounted durable conversation, and change its research-to-files task."
tags: ["agents", "langgraph", "langchain", "harness", "accounting", "standalone"]
keywords: ["KDCubeChatModel", "stream_model_text_tracked", "AsyncPostgresSaver", "ChatCommunicator"]
updated_at: 2026-09-04
see_also:
  - repo:kdcube-ai-app/agents/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/components/chat-with-langgraph-agent-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/langgraph/langgraph-agent-README.md
---
# Run the LangGraph Example

## What it is

This program runs LangChain `create_agent` with KDCube's `KDCubeChatModel`.
It uses the direct self-hosted SDK mode defined in the [parent
README](../README.md). The graph keeps normal LangGraph messages, tools, and
checkpoints while the KDCube harness records events, accounting, and the
durable conversation.

## Run it

Start Redis and Postgres with the [parent instructions](../README.md), then:

```bash
cd agents/langgraph
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp config.template.yaml config.local.yaml
set -a
. ../.env
set +a
.venv/bin/python agent.py --config config.local.yaml --check
.venv/bin/python agent.py --config config.local.yaml --infra-check
export OPENAI_API_KEY='...'
.venv/bin/python agent.py --config config.local.yaml
```

`--check` builds the graph without services or a model call. `--infra-check`
opens Redis and storage and creates both the KDCube conversation tables and
the LangGraph checkpoint tables.

## What the demo shows

Turn one searches the web. Turn two recalls the graph state and creates:

```text
output/research-brief.pdf
output/research-data.xlsx
```

A passing run proves tool events, streamed answer deltas, Redis accounting,
LangGraph checkpoint continuity, two durable KDCube conversation turns, and
readable storage payloads. It ends with `demonstration: PASS`.

## Change the demo

- Change `agent.topic`, model, limits, infrastructure, storage, or output in
  `config.local.yaml`.
- Change `prompts` or the system prompt in `agent.py`.
- Add or replace LangChain tools in `tools.py`; `build_tools()` supplies them
  to `create_agent`.
- Change the final required-file check in `agent.py` when the task creates
  different outputs.

Keep `KDCubeChatModel` when the demonstration should include KDCube model
accounting. A provider model used directly remains outside that accounting
adapter.
