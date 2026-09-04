---
id: repo:kdcube-ai-app/agents/native/README.md
title: "Run the Native ReAct Example"
summary: "Run ReactSolverV2 directly, inspect its events and durable conversation, and change its research-to-files task."
tags: ["agents", "native-react", "harness", "standalone", "demonstration"]
keywords: ["ReactSolverV2", "DirectAgentHarness", "Postgres conversation", "ChatCommunicator", "accounting"]
updated_at: 2026-09-04
see_also:
  - repo:kdcube-ai-app/agents/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/components/chat-with-react-agent-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/react/how/how-to-construct-react-agent-README.md
---
# Run the Native ReAct Example

## What it is

This program runs KDCube's maintained `ReactSolverV2` as a local Python
process. It uses the direct self-hosted SDK mode defined in the [parent
README](../README.md) and gives the agent two local tools: web search and
PDF/XLSX creation.

## Run it

Start Redis and Postgres with the [parent instructions](../README.md), then:

```bash
cd agents/native
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

`--check` builds the agent without services or a model call. `--infra-check`
opens Redis, creates the Postgres conversation tables, and opens storage.

## What the demo shows

Turn one searches the web. Turn two recalls the findings and creates a PDF and
an XLSX file. A passing run ends with `demonstration: PASS`.

Inspect:

```text
output/
  communicator.jsonl
  conversation-store/
  turn-01-*/
  turn-02-*/
```

This proves streamed communicator events, Redis accounting, two durable
Postgres turns, readable storage payloads, and native ReAct timeline
continuity. The PDF and XLSX are under the second turn's output directory.

## Change the demo

- Change `agent.topic`, model, limits, infrastructure, storage, or output in
  `config.local.yaml`.
- Change the two `prompt=` arguments in `main_async()` in `agent.py`.
- Add or replace tools in `tools.py`, then update `tools_specs`,
  `tool_runtime`, and the allowed tool names in `agent.py`.
- Change the final required-file check in `agent.py` when the task creates
  different outputs.

To use another provider, change `model.provider`, `model.name`, and
`model.api_key_ref` together. The model-service helper recognizes OpenAI,
Anthropic, Google, and OpenRouter.
