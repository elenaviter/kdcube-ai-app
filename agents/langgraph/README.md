---
id: repo:kdcube-ai-app/agents/langgraph/README.md
title: "Run the LangGraph Agent"
summary: "Run LangGraph through the KDCube model bridge with durable checkpoints and harness accounting."
tags: ["agents", "langgraph", "langchain", "harness", "accounting", "standalone", "web-search"]
keywords: ["KDCubeChatModel", "KDCube Web Search", "stream_model_text_tracked", "AsyncPostgresSaver", "ChatCommunicator"]
updated_at: 2026-09-05
see_also:
  - repo:kdcube-ai-app/agents/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md
---
# Run the LangGraph Agent

## What it is

This directory runs LangChain `create_agent` through KDCube's
`KDCubeChatModel`. Model streaming is accounted by the harness, while Postgres
stores both the conversation and LangGraph checkpoints.

## Run it

```bash
cd agents/langgraph
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python setup_local.py --provider anthropic
cp config.template.yaml config.local.yaml
docker compose --env-file .env -f compose.yaml up -d --wait
.venv/bin/python agent.py --check
.venv/bin/python agent.py --infra-check
.venv/bin/python agent.py
```

The provider key prompt is hidden. Local secrets and generated descriptors are
ignored by Git.

## What the demo shows

Turn one searches through KDCube Web Search. Turn two resumes the same graph
thread and creates a PDF and XLSX. A successful run proves live events,
accounted model calls, durable harness turns, and Postgres graph checkpoints;
it ends with `demonstration: PASS`.

## Change the demo

Edit `config.local.yaml` to change instructions, run directory, tools, skills,
topic, or limits. Web Search's allowlist, blocklist, and SSRF policy are under
the `web_search` tool row's `settings`. Edit
`descriptors.local/assembly.yaml` for model and infrastructure; edit
`descriptors.local/secrets.yaml` for credentials. The shipped model is
`claude-haiku-4-5-20251001`. Add LangChain tools in `tools.py` and select their
IDs in YAML.
