---
id: repo:kdcube-ai-app/agents/README.md
title: "Run Three Agent Cores with the KDCube SDK"
summary: "Run native ReAct, LangGraph, or Claude Code directly with the KDCube SDK, Redis, Postgres, and local conversation storage."
tags: ["agents", "harness", "native-react", "langgraph", "claude-code", "demonstration"]
keywords: ["direct agent examples", "DirectAgentHarness", "Redis", "Postgres", "conversation storage", "PDF", "XLSX"]
updated_at: 2026-09-04
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/quick-start-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/components/chat-with-langgraph-agent-README.md
---
# Run Three Agent Cores with the KDCube SDK

## What it is

Run an agent directly from your own Python process while importing KDCube from
this source checkout as an SDK. No KDCube server is running. Redis, Postgres,
and configured storage are the only support services.

> **Prerequisite:** This page is for running an agent from source in your
> shell. For a ready runtime with chat UI, authentication, managed tools,
> tool-execution enforcement, and app hosting, follow
> [Quick Start: Run KDCube Locally](../app/ai-app/docs/quick-start-README.md).

| Example | Agent core | Read next |
| --- | --- | --- |
| Native | `ReactSolverV2` | [native/README.md](native/README.md) |
| LangGraph | LangChain `create_agent` | [langgraph/README.md](langgraph/README.md) |
| Claude | local Claude Code CLI | [claude/README.md](claude/README.md) |

## Run it

Start Redis and Postgres once:

```bash
cd agents
cp .env.example .env
# Fill AGENT_DEMO_POSTGRES_PASSWORD and AGENT_DEMO_REDIS_PASSWORD.
chmod 600 .env
set -a
. ./.env
set +a
docker compose --env-file .env up -d --wait
```

Then run one example:

```bash
cd langgraph                    # or native / claude
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp config.template.yaml config.local.yaml
.venv/bin/python agent.py --config config.local.yaml --infra-check
export OPENAI_API_KEY='...'     # native and LangGraph defaults
.venv/bin/python agent.py --config config.local.yaml
```

Claude uses its existing CLI login by default.

Use your own Redis or Postgres by changing `infra` in `config.local.yaml` and
skip the Compose command. The runner creates the required KDCube conversation
tables. LangGraph also creates its checkpoint tables.

## What the demo shows

Every example performs the same two-turn task:

1. Search the web and retain five sourced findings.
2. Use those findings to create a PDF and an XLSX file.

A successful run ends with `demonstration: PASS`. It leaves the PDF, XLSX,
communicator events, accounting evidence, and durable conversation under the
configured output and storage locations.

## Change the demo

Edit the ignored `config.local.yaml` to change the model, research topic,
limits, Redis, Postgres, storage, or output directory.

Edit the chosen example to change the task:

| Change | File |
| --- | --- |
| prompts and required output files | `<example>/agent.py` |
| native or LangGraph tools | `<example>/tools.py` |
| Claude tools and timeout | `claude/config.local.yaml` |

After changing the demo, run `--check`, then `--infra-check`, then the full
command. The individual README names the exact edit points.

Stop the local services from the `agents` directory:

```bash
docker compose --env-file .env down
```
