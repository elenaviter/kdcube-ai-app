---
id: repo:kdcube-ai-app/agents/native/README.md
title: "Run the Native ReAct Agent"
summary: "Run KDCube native ReAct directly from Python with YAML-selected tools and skills."
tags: ["agents", "native-react", "harness", "standalone", "demonstration"]
keywords: ["ReactSolverV2", "DirectAgentHarness", "Postgres conversation", "ChatCommunicator", "accounting"]
updated_at: 2026-09-04
see_also:
  - repo:kdcube-ai-app/agents/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md
---
# Run the Native ReAct Agent

## What it is

This directory runs KDCube's `ReactSolverV2` in your Python process. Its YAML
selects instructions, tools, skills, limits, model, storage, and economics.

## Run it

```bash
cd agents/native
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python configure.py --provider openai
cp config.template.yaml config.local.yaml
docker compose --env-file .env -f compose.yaml up -d --wait
.venv/bin/python agent.py --check
.venv/bin/python agent.py --infra-check
.venv/bin/python agent.py
```

The provider key prompt is hidden. Local secrets and generated descriptors are
ignored by Git.

## What the demo shows

The agent searches the web, creates a PDF and XLSX, then opens another
conversation and uses `react.memsearch` to recover its earlier research. A
successful run ends with `demonstration: PASS`.

## Change the demo

Edit `config.local.yaml` to change instructions, tools, skills, topic, or
limits. Edit `descriptors.local/assembly.yaml` for the model, storage, and
isolated executor; edit `descriptors.local/secrets.yaml` for credentials.

To use isolated Python execution, follow
[the executor step](../../app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md#8-enable-isolated-python-execution).
