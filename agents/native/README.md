---
id: repo:kdcube-ai-app/agents/native/README.md
title: "Run the Native ReAct Agent"
summary: "Run KDCube native ReAct directly from Python with YAML-selected tools and skills."
tags: ["agents", "native-react", "harness", "standalone", "demonstration", "web-search"]
keywords: ["ReactSolverV2", "DirectAgentHarness", "KDCube Web Search", "Postgres conversation", "ChatCommunicator", "accounting"]
updated_at: 2026-09-05
see_also:
  - repo:kdcube-ai-app/agents/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md
---
# Run the Native ReAct Agent

## What it is

This directory runs KDCube's `ReactSolverV2` in your Python process.
`config.local.yaml` selects agent behavior and tool settings;
`descriptors.local/` selects the model and host services.

## Run it

```bash
cd agents/native
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
ignored by Git. `setup_local.py` is this one-time preparation command; it does
not participate in a turn or own agent configuration.

## What the demo shows

The agent searches through KDCube Web Search, creates a PDF and XLSX, then
opens another conversation and uses `react.memsearch` to recover its earlier
research. A successful run ends with `demonstration: PASS`.

## Change the demo

Edit `config.local.yaml` to change instructions, run directory, tools, skills,
topic, or limits. Web Search's allowlist, blocklist, and SSRF policy are under
the `demo.web_search` tool row's `settings`. Edit
`descriptors.local/assembly.yaml` for the model, storage, and isolated executor;
edit `descriptors.local/secrets.yaml` for credentials. The shipped model is
`claude-haiku-4-5-20251001`.

To add a local Python tool, implement it in `tools.py`, register its trusted
source in `agent.py`'s `TOOL_SOURCES`, and select its ID in
`config.local.yaml`. The SDK resolves that registry into Native tool bindings;
this example does not carry a separate planner module.

To use isolated Python execution, follow
[the executor step](../../app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md#8-enable-isolated-python-execution).
