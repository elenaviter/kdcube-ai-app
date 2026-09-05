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
`descriptors.local/` selects the model, storage, and support services. No
KDCube server is required.

## Run it

```bash
cd agents/native
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m playwright install chromium
.venv/bin/python setup_local.py --provider anthropic
cp config.template.yaml config.local.yaml
docker compose --env-file .env -f compose.yaml up -d --wait
cd ../..
docker build -t py-code-exec:latest -f app/ai-app/deployment/docker/all_in_one_kdcube/Dockerfile_Exec app/ai-app
cd agents/native
.venv/bin/python agent.py --check
.venv/bin/python agent.py --infra-check
.venv/bin/python agent.py
```

The provider key prompt is hidden. Local secrets and generated descriptors are
ignored by Git. `setup_local.py` is this one-time preparation command; it does
not participate in a turn or own agent configuration.

## What the demo shows

Turn one hosts a research-request attachment and searches through KDCube Web
Search. Turn two makes the model author Python that uses `openpyxl`; KDCube
executes that code in the configured Docker image to create an XLSX and HTML.
The agent then calls `rendering_tools.write_pdf` to turn the HTML into a
polished PDF. It does not generate PDF bytes in Python.

The YAML also enables `write_docx` for Markdown and `write_pptx` for
section-based HTML. The default demonstration calls only `write_pdf`; alter the
second prompt to exercise either sibling operation.

Native adds a third turn in another conversation and uses `react.memsearch` to
recover its earlier research. A successful run ends with `demonstration: PASS`.
Inspect `output/runs/<conversation>/evidence.json`; it points to durable
records in `output/kdcube-storage`, including the execution ZIP containing
`pkg/user_code.py`.

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

Executor image, timeout, network, and workspace limits live under
`platform.services.proc.exec` in `descriptors.local/assembly.yaml`. The full
storage and inspection procedure is in
[the executable recipe](../../app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md).
