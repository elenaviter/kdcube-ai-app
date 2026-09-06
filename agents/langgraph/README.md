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
stores both the conversation index and LangGraph checkpoints. No KDCube server
is required.

## Run it

```bash
cd agents/langgraph
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m playwright install chromium
.venv/bin/python setup_local.py --provider anthropic
cp config.template.yaml config.local.yaml
docker compose --env-file .env -f compose.yaml up -d --wait
cd ../..
docker build -t py-code-exec:latest -f app/ai-app/deployment/docker/all_in_one_kdcube/Dockerfile_Exec app/ai-app
cd agents/langgraph
.venv/bin/python agent.py --check
.venv/bin/python agent.py --infra-check
.venv/bin/python agent.py
```

The provider key prompt is hidden. Local secrets and generated descriptors are
ignored by Git.

## What the demo shows

Turn one hosts a research-request attachment and searches through KDCube Web
Search. Turn two resumes the same Postgres-backed graph thread. The model
authors Python using `openpyxl`; `execute_python` runs it in the configured
Docker image to create an XLSX and HTML. `write_pdf` then renders the HTML into
a polished PDF.

The same YAML enables `write_docx` for Markdown and `write_pptx` for
section-based HTML. A successful run proves live events, accounted model calls,
durable KDCube turns, Postgres graph checkpoints, isolated code execution, and
document rendering; it ends with `demonstration: PASS`.

Inspect `output/runs/<conversation>/evidence.json`; it points to durable
records in `output/kdcube-storage`, including the execution ZIP containing
`pkg/user_code.py`.

## Change the demo

Edit `config.local.yaml` to change the `workspace-files` instruction profile,
`additional_instructions`, run directory, tools, skills, topic, or limits. The
profile teaches the current-turn artifact workspace; selected skill text and
enabled capability guidance are composed before the administrator override.
Web Search's allowlist, blocklist, and SSRF policy are under the `web_search`
tool row's `settings`. Edit
`descriptors.local/assembly.yaml` for model and infrastructure; edit
`descriptors.local/secrets.yaml` for credentials. The shipped model is
`claude-haiku-4-5-20251001`. Add LangChain tools in `tools.py` and select their
IDs in YAML. The built-in execution and renderer wrappers are turn-bound, so
new file-producing tools should preserve the same current-turn path contract.
The exact composition is documented in
[Direct Agent Instruction Profiles](../../app/ai-app/docs/runtime/harness/direct-agent-instruction-profiles-README.md).
