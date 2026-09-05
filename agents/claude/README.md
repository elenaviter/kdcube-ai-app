---
id: repo:kdcube-ai-app/agents/claude/README.md
title: "Run the Claude Code Agent"
summary: "Run Claude Code through the KDCube harness with a Git-backed transcript and durable conversation."
tags: ["agents", "claude-code", "harness", "conversation", "standalone", "web-search", "mcp"]
keywords: ["ClaudeCodeAgent", "KDCube Web Search MCP", "Claude transcript", "Git session store", "ChatCommunicator", "accounting"]
updated_at: 2026-09-05
see_also:
  - repo:kdcube-ai-app/agents/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md
---
# Run the Claude Code Agent

## What it is

This directory runs the local Claude Code CLI through `ClaudeCodeAgent`. The
harness stores its conversation in Postgres and configured storage; the SDK
stores its resumable CLI transcript on a per-conversation Git branch. No
KDCube server is required.

## Run it

Install and authenticate Claude Code, then run:

```bash
cd agents/claude
claude --version
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m playwright install chromium
.venv/bin/python setup_local.py --provider none
cp config.template.yaml config.local.yaml
docker compose --env-file .env -f compose.yaml up -d --wait
cd ../..
docker build -t py-code-exec:latest -f app/ai-app/deployment/docker/all_in_one_kdcube/Dockerfile_Exec app/ai-app
cd agents/claude
.venv/bin/python agent.py --check
.venv/bin/python agent.py --infra-check
.venv/bin/python agent.py
```

For API-key execution, use `--provider anthropic` and set
`services.anthropic.claude_code_key` in the generated secrets descriptor.

## What the demo shows

Turn one reads a hosted research-request attachment and searches through the
local KDCube Web Search MCP. The generated Claude workspace denies ambient
`WebSearch`, `WebFetch`, and `Bash`, so search and code execution use the
configured KDCube boundaries.

Turn two resumes the same Git-backed Claude session. Claude authors Python
using `openpyxl`; the `kdcube_harness` MCP runs it in the configured Docker
image to create an XLSX and HTML. A second MCP operation calls
`rendering_tools.write_pdf` to render the HTML into a polished PDF. The same
server exposes Markdown-to-DOCX and section-HTML-to-PPTX operations.

The harness independently records both turns and accounting in KDCube storage.
Inspect `output/runs/<conversation>/evidence.json`; it points to
`output/kdcube-storage`, the execution ZIP containing `pkg/user_code.py`, and
the Claude transcript branch. A successful run ends with `demonstration: PASS`.

## Change the demo

Edit `config.local.yaml` to change Claude's adapter model, instructions, run
directory, tools, skills, task, or timeout. Web Search's egress policy is under
the `mcp__kdcube_web_search__web_search` tool row's `settings`. Edit
`descriptors.local/assembly.yaml` to select a private Git
transcript remote, and put its HTTPS token in
`descriptors.local/secrets.yaml` at `services.git.http_token`.
The SDK writes the selected stdio server and exact MCP tool permissions into
Claude's workspace at run time. Both the adapter and shared descriptor template
ship with `claude-haiku-4-5-20251001` selected. Tool and skill selection stays
in `config.local.yaml`; executor, storage, and Git settings stay in the
standard descriptors.
