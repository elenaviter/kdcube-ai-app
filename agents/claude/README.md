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
stores its resumable CLI transcript on a per-conversation Git branch.

## Run it

Install and authenticate Claude Code, then run:

```bash
cd agents/claude
claude --version
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python configure.py --provider none
cp config.template.yaml config.local.yaml
docker compose --env-file .env -f compose.yaml up -d --wait
.venv/bin/python agent.py --check
.venv/bin/python agent.py --infra-check
.venv/bin/python agent.py
```

For API-key execution, use `--provider anthropic` and set
`services.anthropic.claude_code_key` in the generated secrets descriptor.

## What the demo shows

Turn one searches through the local KDCube Web Search MCP and saves
`research.json`. The generated Claude workspace explicitly denies Claude's
ambient `WebSearch` and `WebFetch`, so this path exercises the configured
KDCube tool. Turn two resumes the same Claude session and creates a PDF and
XLSX. The SDK publishes the transcript to Git, while the harness independently
records both turns and their accounting. A successful run ends with
`demonstration: PASS`.

## Change the demo

Edit `config.local.yaml` to change Claude's model, instructions, tools, skills,
task, or timeout. Edit `descriptors.local/assembly.yaml` to select a private Git
transcript remote, and put its HTTPS token in
`descriptors.local/secrets.yaml` at `services.git.http_token`.
Edit `web-search.yaml` to change the search domain allowlist, blocklist, or
SSRF policy. The SDK writes the selected stdio server and exact MCP tool
permissions into Claude's workspace at run time.
