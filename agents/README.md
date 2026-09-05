---
id: repo:kdcube-ai-app/agents/README.md
title: "Run an Agent with the KDCube Harness"
summary: "Choose native ReAct, LangGraph, or Claude Code and run it directly from Python."
tags: ["agents", "harness", "native-react", "langgraph", "claude-code", "quickstart", "web-search"]
keywords: ["agent examples", "DirectAgentHarness", "KDCube Web Search", "Redis", "Postgres", "PDF", "XLSX"]
updated_at: 2026-09-05
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/quick-start-README.md
  - repo:kdcube-ai-app/mcp/web-search/README.md
---
# Run an Agent with the KDCube Harness

Choose one runnable agent:

| Agent | Start here |
| --- | --- |
| Native ReAct | [native](native/README.md) |
| LangGraph | [langgraph](langgraph/README.md) |
| Claude Code | [claude](claude/README.md) |

Each directory is complete: agent code, YAML configuration, standard KDCube
descriptors, requirements, a skill, and Redis/Postgres Compose services.

These examples run directly from Python without a KDCube server. They show web
research, multiple turns, harness events and accounting, durable conversation
state, and PDF/XLSX output.

All three use the repository's KDCube Web Search implementation. Native ReAct
and LangGraph call it through small framework adapters; Claude Code receives
it as an on-demand local stdio MCP server. Claude's ambient `WebSearch` and
`WebFetch` tools are denied in this example, so a successful search proves the
configured KDCube tool was used. Its row in each agent's `agent.tools` list
owns the domain allowlist, blocklist, and SSRF policy under `settings`.

There are two configuration owners. `config.local.yaml` selects agent behavior,
the run directory, tools, skills, and settings attached to each tool row.
`descriptors.local/` selects shared platform services: model, Redis/Postgres,
storage, secrets, economics, and the optional isolated executor.
`setup_local.py` is the one-time command that prepares the ignored local
descriptors and matching Compose credentials. Every example defaults to
`claude-haiku-4-5-20251001`.

The same tool can be run independently for another MCP client. See the
[Web Search MCP quick start](../mcp/web-search/README.md).

For the full copy-and-run procedure, including optional isolated code
execution, use
[Run the Agent Harness from Python](../app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md).

For a ready runtime with chat UI, authentication, managed tools,
tool-execution enforcement, isolated workspaces, and app hosting, use
[KDCube Quick Start](../app/ai-app/docs/quick-start-README.md).
