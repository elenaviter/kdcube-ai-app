---
id: repo:kdcube-ai-app/agents/README.md
title: "Run an Agent with the KDCube Harness"
summary: "Choose native ReAct, LangGraph, or Claude Code and run it directly from Python."
tags: ["agents", "harness", "native-react", "langgraph", "claude-code", "quickstart"]
keywords: ["agent examples", "DirectAgentHarness", "Redis", "Postgres", "PDF", "XLSX"]
updated_at: 2026-09-04
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/quick-start-README.md
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

For the full copy-and-run procedure, including optional isolated code
execution, use
[Run the Agent Harness from Python](../app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md).

For a ready runtime with chat UI, authentication, managed tools,
tool-execution enforcement, isolated workspaces, and app hosting, use
[KDCube Quick Start](../app/ai-app/docs/quick-start-README.md).
