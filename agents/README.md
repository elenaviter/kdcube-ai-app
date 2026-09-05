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

Pick the agent core you already use:

| Agent | Start here |
| --- | --- |
| Native ReAct | [native](native/README.md) |
| LangGraph | [langgraph](langgraph/README.md) |
| Claude Code | [claude](claude/README.md) |

Each directory runs directly from this repository in a shell or IDE. No
KDCube server is required. Redis and Postgres run as independent support
services.

The built-in two-turn demonstration does this:

```text
research request
      |
      v
KDCube Web Search -> retained conversation context
      |
      v
agent authors Python -> isolated executor -> XLSX + HTML
                                             |
                                             v
                                rendering_tools.write_pdf
                                             |
                                             v
                                      polished PDF
```

The YAML-selected renderer family also exposes HTML-to-PPTX and
Markdown-to-DOCX. Each run records communicator events, accounted model calls,
attachments, output files, conversation turns, and the execution ZIP that
contains the model-authored `pkg/user_code.py`.

The division of work is deliberate: the agent authors research, code, data,
HTML, and Markdown; the isolated executor runs its program; KDCube's document
tools own repeatable PDF, DOCX, and PPTX conversion. An existing agent therefore
gains deep-research and file-production capability without regenerating a new
PDF or Office implementation for every request.

Start with the complete command sequence in
[Run the Agent Harness from Python](../app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md).
The standalone [Web Search MCP example](../mcp/web-search/README.md) is also
available.

For a ready runtime with chat UI, authentication, managed tools,
tool-execution enforcement, isolated workspaces, and app hosting, use
[KDCube Quick Start](../app/ai-app/docs/quick-start-README.md).
