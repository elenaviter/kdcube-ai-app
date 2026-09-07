---
id: repo:kdcube-ai-app/agents/README.md
title: "Add the KDCube Harness to Your Agent"
summary: "Keep a LangGraph or Claude Code agent core, or start with KDCube Native ReAct, and add durable conversations, tools, skills, isolated code execution, rendering, and usage evidence."
tags: ["agents", "harness", "native-react", "langgraph", "claude-code", "quickstart", "web-search"]
keywords: ["agent examples", "DirectAgentHarness", "KDCube Web Search", "Redis", "Postgres", "PDF", "XLSX"]
updated_at: 2026-09-07
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/quick-start-README.md
  - repo:kdcube-ai-app/mcp/web-search/README.md
---
# Add the KDCube Harness to Your Agent

This directory is a runnable proof of one specific path:

```text
your LangGraph graph or Claude Code agent ----+
                                               +--> DirectAgentHarness
KDCube Native ReAct, when you need an agent --+         |
                                                         +--> durable conversation
                                                         +--> selected tools + skills
                                                         +--> attachments + generated files
                                                         +--> isolated code workspace
                                                         +--> streamed and accounted events
```

Keep the agent core you already have. The example adds the KDCube boundary
around it. If you do not have an agent core, use the Native ReAct example. Each
example runs from your shell as a normal Python process and imports the KDCube
SDK from this checkout; it does not require a running KDCube server.

Choose the starting point that matches what you have:

| What you have | What to run | Start here |
| --- | --- | --- |
| No agent core yet | KDCube Native ReAct inside the harness | [native](native/README.md) |
| A LangGraph graph, or a graph you want to adapt | LangGraph with `KDCubeChatModel` and the harness | [langgraph](langgraph/README.md) |
| Claude Code | `ClaudeCodeAgent` inside the harness | [claude](claude/README.md) |

## Why use the harness?

The demonstration makes one agent do all of this in observable steps:

- continue a conversation from Postgres and search an earlier conversation;
- receive an attachment and retain generated files in configured storage;
- use YAML-selected tools, skills, instructions, and Web Search policy;
- author Python and run it in an isolated Docker workspace;
- turn HTML or Markdown into PDF, DOCX, and PPTX with reusable renderers;
- stream events while recording model usage and cost evidence; and
- leave an execution archive containing the exact code it ran.

The model and agent core still decide what to do. The harness supplies the
conversation, tool, workspace, file, streaming, and accountability contracts
around those decisions.

## How do I try it?

1. Choose Native ReAct, LangGraph, or Claude Code below.
2. Install that directory's `requirements.txt`.
3. Create its local YAML descriptors and select a model, tools, and skills.
4. Start its Redis and Postgres services and build the isolated executor.
5. Run `agent.py`, then inspect the conversation, files, events, usage, and
   archived generated code under `output/`.

Choose the model path independently where the adapter permits it:

| Agent | Model path | Start here |
| --- | --- | --- |
| Native ReAct | Provider API or on-host model through the KDCube model gateway | `descriptors.local/assembly.yaml` |
| LangGraph | Provider API or on-host model through the KDCube model gateway | `descriptors.local/assembly.yaml` |
| Claude Code | Claude Code's Anthropic model path | `descriptors.local/assembly.yaml` |

Each starting-point link opens a self-contained runner, requirements file,
agent YAML, platform descriptors, skill, Redis/Postgres Compose file, and exact
commands.

Every runner receives its caller and conversation explicitly:

```yaml
agent:
  input:
    user_id: demo-user
    user_type: regular
    session_id: local-session
    conversation_id: native-demo
```

Run the same example again with the same `user_id` and `conversation_id` to
continue that durable conversation. Use another `conversation_id` to start a
separate conversation. The shared conversation key is tenant, project, user,
and conversation; each adapter adds its stable `agent_id` to its private
checkpoint or transcript key. `session_id` identifies the calling session and
accounting lineage, rather than replacing the durable conversation key. The
values can also be overridden with `--user-id`, `--conversation-id`, and
`--session-id`.

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

The agent authors research, code, data, HTML, and Markdown. The isolated
executor runs its program, and KDCube's document tools own repeatable PDF,
DOCX, and PPTX conversion. This turns the selected core into a concrete
research-and-file agent while preserving the core's own loop.

Each YAML also selects an SDK-owned instruction profile. Native uses the
standard ReAct `lite:core` body plus blocks for its enabled tools. LangGraph and
Claude use the framework-neutral `workspace-files` body. Product behavior goes
in `additional_instructions`, after the workspace, capability, and skill
teaching. See
[Direct Agent Instruction Profiles](../app/ai-app/docs/runtime/harness/direct-agent-instruction-profiles-README.md).

Start with one agent link above. The complete shared command sequence is in
[Run the Agent Harness from Python](../app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md).
The standalone [Web Search MCP example](../mcp/web-search/README.md) is also
available.

## Serve the same agent to users

Keep the agent and harness adapter, place their composition in a KDCube app,
and declare a chat, API, or messaging surface. The hosted runtime supplies
authenticated user and conversation IDs, tool-execution enforcement, consent,
rate/spend policy, and multi-user ingress. Follow
[Settle Your Solution in KDCube](../app/ai-app/docs/recipes/apps/settle-your-solution-in-kdcube-README.md)
and the [KDCube Quick Start](../app/ai-app/docs/quick-start-README.md).
