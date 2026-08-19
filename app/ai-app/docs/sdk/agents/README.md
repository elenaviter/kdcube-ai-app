---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/README.md
title: "Agents"
summary: "The map of the agent documentation: the three hosted families (ReAct, LangGraph, Claude Code), the two axes that place any agent (who runs and governs it, who supplies the harness), and where each family's own pages begin."
status: current
tags: ["sdk", "agents", "react", "langgraph", "claude-code", "overview"]
updated_at: 2026-08-19
keywords: ["hosted agent", "resident agent", "agent families", "harness", "run-to-completion", "MCP client", "ReAct", "LangGraph agent", "Claude Code agent", "external agent over MCP"]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/agent-in-the-runtimes-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/events/reactive-turn-delivery-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/apps/settle-your-solution-in-kdcube-README.md
---
# Agents

Two axes place every agent this documentation covers. Keeping them separate is
what keeps the vocabulary honest.

**Who runs and governs it.** A **hosted agent** is one the platform runs and
governs: the app runs its turns, binds its identity, keeps its conversation
record, and bills its calls. That covers the platform's own ReAct agent, a
LangGraph graph settled into an app, the Claude Code CLI, and equally an app
driving a remote agent service (the compute sits elsewhere; the app still
runs and governs the turns). KDCube's own word for a hosted agent is
**resident**: you settle an agent into an app, and it becomes a resident of
that app. The other side of this axis is the **external agent**, an MCP
client of the platform's servers: it keeps its own runtime and reaches
KDCube's governed capability over a delegated, revocable grant.

**Who supplies the harness.** ReAct runs on the platform's harness: the
platform drives the iteration and checks between rounds, so mid-turn events
fold straight into the loop. A LangGraph graph and the Claude Code CLI each
bring their own harness; the platform watches their conversation lane and
reaches the run in each harness's own terms: a cancellation at an await
point, a note or a refusal at the next tool call.
[Reactive turn delivery](../events/reactive-turn-delivery-README.md) carries
the per-lane table.

The two axes are orthogonal. A hosted agent runs on either harness, and the
harness axis decides exactly one thing: how a mid-turn event reaches the run.
An external agent brings its own harness by definition.

## The three hosted families

| Family | Harness | Start here |
| --- | --- | --- |
| **ReAct**, the platform's own agent, itself a coding agent | the platform's | [`react/`](react/): [structure](react/structure-README.md), [flow](react/flow-README.md), [context](react/react-context-README.md), [tools](react/react-tools-README.md) |
| **LangGraph**, your graph, settled into an app unchanged | its own | [`langgraph/`](langgraph/langgraph-agent-README.md): the model bound by role, the seam that supplies identity, tools and events, what a stop obliges |
| **Claude Code**, the hosted CLI | its own | [`claude/`](claude/claude-code-README.md): invocation, streaming, sessions, [workspace bootstrap](claude/claude-code-workspace-bootstrap-README.md), [accounting](claude/claude-code-accounting-README.md) |

What every hosted family inherits from the app around it: ordered turns,
the conversation record and reload, per-user isolation, files and code
execution, capability declarations the chat composer reads
(`accepts_steer` / `accepts_followup`), and economics.

## The external agent

An agent outside KDCube acts as an MCP client of the platform's servers,
with a delegated, revocable grant: the platform's services, its own runtime.
That lane is documented with the connection machinery, starting at
[Platform MCP over Connection Hub](../solutions/mcp/platform-mcp-over-connection-hub-README.md).

## Cross-cutting pages in this directory

- [Prompt exfiltration: internal and direct agents](README-prompt-exfiltration-internal-and-direct-agents.md)
