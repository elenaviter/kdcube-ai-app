---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/langgraph/langgraph-agent-README.md
title: "LangGraph Agent"
summary: "The SDK surface a LangGraph or LangChain agent binds to in KDCube: the chat model that routes and bills through the platform, the foreign-runtime seam that supplies identity, tools, events and recording to a loop the platform does not own, and the obligations that come with being stoppable."
status: current
tags: ["sdk", "agents", "langgraph", "langchain", "foreign-runtime", "run-to-completion"]
updated_at: 2026-08-18
keywords: ["KDCubeChatModel", "foreign runtime", "AgentSpec", "fold_turn_external_events", "LiveLaneWatch", "run_until_stopped", "checkpointer", "create_agent", "run-to-completion", "turn identity", "named services door", "MCP bridge", "stop repair", "dangling tool call"]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/apps/settle-your-solution-in-kdcube-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/agent-in-the-runtimes-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/events/reactive-turn-delivery-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/claude/claude-code-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/conversation/hosted-agent-conversation-README.md
---
# LangGraph Agent

Your graph keeps its own loop. KDCube does not rewrite it, wrap its nodes, or
ask it to yield — it supplies what the graph cannot get for itself: a model that
routes and bills through the platform, this turn's identity and inputs, tools
the operator has consented to, and a record the conversation can be reloaded
from.

This page documents that surface. For how to port a solution step by step, read
[Settle your solution in KDCube](../../../recipes/apps/settle-your-solution-in-kdcube-README.md);
for where a lane of this kind sits among the runtimes, read
[Agent in the runtimes](../../../runtime/agent-in-the-runtimes-README.md). The
worked reference is the `ported-langgraph-agents@2026-07-13` bundle, whose own
README explains its two-agent layout.

## The model: `KDCubeChatModel`

`sdk/frameworks/langchain/chat_model.py` is a LangChain `BaseChatModel`. Give it
the models service and a **role**, and it behaves like any other streaming chat
model — `bind_tools`, `astream`, tool-call chunks:

```python
from kdcube_ai_app.apps.chat.sdk.frameworks.langchain import KDCubeChatModel

model = KDCubeChatModel(models_service=ms, role="lg-react.answer")
```

The role is the point. Your graph names a role, never a model id; the platform
resolves it per deployment, and a per-turn override from the capabilities widget
overlays it for that turn only. Accounting, provider selection and streaming all
follow from the same binding, so a graph written against this model needs no
KDCube-specific code anywhere else in its node functions.

It also reports what the platform did to a response: a generation that spends
its whole output budget is logged as INTERRUPTED with the token count rather
than returned as a mysteriously short answer.

## The seam: `sdk/solutions/foreign_runtime/`

Everything a lane needs that its own framework has no concept of. Import from
the package root; the modules are an implementation detail.

| Concern | What it gives you |
| --- | --- |
| `AgentSpec`, `resolve_agent_spec`, `dispatch` | one app, many agents: build/stream/inputs per agent id |
| `turn_identity`, `TurnIdentity` | platform identity mapped onto your per-user, per-conversation keys |
| `fold_turn_external_events` | this turn's input: the whole pending lane, lane-ordered, read-only |
| `resolve_turn_role_models`, `resolve_turn_model_pick` | the user's model choice for this turn |
| `resolve_turn_mcp`, `narrow_mcp_connections`, `connect_required_outcome` | consented MCP servers, and the in-band answer when a connection is missing |
| `named_service_door_*`, `named_service_inventory` | the named-services door and its roster block |
| `pull_into_workspace`, `WORKSPACE_PULL_TOOL` | the turn workspace, reachable by reference |
| `finalize_conversation_title`, `emit_turn_timing`, `persist_turn_artifacts`, `conversation_is_new` | the conversation record a reload is rebuilt from |
| `LiveLaneWatch`, `run_until_stopped` | reaching and stopping a run while it streams |

Two properties are worth stating because they are easy to lose when adapting
this to another framework:

- **The fold is read-only.** It never consumes or reserves lane events, so a
  turn that dies leaves the lane exactly as it found it and the ordinary handoff
  still runs.
- **The turn's input is everything pending**, not the waking event: attachments
  and context chips that arrived with the prompt, and messages typed while the
  previous turn was still running, arrive together as one turn.

## Rebuild the graph every turn

The bound graph is constructed per turn and never cached on the entrypoint.
KDCube is distributed — consecutive turns of one conversation may run in
different workers or on different machines — so a graph cached in a long-lived
process object is at best useless and at worst stale. State lives in the
checkpointer, which is where it can be shared; the graph is cheap to rebuild
and must be, for a per-turn model override or a per-turn tool set to mean
anything.

## Being stoppable, and what that costs

A steer cancels the streaming task (`run_until_stopped`). The platform never
folds events into your graph's iteration — that iteration is the graph's — it
watches the lane read-only and cancels at an await point, so everything the
watcher saw is still pending afterwards and the next turn folds it in. The stop
control and mid-turn follow-ups are offered per agent through
`capabilities.conversation.accepts_steer` / `accepts_followup`; declare them, or
the composer will not draw the controls.

**A cancelled run leaves an obligation.** The checkpointer holds the last
completed node, but a cancellation can land INSIDE one exchange: the model's
tool call is checkpointed and its result never written. The graph is untroubled
by that; the provider is not, and refuses the whole history on the next request:

```
400 invalid_request_error: 'tool_use' ids were found without 'tool_result'
blocks immediately after: toolu_…
```

Because the same history replays every turn, one stop then costs the whole
conversation rather than the turn. A lane that can be interrupted must answer
the unanswered call — truthfully, with a tool result saying the tool did not
run, so the model reads what happened instead of inventing a result. The
reference implementation is `platform/stop_repair.py` in the ported bundle; it
runs after a stop and before every turn, so a thread wedged by anything (a
timeout kill, a dead worker) heals on its next message.

Two details in it are not optional. The repair must answer **every** unanswered
id, since one is enough to be refused; and the write must be attributed to the
tools node (`aupdate_state(..., as_node="tools")`), because an unattributed
update resumes as the interrupted node and re-evaluates the model→tools edge,
which returns a destination that branch does not declare when middleware is
present (`KeyError: 'SummarizationMiddleware.before_model'`).

This is the general shape for any runtime whose loop KDCube does not own: it can
be stopped without being seized, and whatever the interruption leaves half-done
is the lane's to make whole. See
[Reactive turn delivery](../../events/reactive-turn-delivery-README.md) for the
per-lane table.

## Storage

Agents that share a bundle share one schema and are separated by a column, not
by a database. The graph's checkpointer, its memories, and any tables it owns
are resolved per deployment from the descriptor, and per agent from the turn
identity — so two agents in one app never read each other's state, and the same
agent in two projects never shares one.

## Where to read further

- The reference bundle: `sdk/examples/bundles/ported-langgraph-agents@2026-07-13/README.md`
  — two agents behind one `execute_core`, its storage layout, code execution,
  degradation, and the generic porting procedure.
- The Claude Code lane, for contrast: [Claude Code Agent](../claude/claude-code-README.md)
  — the same seam, with a CLI's loop instead of a graph's, where the stop is a
  tool-call boundary rather than a cancellation.
- [Hosted agent conversation](../../solutions/conversation/hosted-agent-conversation-README.md)
  — what the conversation record needs from a lane with no ReAct timeline.
