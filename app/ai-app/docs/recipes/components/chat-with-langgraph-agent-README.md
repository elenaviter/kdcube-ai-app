---
id: repo:kdcube-ai-app/app/ai-app/docs/recipes/components/chat-with-langgraph-agent-README.md
title: "Recipe: Chat With A LangGraph Agent"
summary: "End-to-end steps for preserving a LangGraph agent loop while binding it to KDCube identity, model routing and accounting, governed tools, durable conversation state, streaming, and interruption handling."
status: current
tags: ["recipes", "components", "chat", "langgraph", "langchain", "agent-harness"]
updated_at: 2026-09-04
keywords: ["chat with langgraph agent", "KDCubeChatModel", "create_agent", "astream_events", "LangGraph checkpointer", "automatic accounting", "foreign runtime", "run_python"]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/langgraph/langgraph-agent-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/components/chat-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/apps/settle-your-solution-in-kdcube-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/README.md
  - repo:kdcube-ai-app/agents/langgraph/README.md
---
# Recipe: Chat With A LangGraph Agent

Use this recipe when an app already has a LangGraph graph, or when LangChain's
`create_agent` is the right loop, and the graph should answer through a KDCube
conversation. Keep the graph's nodes and transitions framework-native. Add a
thin host adapter around the graph for identity, durable state, model routing,
tools, communicator events, live control, and conversation recording.

The maintained hosted implementation is
`ported-langgraph-agents@2026-07-13`. It hosts two different graph shapes behind
one entrypoint and gives each shape its own stream adapter. The direct,
deployment-free SDK demonstration is
[`agents/langgraph`](../../../../../agents/langgraph/README.md).

```text
chat / webhook / supported submitter
              |
              v
BaseEntrypointWithEconomics
  bind caller + accounting + communicator
              |
              v
thin LangGraph host adapter
  fold pending events
  resolve agent/user/conversation identity
  select model and tools for this turn
  build graph + bind durable checkpointer
  map graph events to communicator events
  record the framework-neutral turn
              |
              v
preserved LangGraph loop
```

## 1. Declare the chat and agent

The app declares product intent in `bundles.yaml`. `default_chat` exposes the
conversation surface. The agent block declares the model role, user-visible
model choices, conversation controls, and tool ceiling.

```yaml
surfaces:
  as_provider:
    bundle:
      default_chat: true
  as_consumer:
    default_agent: lg-react
    agents:
      lg-react:
        capability_provider: simple_model_pick
        capabilities:
          models:
            role: lg-react.answer
            default: claude-sonnet-4-6
            supported:
              - provider: anthropic
                model: claude-sonnet-4-6
                label: Sonnet 4.6
          conversation:
            accepts_steer: true
            accepts_followup: false
        tools:
          - name: code_exec
            kind: python
            alias: code_exec
            allowed: [run_python]
            code_exec:
              timeout_s: 180
          - name: web
            kind: python
            alias: web
            allowed: [web_search, web_fetch]

role_models:
  lg-react.answer:
    provider: anthropic
    model: claude-sonnet-4-6
```

The administrator's descriptor is the ceiling. The conversation capabilities
selection may narrow the model and tools for a user. It cannot add a tool the
app did not declare. Connected-account and MCP permissions are checked at the
operation boundary and remain default-closed.

Declare only controls the adapter implements. The maintained LangGraph lane
cancels on steer and leaves mid-turn text pending for the next fold, so it
declares steer and not live follow-up ingestion.

## 2. Keep the solution behind a small interface

The host needs a construction seam, not ownership of graph logic. A practical
registry keeps each agent's graph builder, input builder, answer role, and
stream adapter together:

```python
spec = AgentSpec(
    agent_id="lg-react",
    role="lg-react.answer",
    build_graph=build_agent,
    build_inputs=build_inputs,
    stream=stream_react_turn,
)
```

One app can host multiple graphs. Resolve `agent_id` once per turn and choose a
spec. Do not put graph-shape branches inside one generic stream loop. A graph
with a dedicated answer node and a looping `create_agent` graph produce
different event sequences and need different adapters.

Bundle code imports its own packages relatively. SDK imports remain absolute.

## 3. Rebuild the bound graph per turn

Build a fresh graph for every turn after resolving the current model and tool
selection. A graph instance can close over request identity, selected tools,
and model routes. Caching that instance on an entrypoint leaks stale turn state
and cannot provide continuity when the next turn lands on another worker.

Durable continuity belongs in the checkpointer. Scope its thread key with:

```text
tenant + project + app agent + user + conversation
```

Reuse infrastructure connections where safe. Rebuild request-bound graph
objects. Keep app-owned tables in the tenant/project schema with explicit app,
agent, and user columns rather than creating one database per agent.

## 4. Route model calls through `KDCubeChatModel`

Bind the graph to the platform model adapter:

```python
from kdcube_ai_app.apps.chat.sdk.frameworks.langchain import KDCubeChatModel

model = KDCubeChatModel(
    models_service=entrypoint.models_service,
    role="lg-react.answer",
)
agent = create_agent(model=model, tools=tools, checkpointer=checkpointer)
```

This keeps the graph's ordinary LangChain interface. `bind_tools`, `ainvoke`,
`astream`, and `astream_events` continue to work.

Accounting is automatic for model calls through this adapter. Its
`_astream()` creates the producer inside the bound turn accounting context and
invokes `ModelServiceBase.stream_model_text_tracked()`. LangGraph consumes
normal `ChatGenerationChunk` values while the platform records provider usage
against the active user, app, conversation, and turn. A graph that constructs
and calls a provider client directly is outside this guarantee.

## 5. Bind selected tools, including web and isolated execution

Resolve tools after applying the descriptor ceiling and the conversation's
saved narrowing. Wrap platform services as ordinary LangChain tools so the
solution does not learn transport internals.

The maintained `lg-react` example includes:

- `web_search` and `web_fetch`, routed through the configured web service;
- `run_python`, executed in KDCube's configured isolated execution runtime;
- conversation workspace pull/read helpers;
- optional delegated MCP connections; and
- the named-services door plus a descriptor-declared namespace roster.

When `run_python` hosts output files, it emits normal `chat.files` records with
durable `conv:fi:` references. Nested calls from generated code are available
only when the adapter exports a narrowed execution-tool catalog. Isolated
computation and nested tool access are separate capabilities.

For a local Docker execution profile, build the configured image from the
KDCube repository root when it is absent:

```bash
docker build -t py-code-exec:latest \
  -f app/ai-app/deployment/docker/all_in_one_kdcube/Dockerfile_Exec app/ai-app
```

## 6. Fold input, stream output, and record the turn

At turn start, call the shared foreign-runtime fold. It reads the complete
pending lane in order, including attachment events and messages queued while a
previous turn was active. The fold is read-only, so a failed run leaves
undelivered events recoverable.

Map the graph's `astream_events(version="v2")` output to the communicator:

- final-answer tokens become indexed `chat.delta` events;
- tool calls and results become `chat.step` activity;
- produced files use the platform file-hosting path; and
- title, timing, selected recordable events, and the final answer enter the
  framework-neutral turn record.

Derive the app from `BaseEntrypointWithEconomics`. Its turn boundary provides
admission and accounting settlement, and its fallback recorder writes the
minimal conversation record when a ReAct timeline did not. In the app's
post-run hook, call `persist_turn_artifacts()` so cost, timing, steps, and stream
aggregates survive reload.

## 7. Make interruption truthful

The maintained adapter runs the graph under `run_until_stopped()`. A steer
cancels the graph's stream at an await point. Events observed by the read-only
watch remain pending unless the adapter can prove they were delivered.

A cancellation can leave a model tool call in the checkpointer without a
matching tool result. Repair every such call before the next provider request
with a truthful result saying execution was interrupted. Otherwise the
provider rejects the replayed history and one stop wedges the conversation.
The detailed repair contract is owned by
[LangGraph Agent](../../sdk/agents/langgraph/langgraph-agent-README.md).

## Run the direct proof

The repository-level direct launcher supplies `create_agent` and its tools to
the SDK's `DirectAgentHarness`. The facade binds `ChatCommunicator`, accounting,
and the framework-neutral conversation record. It performs two real model
turns, then requires a PDF and XLSX from the retained research. Start with the
short support-service procedure in the [direct examples README](../../../../../agents/README.md):

```bash
cd agents/langgraph
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python agent.py --check
export OPENAI_API_KEY='...'
.venv/bin/python agent.py
```

The independent Redis service mirrors per-turn accounting events. Every direct
runner uses the KDCube conversation tables in Postgres plus configured payload
storage. `AsyncPostgresSaver.setup()` additionally creates or migrates the
private LangGraph checkpoint tables before the graph is compiled. In a fresh
database those are
`checkpoint_migrations`, `checkpoints`, `checkpoint_blobs`, and
`checkpoint_writes`. The runner exits non-zero when infrastructure bootstrap
fails, a completed turn has no Redis accounting evidence, either durable turn
cannot be materialized, or either output file is absent.

## Verification checklist

- The graph is built after model/tool selection for every turn.
- The checkpointer key includes agent, user, and conversation identity.
- The graph uses `KDCubeChatModel` for every model call claimed as accounted.
- Disabling Web or Code Exec removes those tools from the next turn.
- A web request emits tool activity, not only answer prose.
- A generated file is downloadable after reload through its `conv:fi:` ref.
- A steer stops the active stream, and the next turn passes after tool-call
  repair.
- Two turns continue one graph thread and one KDCube conversation.

Read [LangGraph Agent](../../sdk/agents/langgraph/langgraph-agent-README.md) for
the SDK contract and the maintained bundle's own README for its complete
two-agent implementation.
