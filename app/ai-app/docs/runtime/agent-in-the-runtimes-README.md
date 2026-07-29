---
id: repo:kdcube-ai-app/app/ai-app/docs/runtime/agent-in-the-runtimes-README.md
title: "The Agent In The Runtimes Fusion"
summary: "The agent's-eye view of KDCube: how one agent is fed from any surface through the conversation event lane, the two layers of memory/state and who owns each (the platform timeline vs the agent's working memory), namespace-owned block production and presentation, what the fusion gives an agent, and the detailed difference between the native ReAct agent and an integrated framework — grounded in the two worked apps."
tags: ["runtime", "agent", "timeline", "event-bus", "memory", "state", "react", "langgraph", "hosted-agents", "fusion"]
keywords:
  [
    "agent in the runtimes",
    "conversation event lane",
    "shared timeline event bus",
    "two memories two owners",
    "timeline source of truth",
    "working memory",
    "session view",
    "compaction",
    "everything is a URI",
    "native agent",
    "integrated agent",
    "event sources",
    "block production",
    "rehosting",
    "turn lifecycle",
  ]
updated_at: 2026-07-29
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/runtimes-map-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/conversation/hosted-agent-conversation-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/events/reactive-turn-delivery-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/react/shared-timeline-event-bus-steer-followup-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/react/session-view-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/react/compaction-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/react/memory-recovery-path-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/timeline/turn-log-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/timeline/provider-projection-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/workspace/workspace-model-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/conversation/search-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/timeline/fork-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/claude/claude-code-README.md
---
# The Agent In The Runtimes Fusion

[The KDCube Runtimes: Map And Model](runtimes-map-README.md) describes the
runtimes as a system. This page turns the map around and looks from inside
the agent: how ONE agent — the native KDCube ReAct or an integrated
framework — is fed from any surface, what state exists for it and who owns
each layer, and what the fusion of runtimes concretely gives it.

Two worked apps ground everything here:

- **`workspace@2026-03-31-13-36`** — the native path. Its `main` agent is the
  KDCube ReAct adapter, constructed fresh per turn from the app descriptor.
- **`ported-langgraph-agents@2026-07-13`** — the integrated path. Two
  LangGraph agents behind one `execute_core`: a dispatcher resolves the
  agent, builds a fresh graph bound to the turn's identity, and routes the
  graph's own working store (its checkpointer) onto KDCube's Postgres.

## 1. Any Surface, One Lane

An agent never binds to a transport. Every way work can arrive — the chat
widget over SSE/Socket.IO, an authenticated REST operation, a public webhook
(Telegram included), an MCP call, a Data Bus message, a scheduled automation
— normalizes into the same `ExternalEventPayload`. From there the serving
runtime splits two ways (the [runtimes map](runtimes-map-README.md), §3): an
app's direct surfaces — `@api`, `@mcp`, widgets — answer in place,
request/response, and any of them can *submit* conversational work; work
**for the agent** lands on the **conversation event lane**: the ordered work
lane for one conversation. The lane reserves one accepted start batch for one
turn. That batch may contain same-ingress siblings, such as a prompt and its
attachments. Same-conversation turns serialize across workers while different
conversations run in parallel
([Reactive Turn Delivery](../sdk/events/reactive-turn-delivery-README.md)).

What makes the lane an *agent* concept rather than plumbing is what happens
when events arrive **while a turn is running**. Ingress appends busy-turn
events — `followup`, `steer`, and other supported external events — into one
shared, Redis-backed conversation event source. The active turn holds a
fenced owner lease on that source and listens live:

- a **follow-up** folds into the current turn: it lands on the turn's own
  timeline as a real block, can trigger another decision round before
  completion, and mints extra iteration credit for the turn;
- a **steer** folds as a control interrupt: it cancels the active generation
  or cancellable tool phase where possible (in isolated execution that
  becomes a container/subprocess kill), then the agent re-enters with the
  steer already on its timeline for a bounded finalize;
- an eligible continuation event no live owner consumes remains in the lane
  and becomes schedulable after the current turn releases;
- an unconsumed `event.user.steer` **expires**. A steer controls only the turn
  that was active when it arrived and is never promoted as future work.

```text
   followup / steer / external event, arriving MID-TURN
                        |
                        v
      shared conversation event source (Redis-backed)
                        |
              classify + check live ownership
                 |              |              |
                 v              v              v
   FOLD into the running   QUEUE eligible     EXPIRE unconsumed
   turn when consumed      continuation       event.user.steer
   followup -> +round      for a later turn   (never future work)
   steer -> interrupt ->
            bounded finalize
```

Eligible domain and subagent-completion events use the same fold-or-queue
path on the parent's lane. The steer exception is intentional. Contract and
event shapes:
[Shared Timeline Event Bus, Steer, Follow-up](../sdk/agents/react/shared-timeline-event-bus-steer-followup-README.md).

The two worked apps split exactly along this line. The native `workspace`
agent opens the live lane handler and folds mid-turn events at decision
boundaries. The LangGraph app takes the safe default: it folds the turn's
**ingress batch** before reading inputs (`platform/turn_batch.py` —
attachments ride sibling lane events of the prompt, so a run-to-completion
door must fold the batch or the agent sees the prompt alone), runs the graph
start to finish, and lets eligible mid-turn messages wait for the next turn.
It does not consume live steer; an unconsumed steer expires.

## 2. Two Layers Of Memory/State, Two Owners

State for a hosted agent is two distinct layers with two distinct owners.
Both must reflect the same conversation; neither substitutes for the other.
([The Conversation For Any Agent](../sdk/solutions/conversation/hosted-agent-conversation-README.md)
owns this contract; this section places it in the fusion.)

```text
            TIMELINE  (layer 1 — platform-owned, keyed by conversation)
            ordered blocks: messages inline, external events as URIs
              |                    |                       |
              v                    |                       |
   read side, any worker:          |                       |
   restore | search | hosting      |                       |
   downloads | pull-by-URI         |                       |
                                   v                       v
                             NATIVE agent            INTEGRATED agent
                       (timeline IS the state)  (timeline REFLECTS the state)
                                   |                       |
                                   v                       v
                        session view rendered      own checkpointer
                        per round: recent full,    (platform-scoped key from
                        older -> summaries + URIs, agent + user + conversation);
                        refresh by pull            inputs/outputs captured
                                                   back into the timeline
```

### Layer 1: the platform timeline — the durable record

The platform owns the **timeline**: an ordered event log per conversation.
User and assistant messages sit inline as blocks; external events — tool
results, files, anything the runtime ingests — are captured as blocks whose
payloads live behind **URIs** (`conv:fi:`, `conv:ar:`, `conv:tc:`, ...). The
turn log is the single source of truth for reconstructing a turn; fetch
rebuilds the chat entries from the ordered block stream
([Turn Log](harness/timeline/turn-log-README.md)).

The timeline feeds *everything downstream of the agent*:

- **conversation restore** — reload rebuilds bubbles, attachment and file
  cards, citations, cost and timing from the recorded blocks and artifacts;
- **cross-conversation search** — the same turn logs power the hybrid search
  engine behind the ReAct tool, the `conv` named service, REST, and the chat
  UI ([Conversation Search](../sdk/solutions/conversation/search-README.md));
- **file hosting, both directions** — a user downloads uploads and
  agent-produced files later via the object-action contract, and the agent's
  own generated code `pull`s those same files into its execution workspace.
  One durable link, resolved two ways;
- **forking** — a subagent's conversation is seeded with a projection copy of
  the parent's, by value ([Timeline Fork](../sdk/solutions/timeline/fork-README.md)).

The timeline is **progressive and portable**: because the platform owns it as
neutral blocks, the harness can pull it on any supported trusted worker or
runtime, prune and compact it, and render blocks at different levels of detail
by age and by consuming surface. It is keyed to the conversation, never to a
process.

### Layer 2: the agent's working memory — what the model sees

The second layer is what the model actually receives on a turn. Its owner
depends on the agent kind:

**Native agent.** For KDCube ReAct the timeline is not only the record — it
is the **source of truth for the working memory too**. Each round the harness
renders a **session view** from the timeline: recent turns in full; turns
past the recency window collapsed to compact working-summary cards; under the
hard token ceiling, the oldest visible prefix replaced by a single
range-summary checkpoint. Large tool results and reads appear as bounded
previews; the full content stays stored and addressable. The summaries carry
the URIs of what they collapsed, so the agent can **refresh itself** —
retrieve exact collapsed data by its URI (`react.read`, `react.pull`) or find
it when no path is known (`react.memsearch`) — depth on demand rather than
history resent. Volatile per-round state (limits, live turn events, workspace
status, temporal context) rides the regenerated ANNOUNCE tail so the stable
prefix stays cache-pure.
([Session View](../sdk/agents/react/session-view-README.md),
[Compaction](../sdk/agents/react/compaction-README.md),
[Memory Recovery Path](../sdk/agents/react/memory-recovery-path-README.md).)

**Integrated agent.** A ported framework keeps its own working memory in its
own store. The LangGraph app opens a durable checkpointer per agent
(`AsyncPostgresSaver` routed onto KDCube's Postgres). Its `thread_id` is a
platform-derived key containing tenant, project, active agent, user, and
conversation identity — never the browser session id, which changes per
session and would open an empty thread on reload. The alternative
design — reconstructing prior turns from the platform record each turn and
feeding them in — is equally valid; the record is the single source of truth
at the cost of a reconstruction step. What is unsafe is treating
client-submitted history or process memory as durable.

The timeline's relationship to the two kinds differs in one sentence: for the
native agent the timeline **is** the state; for an integrated agent the
timeline **reflects** the state — the runtime captures the agent's inputs and
outputs into the record (rich adapter blocks where the framework provides
them, the framework-neutral fallback recorder where it does not), so
restore, search, hosting, and title work identically while the framework's
own memory stays framework-native.

## 3. Block Production Is Namespace-Owned

The timeline is not a fixed schema. Event sources register per-namespace
resolvers, and each namespace **owns how its events become blocks and how
those blocks present everywhere**:

```text
   event arrives for a namespace  (task: | mem: | conv: | ...)
                        |
                        v
        registered namespace resolver — the OWNER
          |                 |                  |
          v                 v                  v
     block             rehosting          presentation
     production        pull an owner      the same owner shapes
     (produces and     ref -> content     the block on EVERY surface:
     patches ONLY      mirrored in,       agent context | chat reload |
     its own blocks)   conv:fi: +         external view | MCP
                       physical path
                       returned
```

- **Block production.** A provider for `task:` produces and may patch only
  the timeline blocks it owns; it cannot touch `mem:` or `conv:` blocks.
  This is an integrity boundary, not a rendering convenience
  ([Provider-Owned Timeline Projection](harness/timeline/provider-projection-README.md)).
- **Rehosting.** When the agent pulls an owner ref (`mem:mem_123`,
  `task:issue:...`), the namespace rehoster mirrors the owner's content into
  the conversation's artifact space and returns concrete `conv:fi:` +
  physical paths. External owner refs are opaque until pulled; the agent uses
  returned paths, never derived ones
  ([Artifact Resolution And Materialization](harness/events/artifact-resolution-and-materialization-README.md)).
- **Cross-surface presentation.** The same owner shapes how its event renders
  in the agent's context, in the chat UI on reload, in the compact external
  timeline served to MCP and named-service consumers, and in a scene. One
  owner of "how my events look", on every surface.

For the agent this means the timeline can absorb **any kind of event** — a
provider app ships its namespace resolver and its events become first-class
timeline citizens with correct presentation and pull behavior, with no change
to any agent framework.

## 4. What The Fusion Gives An Agent

The concrete gains an agent receives by living inside the runtimes — native
or integrated:

| Gain | Where it comes from |
| --- | --- |
| Multi-user serving with bound identity | every entry binds the authenticated actor; platform storage helpers apply tenant/project/user scope, and trusted app code must preserve those owner keys |
| Horizontal scale | state keys on the conversation, the agent is rebuilt per turn, so any worker can take the next turn |
| Ordered conversations under concurrency | the event lane serializes per conversation, parallelizes across conversations |
| Any-event ingestion | a common request/event context across transports; conversational work enters the lane, while namespaces own block production for domain events |
| Live follow-up and steer | the shared event source + owner lease; a live ReAct turn can fold both, eligible continuations can queue, and an unconsumed steer expires |
| Live streaming back to the initiator, across communicator-enabled runtimes | a comm spec crosses with supported work and the far side rebuilds it; selected recordable events and durable turn blocks later hydrate the conversation view |
| Durable history, restore, titles | the platform timeline and turn recording, framework-neutral |
| Cross-conversation search | the same record, searchable under the caller's user boundary |
| File hosting both ways | durable `conv:fi:` links: user downloads later, agent code pulls the same bytes into its workspace |
| Depth on demand over URIs | summaries carry refs; read/pull/search recover exact collapsed content |
| Cheap subagents | timeline fork by value + fenced child runtimes with reduce |
| Isolated generated code | the split execution fence; tool calls brokered by the trusted supervisor |
| Accounting that follows the request | the accounting subject rides supported runtime crossings and attributes integrated model, embedding, search, and participating custom-call paths |
| Consent-gated capability | demand-driven claims at tool-attempt time; grants per user, per agent, revocable |

None of it is automatic exposure: the app descriptor declares what exists,
the per-agent inventory narrows it, and the user narrows it further
([Agentic Runtime capability matrix](../sdk/bundle/surfaces/as-consumer-surfaces-README.md)).

## 5. Native And Integrated, Side By Side

The two worked apps make the difference concrete:

| Dimension | `workspace` (native ReAct) | `ported-langgraph-agents` (integrated) |
| --- | --- | --- |
| Reasoning core | KDCube ReAct rounds, protocol, action governance | framework/domain-owned LangGraph graphs under `solution/`; deliberate integration changes stay small and documented |
| Construction | agent built fresh per turn from `config.react` ⊕ `surfaces.as_consumer` (inventory, instructions, traits, event sources) | dispatcher resolves the agent id, builds a fresh graph per turn bound to the turn's identity |
| Working memory | the rendered session view over the timeline (age-graded detail, refresh-by-URI) | the framework checkpointer on KDCube Postgres, keyed by platform-scoped agent + user + conversation identity |
| Timeline role | source of truth — state and record are one | reflection — inputs/outputs captured into the record |
| Mid-turn events | folds follow-up/steer live at decision boundaries | folds the accepted ingress batch at start; eligible mid-turn messages wait for the next turn, while unconsumed steer expires |
| Tools | SDK tools + named services + MCP, taught by composed instruction blocks | app `@tool`s + selected SDK wrappers + MCP tools + consent placeholders, mapped by the adapter |
| Generated code | full exec path with exported tool catalog (nested `tool_call` through the supervisor) | `run_python` — isolated computation + hosted files; nested catalog exported only if the adapter passes specs |
| Streaming | channel protocol through the communicator | LangGraph events mapped to chat events by the stream adapters (`platform/stream_*.py`) |

What the integrated agent **gains** without changing its core: everything in
§4 — hosting, restore, search, any-event handling, scale, accounting, consent
— plus the same workspace grammar (fresh per turn, refs in, produced files
hosted out) and the same pull-over-URIs primitives the native agent uses
(`pull_refs_into_dir`; the react tools are one adapter over them, the ported
app binds the same helpers as `read_file`/`pull_files`).

One integrated-agent nuance deserves its own note: **Claude Code**. Hosted
Claude Code keeps continuity through its *own* session substrate
(`--session-id`/`--resume` with a deterministic id from
user + conversation + agent), not through the platform record; KDCube makes
that substrate durable across workers with a git-backed session store, one
branch per conversation boundary
([Claude Code Agent](../sdk/agents/claude/claude-code-README.md),
[Workspace Management](../sdk/agents/claude/claude-code-workspace-bootstrap-README.md)).
It is the clearest illustration that the platform record and a framework's
working memory are genuinely different layers: the record is still captured
and serves restore/search, while the framework's continuity lives in its own
files.

## 6. One Turn, End To End

The whole fusion in a single pass — native agent shown; the integrated path
differs only where marked:

```text
a person, webhook, automation, or MCP client submits work
        |
        v
ingress normalizes to ExternalEventPayload
  and atomically appends the accepted start batch to the conversation source
        |
        v
the lane reserves the accepted batch for ONE turn <- serialized per conversation
        |
        v
run() binds identity, opens turn + accounting context,
  refreshes effective app properties, starts recording
        |
        v
execute_core(...)
  native: compose instructions, render the session view
          from the timeline, enter ReAct rounds
  ported: fold the ingress batch, restore the checkpointer
          thread, build the graph fresh, run to completion
        |
        +-- tools cross their fences as needed:
        |     exec -> supervisor/executor; accounts -> credential broker;
        |     namespace ops -> provider; all carrying the portable context
        |
        +-- progress streams the whole way: the communicator peers deltas,
        |     steps, and files back to the initiator's surfaces — from proc,
        |     supervisor, and subagent alike
        |
        +-- (native) live followup/steer fold in;
        |   subagent forks spawn and reduce
        |
        v
recording lands the turn:
  turn.log blocks, conversation timeline artifact,
  chat events (cost, timing, citations), stream artifacts,
  produced files hosted as conv:fi: links
        |
        v
the lane reservation finalizes; eligible pending work may schedule next
        |
        v
read side, any time later, any worker:
  restore in chat | cross-conversation search | file download | pull by URI
```

Every supported runtime crossing follows one of the boundary contracts from
the [runtimes map](runtimes-map-README.md). Conversation-facing durable output
on the right-hand side becomes timeline blocks or durable refs; framework
checkpoints and app-domain state remain in their own stores. That is the
fusion: one agent, any declared surface in, several runtimes underneath, one
platform-owned conversation record out.
