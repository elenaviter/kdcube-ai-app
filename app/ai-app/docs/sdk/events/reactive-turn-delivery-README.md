---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/events/reactive-turn-delivery-README.md
title: "Reactive Turn Delivery"
summary: "Framework-neutral contract for ordered reactive delivery: ReAct may fold live, while a foreign runtime folds the whole pending lane once and watches control read-only; both release before liveness handoff."
status: active
tags: ["sdk", "events", "external-events", "turns", "react", "run-to-completion", "followup", "steer", "ordering"]
updated_at: 2026-08-18
keywords:
  [
    "reactive turn delivery",
    "on_reactive_event door",
    "run() execute_core",
    "conversation event lane wakeup",
    "consumer reservation scheduled active none",
    "per-conversation serialization",
    "run-to-completion turn",
    "reactive_lane finalize",
    "followup promotion",
    "scheduled_consumer_fresh",
  ]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/events/conversation-event-lane-state-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/events/event-ingress-to-react-turn-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/events/external-events-journey-and-handling-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/dataflow/connect-agentic-loop-to-ordered-delivery-README.md
---
# Reactive Turn Delivery

This page answers one question, for **any** agent — ReAct, a ported graph, a
bespoke loop:

```text
How does accepted reactive work reach my agent in arrival order,
either by joining an open live turn or by starting the next
serialized turn for this conversation?
```

The delivery contract is the same for every agent. What differs is whether the
agent owns a live handler that can fold new work or takes one read-only snapshot
of the pending lane before it runs. That choice determines which runtime
finalizer releases the lane.

For the Redis state fields this page refers to (`T.consumer.status`,
`T.handler.status`, the reactive cursor), see the
[Conversation Event Lane State](./conversation-event-lane-state-README.md)
reference — this page does not restate them. For the ReAct-specific,
field-level transport journey, see
[Event Ingress To React Turn](./event-ingress-to-react-turn-README.md).

## One-page model

```text
turn-starting event (prompt / queued followup)
        │  written to the conversation event lane (ordered log)
        │  + primary wakeup enqueued                             [atomic]
        ▼
   processor claims the wakeup
        │  acquires the per-conversation lock  ── serialization point
        │
        ├─ open live handler owns it  → fold into the running turn
        │
        └─ otherwise reserve: T.consumer.status = scheduled
        ▼
   run()   ── the @on_reactive_event door (shared by every agent)
        │
        ▼
   execute_core(state, thread_id, params)   ── your agent runs the turn
        │
        ▼
   turn ends  →  account accepted work
                 →  release the lane consumer reservation
                 →  optionally publish a duplicate liveness wake
```

The wakeup is the only thing that starts a new agent turn. It points at one
accepted lane occurrence, not at the complete input. `run()` is the shared
`@on_reactive_event` door on the app base (`BaseEntrypoint.run` /
`BaseEntrypointWithEconomics.run`). The native ReAct agent opens the live handler there; a
foreign-runtime adapter reads the whole still-pending lane once, sequence orders
it, and maps that snapshot into its framework input before `execute_core` runs.

## Ordered, serialized delivery

Turns of one conversation are **serialized** by a per-conversation lock. A second
event that arrives while a turn is running does **not** start a concurrent
`execute_core`:

```text
Event 1 → wakeup → lock acquired → run()/execute_core (turn 1 running)
Event 2 arrives now → wakeup enqueued → processor tries to claim it
        → cannot acquire the conversation lock → REQUEUES (waits)
turn 1 ends → reservation released → lock released
        → Event 2's wakeup is claimed → run()/execute_core (turn 2, in order)
```

- **Same conversation:** one turn at a time, in arrival order. The lock holds
  across processor workers, so two workers cannot run two turns of the same
  conversation at once.
- **Different conversations:** run in parallel (independent locks).

This is a platform guarantee — agent code does not implement it. The active
consumption model must leave the lane in a releasable state; ReAct owns that
lifecycle inside its workflow, while the shared door finalizes it for a
run-to-completion loop.

## The lane consumer reservation

When the processor dispatches a wakeup it **reserves** the lane consumer
(`T.consumer.status = scheduled`) before the turn runs. The reservation exists so
the platform knows a turn is responsible for this lane. Finalization releases it
(`→ none`) when the turn is done. A fresh duplicate wake is deferred while the
reserved starter is loading; after turn work is persisted, release happens
before any additional liveness wake is published.

Releasing the reservation is where the two agent models diverge.

## Two consumption models

### 1. ReAct — a live consumer that folds mid-turn

A ReAct `execute_core` drives a `BaseWorkflow`, which **opens the lane handler**
(`T.handler.status = open`), marks the consumer `active`, and reads the lane
*during* the turn. A followup that lands mid-turn is folded into the running turn
at a decision boundary. The close gate first stops the live reader once the
handler is closed. After turn artifacts persist, finalization advances the
reactive cursor, releases the consumer (`→ none`), and only then publishes a
duplicate liveness wake for anything still unconsumed. ReAct owns its lane
lifecycle end-to-end, inside its own workflow. See
[Event Ingress To React Turn](./event-ingress-to-react-turn-README.md).

### 2. Run-to-completion — one pending snapshot, one turn

A run-to-completion `execute_core` (a ported graph, a bespoke loop) runs
start-to-finish without opening the ReAct handler. At turn start its adapter
performs one read-only lane fold. It selects the wake occurrence and every other
still-pending occurrence, skips consumed/promoted/failed work, sorts by lane
sequence, and records the exact selected message ids on turn state. The snapshot
can contain several user messages and their attachments across several ingress
batches; the model reads them together, in order, and answers once.

After that snapshot, a foreign-runtime watcher may observe control but does not
fold content into the running loop. Its turn-owned heartbeat keeps the
`scheduled` reservation fresh. LangGraph steer cancels the stream while its
checkpointer retains the last completed node; ordinary arrivals remain pending.
Because the runtime never opens the handler, the shared door accounts for the
snapshot by exact id and releases the consumer after the turn.

## The finalize invariant (`reactive_lane`)

The shared door (`run()`) finalizes the reactive-event lane after `execute_core`
returns — on success **and** error, skipped only on cancel (which stays on the
inflight-recovery path). It lives in a dedicated module,
`chatbot/reactive_lane.py`, and the door makes a single thin call to it. The
finalize is a **state-conditional, idempotent invariant — never an agent-type
branch**:

```text
already_released = T.consumer.status == "none"
own_accounted    = the wake occurrence is consumed / past the reactive cursor

if already_released and own_accounted:
    return                       # no-op

# otherwise a run-to-completion turn still owns the reservation:
mark the wake occurrence consumed          # normal lane cursor
mark every folded snapshot event consumed  # exact ids, not a range
release this turn's reservation (→ none)
optionally wake ONE eligible later intent  # after the last steer boundary
terminalize a bare event.user.steer
```

The `already_released and own_accounted` state is *exactly* what a ReAct turn's
`BaseWorkflow` leaves behind — so a ReAct turn is inert here **by state, with no
`if react` check**. A run-to-completion turn left the reservation `scheduled`, so
the predicate is false and the door releases it. The finalize reuses only the
existing lane primitives (the same exactly-once `mark_consumed_up_to` a
`BaseWorkflow` uses for the wake occurrence, exact `mark_consumed_event` for
the folded snapshot, owner-fenced `mark_consumer_none`, and the wake re-publish);
it does not infer an agent type and touches nothing on the ReAct path. The folded event-id
input is absent for the native ReAct agent, so the shared finalizer remains inert there by
state.

Consuming only the wake occurrence is not enough. If the snapshot contains a
prompt, its attachment, and two messages queued during the previous turn, all
four were shown to this turn and all four exact lane occurrences must be
accounted after it returns. A sequence-range consume is not equivalent: an
arrival may land after the snapshot while the runtime is already executing and
must remain pending for the handoff.

A batch with no reactive event enqueues no processor wake. It is retained or
callbacked according to the normal external-event rules, but it does not enter
this finalizer and cannot start a turn by itself.

## Followup and steer, per model

`accepts_followup` / `accepts_steer` are per-agent capability declarations. They
change what the composer *offers*, not what is delivered:

- **ReAct (accepts both):** a mid-turn followup is folded into the running turn; a
  steer cancels and finalizes it.
- **Run-to-completion:** a mid-turn message is not folded into the running turn's
  own loop — that loop belongs to the graph or the CLI, and folding into it would
  contradict the iteration management the agent's own design owns. What the turn
  does instead is WATCH (`foreign_runtime/live_watch.py`), and each lane decides
  what watching buys it:

  | Lane | A follow-up mid-run | A steer mid-run |
  | --- | --- | --- |
  | LangGraph (ported agents) | not delivered; folds into the next turn | cancels the streaming task; the checkpointer holds the last completed node |
  | Claude Code (hosted) | delivered before the next tool call, via a `PreToolUse` hook — the model reads it and keeps working | the same hook DENIES the next tool call, so the model answers with what it has |

  Neither can reach a run that is inside a long tool call, and neither kills a
  process to stop it.

**What arrives while a turn runs now waits for the handoff, for the whole turn.**
That used to hold for thirty seconds: a run-to-completion turn never marked the
lane reservation, so `scheduled` went stale (`scheduled_ttl_ms`, 30 000 ms) and a
later message was admitted as its own turn. The watcher heartbeats the consumer
on its poll, but only when `consumer_turn_id` matches and status is still
`scheduled`, so the reservation lasts as long as the owning turn does.

**The handoff wakes ONE turn**, not one per pending event, and the fold takes the
whole pending lane rather than the wakeup's own batch. Two messages typed during
a turn are answered together, in order, by one turn — before, each promoted
alone and the agent answered the first without knowing the second existed.

**A steer is a boundary at the handoff.** Re-wake only when reactive events
arrived AFTER the last steer:

- before it — said while the stopped run was going; superseded as a REASON to
  run, kept pending, folded into whatever turn a person starts next
- the steer itself — a bare one is terminalized (spent: the turn it asked to
  stop is over, and waking for it handed the agent an empty message). One with
  text does not wake either, and its text stays pending for the next fold
- after it — new intent, and it wakes one turn

An agent that *can* consume mid-turn into its own loop integrates the ReAct-style
handler (open/close, `mark_consumed_up_to`) inside its own `execute_core`; see
the recipe.

## The failure mode this prevents

Before the finalize existed, a run-to-completion turn left the reservation
`scheduled`. The next turn's wakeup, arriving inside the reservation's TTL, was
dropped as `scheduled_consumer_fresh` — the turn "completed" in the UI but the
*next* message never reached `execute_core` (nothing in the processor log). It
self-recovered only after the TTL went stale, which read as intermittent
"second/third turn hangs." The finalizer now accounts for the start snapshot,
releases the reservation, and only then may publish a liveness wake.

## Boundary

This page is the framework-neutral delivery contract. It does not cover the
transport-level field origins (see the ReAct ingress page), the lane state fields
(see the lane-state reference), or how to build a specific agent (see the
recipe). The one rule every run-to-completion integration must respect is
implicit and handled for you: **a turn releases the lane it was handed.** The
`run()` door guarantees it; you implement `execute_core` and get ordered,
serialized delivery with exactly-once lane accounting.
