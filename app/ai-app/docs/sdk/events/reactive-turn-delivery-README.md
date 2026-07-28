---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/events/reactive-turn-delivery-README.md
title: "Reactive Turn Delivery"
summary: "Framework-neutral contract for ordered reactive delivery: an open ReAct handler may fold new work into its live turn, while a run-to-completion agent receives one accepted start batch per serialized turn; both release the lane before any liveness handoff."
status: active
tags: ["sdk", "events", "external-events", "turns", "react", "run-to-completion", "followup", "steer", "ordering"]
updated_at: 2026-07-28
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
agent owns a live handler that can fold new work or consumes one fixed start
batch. That choice determines which runtime finalizer releases the lane.

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

The wakeup is the only thing that starts a new agent turn. `run()` is the shared
`@on_reactive_event` door on the app base (`BaseEntrypoint.run` /
`BaseEntrypointWithEconomics.run`); it calls your `execute_core`, which reads the
accepted start batch out of `state`/`params` and produces the turn.

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

### 2. Run-to-completion — one start batch, one turn

A run-to-completion `execute_core` (a ported graph, a bespoke loop) runs
start→finish and does **not** watch the lane. It consumes exactly the accepted
start batch; a followup that lands mid-turn is *not* folded. Because it never
opens the handler, it never releases the reservation on its own — so the
platform accounts for the batch and releases the consumer *for* the agent, from
the door, after the turn. That is the finalize invariant below.

The net behavior for a run-to-completion agent: **one accepted start batch → one
turn**, strictly serialized and in order. Work that arrives mid-turn stays in the
lane and becomes eligible to schedule after the current turn releases it.

## The finalize invariant (`reactive_lane`)

The shared door (`run()`) finalizes the reactive-event lane after `execute_core`
returns — on success **and** error, skipped only on cancel (which stays on the
inflight-recovery path). It lives in a dedicated module,
`chatbot/reactive_lane.py`, and the door makes a single thin call to it. The
finalize is a **state-conditional, idempotent invariant — never an agent-type
branch**:

```text
already_released = T.consumer.status == "none"
own_accounted    = the turn's start batch is consumed / past the reactive cursor

if already_released and own_accounted:
    return                       # no-op

# otherwise a run-to-completion turn still owns the reservation:
mark the turn's start batch consumed      # exactly-once lane accounting
release the consumer (→ none)
optionally re-wake pending reactive work  # duplicate liveness signal
expire an unconsumed event.user.steer     # active-turn control, never future work
```

The `already_released and own_accounted` state is *exactly* what a ReAct turn's
`BaseWorkflow` leaves behind — so a ReAct turn is inert here **by state, with no
`if react` check**. A run-to-completion turn left the reservation `scheduled`, so
the predicate is false and the door releases it. The finalize reuses only the
existing lane primitives (the same exactly-once `mark_consumed_up_to` a
`BaseWorkflow` uses, `mark_consumer_none`, and the wake re-publish); it adds no
new orchestrator behavior and touches nothing on the ReAct path.

## Followup and steer, per model

`accepts_followup` / `accepts_steer` are per-agent capability declarations. They
change what the composer *offers*, not what is delivered:

- **ReAct (accepts both):** a mid-turn followup is folded into the running turn; a
  steer cancels and finalizes it.
- **Run-to-completion (declares both false):** a mid-turn message is queued for
  the next turn — it is not folded into the running turn. The finalizer releases
  the consumer before any optional liveness wake. This is the "Queue for next
  turn" composer state; agent code does not manage the lane.

`event.user.steer` is never included in that next-turn handoff. It controls only the
turn that was active when ingress accepted it; if that turn does not consume
the control before closing, the control expires.

An agent that *can* consume mid-turn integrates the ReAct-style handler
(open/close, `mark_consumed_up_to`) inside its own `execute_core`; see the recipe.

## The failure mode this prevents

Before the finalize existed, a run-to-completion turn left the reservation
`scheduled`. The next turn's wakeup, arriving inside the reservation's TTL, was
dropped as `scheduled_consumer_fresh` — the turn "completed" in the UI but the
*next* message never reached `execute_core` (nothing in the processor log). It
self-recovered only after the TTL went stale, which read as intermittent
"second/third turn hangs." The finalizer now accounts for the start batch,
releases the reservation, and only then may publish a liveness wake.

## Boundary

This page is the framework-neutral delivery contract. It does not cover the
transport-level field origins (see the ReAct ingress page), the lane state fields
(see the lane-state reference), or how to build a specific agent (see the
recipe). The one rule every run-to-completion integration must respect is
implicit and handled for you: **a turn releases the lane it was handed.** The
`run()` door guarantees it; you implement `execute_core` and get ordered,
serialized delivery with exactly-once lane accounting.
