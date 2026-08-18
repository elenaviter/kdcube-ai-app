---
id: repo:kdcube-ai-app/app/ai-app/docs/recipes/dataflow/connect-agentic-loop-to-ordered-delivery-README.md
title: "Connect Your Agentic Loop To Ordered Message Delivery"
summary: "Executable recipe for wiring any agentic loop into KDCube's ordered, serialized delivery: fold the whole pending lane once before execute_core, let the shared door account exact events, and optionally add a read-only stop watcher or a true live handler."
status: draft
tags: ["recipes", "dataflow", "events", "turns", "run-to-completion", "followup", "steer", "execute_core"]
updated_at: 2026-08-18
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/events/reactive-turn-delivery-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/events/conversation-event-lane-state-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/apps/settle-your-solution-in-kdcube-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/chat/chat-stream-events-README.md
---
# Connect Your Agentic Loop To Ordered Message Delivery

You have an agentic loop — your own graph, framework, or hand-written control
flow. This recipe wires it so a user's messages arrive **in order, one
serialized turn at a time per conversation, exactly once, with no turn lost**.
A turn may contain several lane occurrences: when a foreign runtime starts, it
reads every event still pending and answers those messages together, in order.
If the loop supports live control, the adapter can also watch followup and steer
while the native loop runs.

You do **not** implement event-lane primitives, locks, or wakeups. You call the
shared read-only fold helper from one method and declare the controls your
adapter actually supports. The delivery guarantees are the platform's; the
mechanism behind them is in
[Reactive Turn Delivery](../../sdk/events/reactive-turn-delivery-README.md) —
read that first if you want the model; this page is the how-to.

## What you implement vs. what you get for free

| You implement | The platform gives you |
| --- | --- |
| `execute_core(...)` — run one turn over one pending-lane snapshot | The `@on_reactive_event` door (`run()`) that invokes it |
| One read-only fold before the native loop starts | Per-conversation serialization and a turn-owned reservation |
| An honest capability declaration | Exact-id accounting, release, and at most one next-turn wake |
| (optional) read-only stop watcher or live handler | Runtime-specific mid-turn control without changing the native loop's ownership |

## Step 1 — Implement `execute_core`

Subclass an app base (`BaseEntrypoint`, or `BaseEntrypointWithEconomics` /
`BaseEntrypointWithMemory` for the extra seams) and implement the one abstract
method. It receives the turn's event and runs your loop to completion:

```python
from kdcube_ai_app.apps.chat.sdk.protocol import external_events_texts
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime import (
    fold_turn_external_events,
)


class MyAppEntrypoint(BaseEntrypointWithEconomics):
    async def execute_core(self, *, state, thread_id, params):
        # A wake names one occurrence, not the complete turn input. Fold every
        # still-pending occurrence once, before the native loop starts.
        state["external_events"] = await fold_turn_external_events(self, state)

        # Keep every user message. A later one may correct an earlier one.
        messages = external_events_texts(state.get("external_events") or [])
        question = "\n".join(
            f"{index}. {text}" for index, text in enumerate(messages, start=1)
        )
        # The port recipe also demonstrates preserving attachment siblings and
        # each message's batch/sequence/arrival metadata.
        # ... run YOUR loop / graph to completion ...
        # stream tokens + steps through the current communicator:
        #   comm_ctx.delta(...) / comm_ctx.step(...) / comm_ctx.complete(...)
        # return your turn result
```

- One call to `execute_core` is **one serialized turn, not necessarily one
  event**. `fold_turn_external_events(...)` reads the pending lane once, in
  sequence order, and records the exact selected ids for shared finalization.
  It is read-only and fail-open: it does not consume events or change the lane
  reservation.
- Treat all user messages in that snapshot as one ordered input. A later message
  may correct an earlier one; do not silently reduce the snapshot to its first
  text event.
- Do not keep polling for content after the snapshot. Events arriving while the
  native loop runs remain pending for the handoff or may be observed by an
  optional control watcher.
- Stream the answer through `comm_ctx` (see
  [Chat Stream Events](../../sdk/solutions/chat/chat-stream-events-README.md)) so
  the reusable chat component renders it live.
- Be stateless across turns — persist any per-user/per-conversation state to a
  backend (the turn may run on a different worker next time).

That is the entire basic "connect a loop" step. The door (`run()`) reserves the
lane for this turn, invokes the loop under the per-conversation lock, accounts
the fold's exact ids, and releases the reservation. If eligible intent remains
after the latest steer boundary, finalization publishes at most one liveness
wake; the next turn folds the whole pending lane again. This is the
**run-to-completion** path.

## Step 2 — Declare how your agent handles mid-turn messages

A message sent while a turn is running is a *followup*; a cancel is a *steer*.
Whether your agent can act on them mid-turn is a per-agent declaration:

```yaml
# in the app descriptor, for this agent
conversation:
  accepts_followup: false     # can this loop fold a new message mid-turn?
  accepts_steer:    false     # does this adapter actually stop the running loop?
```

### Basic run-to-completion (both `false`)

Pick this when your loop runs start→finish without checking for control. You
write no watcher or live-handler code:

- Mid-turn messages remain pending. At handoff, one eligible wake starts the
  next turn, which folds all pending messages together rather than promoting
  them one turn each.
- A steer is not offered as a supported live action because this adapter cannot
  stop the native loop.
- Turns remain serialized and ordered; the shared door performs exact-id
  accounting and owner-fenced release.

This is the correct declaration for a loop that cannot safely absorb or act on
new input mid-flight. Do not declare `true` unless the corresponding behavior is
actually wired.

### Read-only stop control (`accepts_steer: true`)

A foreign runtime can support stop without pretending that KDCube owns its
iteration loop. Wrap the native awaitable with
`foreign_runtime.run_until_stopped(...)`. Its watcher reads control events
without consuming or folding them and refreshes only this turn's own
`scheduled` reservation. On steer, it cancels the adapter's stream task; the
native framework remains responsible for its own recovery. The ported LangGraph
bundle uses this model, with the checkpointer retaining its last completed node.
It declares `accepts_followup: false` and `accepts_steer: true`.

At finalization, a bare steer is spent as the terminal stop boundary. A steer
carrying text stays pending for a later fold but does not wake a turn by itself.
Only eligible intent after the latest steer may publish one wake.

### Advanced: consume content mid-turn

Pick this only if your loop can accept a new event at a boundary while running.
Then your `execute_core` must own the live lane lifecycle itself, the way the
ReAct workflow does: open the handler at turn start, read/fold events at your
boundaries, and close plus account at completion
(`open/close_external_event_handler`, `mark_consumed_up_to`). Follow the ReAct
integration as the reference
implementation rather than reinventing it — see
[Event Ingress To React Turn](../../sdk/events/event-ingress-to-react-turn-README.md)
and the lane-state rules in
[Conversation Event Lane State](../../sdk/events/conversation-event-lane-state-README.md).

The rule that keeps every path correct: **a turn releases only the lane
reservation it owns.** Run-to-completion lets the door perform an owner-fenced
release; a live consumer releases through its own workflow. Declaring
`accepts_followup: true` without owning an open handler folds nothing and
mislabels the composer.

## Step 3 — Verify

1. Hold one turn open, send several messages behind it, then let it finish.
2. Confirm the handoff starts at most one next `execute_core`, whose snapshot
   contains all pending messages in lane order and answers them once.
3. With basic run-to-completion, confirm a mid-turn message remains pending and
   appears in that next snapshot. With a live handler, confirm it folds into the
   running turn instead.
4. If steer is enabled, stop a running turn. Confirm the adapter cancels, a bare
   stop does not create an empty next turn, and new intent after the stop starts
   exactly one turn.
5. Confirm a second conversation runs concurrently with the first.

If a later turn "completes" in the UI but never appears in the processor log,
the lane consumer was not released — re-read
[Reactive Turn Delivery](../../sdk/events/reactive-turn-delivery-README.md) §"The
failure mode this prevents".

## Related

- [Reactive Turn Delivery](../../sdk/events/reactive-turn-delivery-README.md) —
  the mechanism this recipe builds on.
- [Settle Your Solution In A KDCube App](../apps/settle-your-solution-in-kdcube-README.md)
  — the end-to-end host integration; this recipe is the delivery slice of it.
