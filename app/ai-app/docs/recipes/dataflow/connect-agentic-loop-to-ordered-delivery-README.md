---
id: repo:kdcube-ai-app/app/ai-app/docs/recipes/dataflow/connect-agentic-loop-to-ordered-delivery-README.md
title: "Connect Your Agentic Loop To Ordered Message Delivery"
summary: "Executable recipe for wiring any agentic loop into KDCube's ordered, serialized, exactly-once message delivery: implement execute_core behind the @on_reactive_event door and get one-event-per-turn ordering for free; if your loop can consume mid-turn, how to opt into followup/steer integration."
status: draft
tags: ["recipes", "dataflow", "events", "turns", "run-to-completion", "followup", "steer", "execute_core"]
updated_at: 2026-07-13
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/events/reactive-turn-delivery-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/events/conversation-event-lane-state-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/apps/settle-your-solution-in-kdcube-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/chat/chat-stream-events-README.md
---
# Connect Your Agentic Loop To Ordered Message Delivery

You have an agentic loop — your own graph, framework, or hand-written control
flow. This recipe wires it so a user's messages arrive **in order, one turn at a
time per conversation, exactly once, with no turn lost** — and, if your loop
supports it, so a mid-turn followup or steer is handled correctly.

You do **not** touch the event lane, locks, or wakeups. You implement one method
and (optionally) declare one capability. The delivery guarantees are the
platform's; the mechanism behind them is in
[Reactive Turn Delivery](../../sdk/events/reactive-turn-delivery-README.md) —
read that first if you want the model; this page is the how-to.

## What you implement vs. what you get for free

| You implement | The platform gives you |
| --- | --- |
| `execute_core(...)` — run one turn for one event | The `@on_reactive_event` door (`run()`) that invokes it |
| (optional) capability declaration | Per-conversation serialization (one turn at a time, in order) |
| (optional) mid-turn integration | Exactly-once delivery + next-turn promotion of a queued followup |

## Step 1 — Implement `execute_core`

Subclass an app base (`BaseEntrypoint`, or `BaseEntrypointWithEconomics` /
`BaseEntrypointWithMemory` for the extra seams) and implement the one abstract
method. It receives the turn's event and runs your loop to completion:

```python
class MyAppEntrypoint(BaseEntrypointWithEconomics):
    async def execute_core(self, *, state, thread_id, params):
        # The triggering event(s) arrive as the external-event batch on `state`.
        # Extract the user's message from it (helper shown in the port recipe):
        question = external_events_text(state.get("external_events") or [])
        # ... run YOUR loop / graph to completion ...
        # stream tokens + steps through the current communicator:
        #   comm_ctx.delta(...) / comm_ctx.step(...) / comm_ctx.complete(...)
        # return your turn result
```

- One call to `execute_core` = **one event = one turn**. Read the triggering
  message from `state`/`params`; do not try to poll the lane for more.
- Stream the answer through `comm_ctx` (see
  [Chat Stream Events](../../sdk/solutions/chat/chat-stream-events-README.md)) so
  the reusable chat component renders it live.
- Be stateless across turns — persist any per-user/per-conversation state to a
  backend (the turn may run on a different worker next time).

That is the entire "connect a loop" step. The door (`run()`) already calls it,
serializes it per conversation, and — because your `execute_core` does not hold
the lane open — releases the lane consumer for you at turn end so the next
message promotes. You get ordered, one-at-a-time delivery with nothing else to
write. This is the **run-to-completion** path.

## Step 2 — Declare how your agent handles mid-turn messages

A message sent while a turn is running is a *followup*; a cancel is a *steer*.
Whether your agent can act on them mid-turn is a per-agent declaration:

```yaml
# in the app descriptor, for this agent
conversation:
  accepts_followup: false     # can this loop fold a new message mid-turn?
  accepts_steer:    false     # can it cancel + finalize mid-turn?
```

### Default: run-to-completion (both `false`)

Pick this when your loop runs start→finish without checking for new input (most
ported graphs). You write **no** followup/steer code:

- A mid-turn followup is **queued and promoted to the next turn** automatically —
  the door's finalize re-wakes it after the current turn ends. The composer shows
  "Queue for next turn".
- Turns still serialize and stay in order; nothing is lost.

This is the correct, honest choice for a loop that cannot safely absorb new input
mid-flight. Do not declare `true` unless Step 3 is actually wired.

### Advanced: consume mid-turn (either `true`)

Pick this only if your loop can accept a new event at a boundary while running.
Then your `execute_core` must own the lane lifecycle itself, the way the ReAct
workflow does: open the lane handler at turn start, read/fold events at your
boundaries, and close + re-wake at completion (`open/close_external_event_handler`,
`mark_consumed_up_to`). Follow the ReAct integration as the reference
implementation rather than reinventing it — see
[Event Ingress To React Turn](../../sdk/events/event-ingress-to-react-turn-README.md)
and the lane-state rules in
[Conversation Event Lane State](../../sdk/events/conversation-event-lane-state-README.md).

The rule that keeps both paths correct: **a turn releases the lane it was handed.**
Run-to-completion lets the door release it; a mid-turn consumer releases it
itself. Declaring `accepts_followup: true` without owning the handler will fold
nothing and mislabel the composer — declare what you actually implement.

## Step 3 — Verify

1. Send several messages back-to-back in one conversation (include one that
   triggers your heaviest path, e.g. a sub-step or retrieval).
2. Confirm each becomes its **own** turn, processed **in order**, one at a time —
   each `execute_core` appears in the processor log sequentially, none skipped.
3. Send a message *while a turn is running*:
   - run-to-completion → it lands as the **next** turn after the current one
     finishes;
   - mid-turn consumer → it folds into the running turn.
4. Confirm a second conversation runs concurrently with the first.

If a later turn "completes" in the UI but never appears in the processor log,
the lane consumer was not released — re-read
[Reactive Turn Delivery](../../sdk/events/reactive-turn-delivery-README.md) §"The
failure mode this prevents".

## Related

- [Reactive Turn Delivery](../../sdk/events/reactive-turn-delivery-README.md) —
  the mechanism this recipe builds on.
- [Settle Your Solution In A KDCube App](../apps/settle-your-solution-in-kdcube-README.md)
  — the end-to-end host integration; this recipe is the delivery slice of it.
