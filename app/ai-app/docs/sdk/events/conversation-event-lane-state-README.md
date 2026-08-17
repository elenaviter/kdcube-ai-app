---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/events/conversation-event-lane-state-README.md
title: "Conversation Event Lane State"
summary: "SDK/runtime reference for the Redis state record that synchronizes conversation external-event ingress, readers, handlers, and wake handoff."
status: active
tags: ["sdk", "events", "external-events", "redis", "react", "synchronization"]
updated_at: 2026-08-17
keywords:
  [
    "conversation event lane state",
    "external event bus state",
    "handler status",
    "reactive event wake",
    "Redis key",
    "event timestamp",
  ]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/ecosystem-component/components-ecosystem-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/events/event-ingress-to-react-turn-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/events/reactive-turn-delivery-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/service/comm/conversation-event-bus-orchestrator-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/events/external-events-journey-and-handling-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/events/external-events-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/events/event-subsystem-README.md
---
# Conversation Event Lane State

The conversation event lane state is the synchronization record for one
conversation/agent external-event lane. The first implementation stores it as
JSON in Redis and updates it under a short Redis lock.

```text
<conversation-external-event-lane-key>:state
<conversation-external-event-lane-key>:state:lock
```

The lane identity is:

```text
tenant + project + user_id + conversation_id + agent_id
```

## Fields

```text
T.handler.turn_id
T.handler.status                         open | closed
T.handler.status_at

T.last_processed_event_timestamp
T.last_processed_event_id
T.last_processed_reactive_event_timestamp

T.consumer.status                        active | scheduled | none
T.consumer.status_at
```

Field meanings:

| Field | Writer | Meaning |
| --- | --- | --- |
| `T.handler.turn_id` | BaseWorkflow / ContextBrowser handler setup | Runtime turn id of the currently open/last closed handler. |
| `T.handler.status` | BaseWorkflow / ContextBrowser handler setup and ReAct close gate | `open` while the handler can accept lane events into this turn; `closed` after the close gate succeeds. |
| `T.handler.status_at` | Same writer as `T.handler.status` | Latest acknowledgement timestamp for `T.handler.status`. |
| `T.last_processed_event_timestamp` | Reader/Consumer while holding `lock(T)` | Maximum event-envelope timestamp accepted into the live turn path for any event type. |
| `T.last_processed_event_id` | Reader/Consumer while holding `lock(T)` | Event id for the latest accepted event at `T.last_processed_event_timestamp`; used as the tie-breaker when multiple events share a timestamp. |
| `T.last_processed_reactive_event_timestamp` | Reader/Consumer while holding `lock(T)` | Maximum event-envelope timestamp accepted into the live turn path for reactive events. |
| `T.consumer.status` | Proc and Reader/Consumer while holding `lock(T)` | `scheduled`, `active`, or `none`. |
| `T.consumer.status_at` | Same writer as `T.consumer.status`, and Reader acknowledgements | Latest real lane-local acknowledgement timestamp for `T.consumer.status`. |

`T.last_processed_*` values come from event-envelope timestamps. They are not
Redis Stream ids, internal sequence numbers, current timeline timestamps, or
wall-clock `now`.

## State Rules

Ingress prepares event batches before they are visible in the lane. For a
reactive batch, ingress accepts the batch only through an atomic Redis
operation that:

```text
publish prepared lane records
enqueue one wake for the first reactive event
```

If that atomic operation is rejected, no lane event from the batch exists and
the client receives a rejection. Ingress does not write `T.handler.*`,
`T.consumer.*`, or `T.last_processed_*`.

Proc reads wake items. Under `lock(T)`, proc sets:

```text
T.consumer.status = scheduled
T.consumer.status_at = now
```

only when the wake is not stale and no fresh active/scheduled Consumer is
already responsible for the lane.

BaseWorkflow / ContextBrowser handler setup sets:

```text
T.handler.turn_id = handler runtime turn id
T.handler.status = open
T.handler.status_at = now
```

Reader/Consumer activation sets:

```text
T.consumer.status = active
T.consumer.status_at = now
```

only when `T.handler.status == open`.

Reader/Consumer acceptance of lane entries is atomic with state updates:

```text
lock(T)
  if T.handler.status == open:
    contribute entries to the live turn path
    mark accepted source entries consumed
    update T.last_processed_event_timestamp
    update T.last_processed_reactive_event_timestamp
    set T.consumer.status = active
    set T.consumer.status_at = now
  else:
    leave fetched entries unconsumed in the lane
unlock(T)
```

ReAct close gate compares the timeline render cursor with `T`:

```text
lane state:
  T.last_processed_event_timestamp
  T.last_processed_event_id

timeline:
  timeline.last_rendered_event_cursor.timestamp
  timeline.last_rendered_event_cursor.event_id

if timeline.last_rendered_event_cursor is older than T.last_processed_event_*:
  keep T.handler.status = open
  ReAct continues
else:
  set T.handler.status = closed
  set T.handler.status_at = now
```

The cursor is committed on the timeline after prompt rendering succeeds. It
means the rendered model context included timeline content produced from events
up to that event timestamp/id. The model may have seen a compacted
representation rather than raw event text; that still counts because the event
was processed into the rendered context. The cursor is stored on the timeline so
in-turn compaction does not lose the progress marker.

Turn finalization runs after the handler is closed. A successful close gate
stops the live reader immediately: a closed handler cannot accept another lane
event. After artifacts persist, the Reader/Consumer is released:

```text
T.consumer.status = none
T.consumer.status_at = now
```

ContextBrowser then publishes one liveness wake when unprocessed, promotable
reactive lane work remains. Every reactive event already received an atomic
ingress wake; this post-save wake is a duplicate safety signal and must observe
the released reservation. An unconsumed `event.user.steer` is terminalized
instead: it is bound to the turn that was active at ingress and cannot become a
later turn.

Run-to-completion apps use the same door but do not open a live handler. The
processor wake reserves `T.consumer.status = scheduled`; the app executes once;
then the shared door finalizer accounts for the start batch and releases the
reservation. The wake names one accepted occurrence, so a same-ingress start
batch may be larger than the wake: for example, sequence 1 is
`event.user.prompt` and sequence 2 is `event.user.attachment.file`, both sharing
one `batch_id`. A hosted-runtime batch fold is read-only while the app runs, but
it records the exact folded lane event ids on turn state. Finalization consumes
the wake occurrence with the existing cursor primitive, then terminalizes folded
same-batch siblings by exact event id before `mark_consumer_none()`. It must not
consume by sequence range: different-`batch_id` work can interleave between
same-batch siblings and must remain pending for the next turn.

This is still the July release-before-handoff contract: consume the accepted
start batch, release `T.consumer` to `none`, then publish any duplicate liveness
wake for remaining work. Same-batch siblings are not later work. A true mid-turn
followup is outside the folded start batch, remains unconsumed, and may be
re-woken after release. Native ReAct is unaffected by this folded-batch handoff:
ReAct advances its own lane state inside `BaseWorkflow`, leaves the reservation
released and the event accounted, and the shared finalizer returns as a state
no-op.

A subagent completion (`subagent.converged` / `subagent.failed`) rides this same
lane as a promotable event: a live parent turn folds it and the promoter acks,
and only an idle lane promotes it into a parent continuation turn — the
promote-only-if-unconsumed rule this table enforces. See
[Subagent Participant Protocol](../solutions/chat/subagent-participant-protocol-README.md).

## Freshness

Freshness is calculated from table values and configured TTLs:

```text
active_is_fresh =
  T.consumer.status == active
  and now - T.consumer.status_at <= event_bus.consumer.active_ttl_ms

scheduled_is_fresh =
  T.consumer.status == scheduled
  and now - T.consumer.status_at <= event_bus.consumer.scheduled_ttl_ms
```

`T.consumer.status_at` is written by the lane Consumer or proc. Processor
heartbeat can help diagnose long-running work, but it does not replace this
lane-local acknowledgement.

A fresh `active` acknowledgement suppresses a wake only while
`T.handler.status == open`. `handler=closed, consumer=active` is a recoverable
finalization residue: no reader can consume through a closed gate, so proc may
replace it with a new `scheduled` reservation.

## SDK Primitive

The isolated state/orchestrator primitive lives under:

```text
kdcube_ai_app.apps.chat.sdk.events.event_bus
```

It provides:

```python
from kdcube_ai_app.apps.chat.sdk.events.event_bus import (
    ConversationEventBusOrchestrator,
    RedisEventLaneStateTable,
)

table = RedisEventLaneStateTable(redis=redis, state_key=state_key)
orchestrator = ConversationEventBusOrchestrator(table=table)
```

Core operations:

| Operation | Writer role |
| --- | --- |
| `schedule_consumer_from_wake(wake_event_timestamp=...)` | Proc after reading a wake item. |
| `open_handler(turn_id=...)` | BaseWorkflow / ContextBrowser handler setup before timeline load. |
| `mark_consumer_active(turn_id=...)` | Reader/Consumer activation or active acknowledgement. |
| `accept_events_for_open_handler(events, turn_id=..., accept=...)` | Reader/Consumer lane drain. |
| `try_close_handler(turn_id=..., handler_processed_event_timestamp=...)` | ReAct handler close gate. |
| `mark_consumer_none()` | Reader/Consumer release after turn finalization. |

Each lock holder records its operation, process, task, and acquisition time in
the lock value. A timeout reports those fields plus the remaining lock TTL,
without exposing the random ownership token. Cancellation waits for an in-flight
Redis lock command to reach a known result and removes the exact token if Redis
accepted it. A processor timeout before task execution is transient and is
requeued; it is not acknowledged as malformed input.

Wake publication lives next to the state primitive. For initial reactive
ingress the publisher is used inside the atomic lane-publish/queue-enqueue
operation. For post-save handoff, a promotable event already exists in the lane,
so the publisher only enqueues the wake. Active-turn controls do not take this
path:

```python
from kdcube_ai_app.apps.chat.sdk.events.event_bus import (
    EventLaneWakePublisher,
    RedisEventLaneWakeEnqueuer,
)

publisher = EventLaneWakePublisher(
    RedisEventLaneWakeEnqueuer(redis=redis, tenant=tenant, project=project)
)
result = await publisher.publish_for_event(
    payload=payload,
    event=event,
    tenant=tenant,
    project=project,
    user_id=user_id,
    conversation_id=conversation_id,
    agent_id=agent_id,
    reason="reactive_event",
)
```

The Redis enqueuer writes the wake to the normal processor ready queue for the
tenant/project/user type. Proc resolves wake items and schedules/ignores them;
it does not scan the lane after task completion.

Focused simulator and tests:

```text
app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/tests/event-bus-simulator-README.md
app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/tests/test_event_bus_state.py
```
