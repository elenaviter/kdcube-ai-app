---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/events/conversation-event-lane-state-README.md
title: "Conversation Event Lane State"
summary: "SDK/runtime reference for the Redis state record that synchronizes conversation external-event ingress, live handlers, owned run-to-completion reservations, and wake handoff."
status: active
tags: ["sdk", "events", "external-events", "redis", "react", "synchronization"]
updated_at: 2026-08-18
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

T.consumer.turn_id
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
| `T.consumer.turn_id` (`consumer_turn_id`) | Proc and Reader/Consumer while holding `lock(T)` | Runtime turn that owns the current `scheduled` or `active` reservation. Ownership fences heartbeat and release. |
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
T.consumer.turn_id = wake routing turn id
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
T.consumer.turn_id = handler runtime turn id
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
    set T.consumer.turn_id = handler runtime turn id
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
T.consumer.turn_id = ""
```

ContextBrowser then publishes one liveness wake when unprocessed, promotable
reactive lane work remains. Every reactive event already received an atomic
ingress wake; this post-save wake is a duplicate safety signal and must observe
the released reservation. An unconsumed `event.user.steer` is terminalized
instead: it is bound to the turn that was active at ingress and cannot become a
later turn.

Run-to-completion apps use the same door but do not open a live handler. The
processor wake reserves `T.consumer.status = scheduled` for
`T.consumer.turn_id`. Before the foreign runtime starts, its adapter reads the
lane without consuming it and folds the **whole pending snapshot**: the wake
occurrence, same-ingress attachments and context, and any still-pending messages
queued while the previous turn ran. Consumed, promoted, and failed occurrences
stay out; selected occurrences are sequence ordered and their exact lane message
ids are recorded on turn state. A single fold may therefore span several
`batch_id` values.

While the foreign runtime runs, its lane watcher remains read-only. It calls
`heartbeat_scheduled_consumer(turn_id=...)`, which refreshes `status_at` only
when both the `scheduled` status and reservation owner match. It does not open a
handler, mark the consumer `active`, or fold arriving content. LangGraph may use
the same watch to cancel its stream on steer; Claude Code may deliver control at
its own tool boundary. Content not proven delivered remains pending.

The shared door finalizer consumes the wake occurrence with the existing cursor
primitive, terminalizes every event in the start snapshot by exact message id,
and calls `mark_consumer_none(turn_id=...)`. Events arriving after the snapshot
remain pending. After release it emits at most one liveness wake for eligible
reactive intent after the last steer. A bare steer is terminalized; textual steer
stays pending but does not itself start a turn. The native ReAct agent is unaffected:
`BaseWorkflow` already leaves the reservation released and the event accounted,
so the shared finalizer is a state-based no-op.

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

`T.consumer.status_at` is written by proc, the live lane Consumer, or the
matching foreign-runtime reservation watcher. Processor heartbeat can help
diagnose long-running work, but it does not replace this lane-local
acknowledgement.

**A long turn must keep writing it through the primitive for its ownership
model.** The native ReAct agent calls `mark_consumer_active` while its handler is open. A
foreign runtime calls `heartbeat_scheduled_consumer(turn_id=...)`; the update is
ignored unless that same turn still owns a `scheduled` reservation, and the
status remains `scheduled`. Before this heartbeat existed, a hosted turn's
reservation went stale after 30 seconds while the turn itself ran for minutes,
allowing a later message to start another turn. A worker that dies stops
heartbeating, so TTL expiry remains the self-healing path.

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
| `schedule_consumer_from_wake(wake_event_timestamp=..., turn_id=...)` | Proc after reading a wake item; records the reservation owner. |
| `open_handler(turn_id=...)` | BaseWorkflow / ContextBrowser handler setup before timeline load. |
| `mark_consumer_active(turn_id=...)` | Reader/Consumer activation or active acknowledgement. |
| `heartbeat_scheduled_consumer(turn_id=...)` | Foreign-runtime watcher; refreshes only its matching scheduled reservation. |
| `accept_events_for_open_handler(events, turn_id=..., accept=...)` | Reader/Consumer lane drain. |
| `try_close_handler(turn_id=..., handler_processed_event_timestamp=...)` | ReAct handler close gate. |
| `mark_consumer_none(turn_id=...)` | Owner-fenced release after turn finalization. |

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
