# Using the Chat Emitter in Agent Bundles

This guide shows bundle authors how to stream tokens, post step updates, and send typed UI cards (like follow-ups) through the **emitter** in the chat runtime.
Streaming UX with minimal boilerplate.

Utility: [emitters.py](emitters.py)
Example of usage - [agentic_app.py](../../default_app/agentic_app.py).
---



## TL;DR

* Use `AIBEmitters` inside your workflow.
* Emit **steps** for progress, **deltas** for streaming, and **typed events** for UI cards.
* Typed events default-route to `chat_step` (the client already listens there).

```python
from kdcube_ai_app.apps.chat.sdk.comm.emitters import AIBEmitters, DeltaPayload, StepPayload
from kdcube_ai_app.apps.chat.emitters import ChatCommunicator

class ChatWorkflow:
    def __init__(self, config, communicator: ChatCommunicator, streaming: bool = True):
        self.emit = AIBEmitters(communicator)
```

---

## What the runtime already does (so you don’t have to)

* The processor sends `chat.start` and `chat.complete` envelopes for each turn.
* If your workflow raises, the processor emits `chat.error`.

You should emit **steps**, **deltas**, and optional **typed events** from inside your nodes.

---

## API at a glance

### Steps (progress cards)

```python
await self.emit.step(StepPayload(
    step="classifier",
    status="started",                 # "started" | "completed" | "error" | "skipped"
    title="🧭 Classifier",
))
# ... do work ...
await self.emit.step(StepPayload(
    step="classifier",
    status="completed",
    title="🧭 Classifier",
    markdown="**Result:** In scope\n\n**Confidence:** 92%",
    data={"details": {"confidence": 0.92}},
))
```

* `markdown` (if provided) is wrapped into a compose-block automatically.
* Use `data` for structured payloads you may render later.

---

### Deltas (token streaming)

```python
i = 0
await self.emit.delta(DeltaPayload(text="Let’s start with ", index=i)); i += 1
await self.emit.delta(DeltaPayload(text="light needs.", index=i)); i += 1

# Optional end markers (some UIs use them, safe to omit if not needed)
await self.emit.delta(DeltaPayload(text="", index=i, marker="answer", completed=True))
```

* `index` must be **monotonic** per stream.
* `marker` may be `"thinking"` or `"answer"`. UIs can render separate tracks.

---

### Typed events (cards/chips such as follow-ups)

```python
await self.emit.event(
    agent="answer_generator",
    type="chat.followups",
    title="Follow-ups: User Shortcuts",
    step="followups",
    status="completed",
    data={"items": ["Adjust watering schedule.", "Reposition plant to bright, indirect light."]},
)
```

* **Routing:** By default, typed events are emitted on **`chat_step`** so the existing client receives them without any UI changes.
* If you *do* set `route="chat_followups"` (or other), make sure your client listens to that socket event.

Convenience for follow-ups:

```python
await self.emit.followups(["Adjust watering schedule.", "Inspect for pests."])
```

---

## Common patterns

### 1) A full node with start/complete + streaming

```python
await self.emit.step(StepPayload(step="answer_generator", status="started", title="✍️ Compose Answer"))

idx = 0
# thinking track
await self.emit.delta(DeltaPayload(text="• Identify likely causes\n", index=idx, marker="thinking")); idx += 1
await self.emit.delta(DeltaPayload(text="• Propose fixes\n", index=idx, marker="thinking")); idx += 1
await self.emit.delta(DeltaPayload(text="", index=idx, marker="thinking", completed=True)); idx += 1

# answer track
await self.emit.delta(DeltaPayload(text="**Overwatering** can cause yellowing. ", index=idx)); idx += 1
await self.emit.delta(DeltaPayload(text="Check drainage and reduce frequency.", index=idx)); idx += 1
await self.emit.delta(DeltaPayload(text="", index=idx, marker="answer", completed=True)); idx += 1

await self.emit.step(StepPayload(step="answer_generator", status="completed",
                                 title="✍️ Compose Answer",
                                 markdown="_Answer composed._"))
```

### 2) Follow-ups as chips + an optional summary card

```python
items = ["Adjust watering schedule.", "Reposition for indirect light.", "Inspect for pests."]
await self.emit.followups(items)

# (Optional) also show a visible markdown card in the timeline:
await self.emit.step(StepPayload(
    step="followups",
    status="completed",
    title="🧠 Follow-ups",
    markdown="### Suggested next actions\n\n" + "\n".join(f"- {s}" for s in items),
))
```

---

## Envelope compatibility (client v1)

The web client listens to these socket events:

* `chat_start` / `chat_step` / `chat_delta` / `chat_complete` / `chat_error`

Your **steps** and **typed events** arrive on **`chat_step`** by default.
Your **deltas** arrive on **`chat_delta`**.

Typed events keep their semantic **`env.type`** (e.g., `"chat.followups"`) inside the envelope so the UI can branch on it.

---

## Gotchas & troubleshooting

* **“My follow-ups don’t show.”**
  You probably routed to a socket name the client doesn’t listen to. Don’t set `route`, or set `route="chat_step"`.

* **“I see duplicate content.”**
  Ensure you only stream **public** content. If you parse model output that includes a private “prelude/log”, do not emit it as deltas. Keep a clear parser state (e.g., switch modes on markers) and only emit what belongs to `"thinking"` or `"answer"` sections.

* **“Deltas arrive out of order.”**
  Use a single, monotonic `index` counter per stream.

* **“Do I need to call `chat.start` or `chat.complete`?”**
  No—handled by the runtime unless you’re writing a custom processor.

---

## Minimal integration checklist

1. Construct once in your workflow: `self.emit = AIBEmitters(communicator)`.
2. For each node:

    * `step(... started ...)`
    * work
    * `delta(...)` (optional streaming)
    * `step(... completed ...)`
3. When you have shortcuts or UI cards, use:

    * `emit.followups(items)` **or**
    * `emit.event(agent=..., type="chat.followups", ... data={"items": [...]})`
4. Don’t set `route` unless you’ve added a matching client listener.

---

## Unified sample

```python
from kdcube_ai_app.apps.chat.sdk.comm.emitters import AIBEmitters, DeltaPayload, StepPayload
from kdcube_ai_app.apps.chat.emitters import ChatCommunicator

communicator: ChatCommunicator = ...  # provided to your workflow/node
emit = AIBEmitters(communicator)

# Step card
await emit.step(StepPayload(step="classifier", status="started", title="🧭 Classifier"))

# Streaming
i = 0
await emit.delta(DeltaPayload(text="plan → ", index=i, marker="thinking")); i += 1
await emit.delta(DeltaPayload(text="answer part 1 ", index=i)); i += 1
await emit.delta(DeltaPayload(text="", index=i, marker="answer", completed=True)); i += 1

# Typed UI card (chips)
await emit.event(
  agent="answer_generator",
  type="chat.followups",
  title="Follow-ups: User Shortcuts",
  step="followups",
  status="completed",
  data={"items": ["Adjust watering schedule.", "Reposition for indirect light."]},
)
# or simply:
await emit.followups(["Adjust watering schedule.", "Reposition for indirect light."])
```