---
title: "OpenRouter Data Bundle — Entrypoint"
summary: "Deep-dive into OpenRouterDataBundle: LangGraph wiring, configuration property, model resolution, accounting integration, and how to extend the bundle."
tags: ["bundle", "openrouter", "entrypoint", "langgraph", "accounting"]
keywords: ["OpenRouterDataBundle", "BaseEntrypoint", "agentic_workflow", "BundleState", "execute_core", "configuration", "role_models", "data-processor", "openrouter_completion", "with_accounting"]
see_also:
  - ks:docs/sdk/bundle/openRouter/openrouter-data-README.md
  - ks:docs/sdk/bundle/bundle-dev-README.md
---

# OpenRouter Data Bundle — Entrypoint

This document covers the internals of `entrypoint.py` and `orchestrator/workflow.py` for
the `openrouter-data@2026-03-11` bundle. Read the [overview](openrouter-data-README.md) first
if you just want to use or configure the bundle.

## Class: `OpenRouterDataBundle`

```python
@agentic_workflow(name="openrouter-data", version="1.0.0", priority=50)
class OpenRouterDataBundle(BaseEntrypoint):
```

Inherits from `BaseEntrypoint` (SDK base class for all bundles). The decorator registers
it with the plugin system under the name `openrouter-data`.

### Constructor

```python
def __init__(self, config, pg_pool=None, redis=None, comm_context=None)
```

Calls `super().__init__()` with a `BundleEventFilter` instance, then immediately
compiles the LangGraph by calling `_build_graph()`.

> The graph is compiled once per bundle instance. If `singleton=false` in the registry,
> a new instance (and graph) is created per request. If `singleton=true`, one graph
> is shared across requests.

### `_build_graph()` — Graph construction

Builds a minimal one-node LangGraph:

```
START → process → END
```

The `process` node:
1. Constructs an `OpenRouterDataWorkflow` instance
2. Calls `await orch.process(payload)` with fields extracted from `BundleState`
3. Writes `res["answer"]` → `state["final_answer"]` and `res["followups"]` → `state["followups"]`
4. On any exception: calls `self.report_turn_error()` (SDK helper, emits error event + logs)

### `configuration` property

Extends the parent `configuration` dict with a `data-processor` role:

```python
@property
def configuration(self) -> dict:
    config = dict(super().configuration)
    role_models = dict(config.get("role_models") or {})
    role_models["data-processor"] = {
        "provider": "openrouter",
        "model": "google/gemini-2.5-flash-preview",
    }
    config["role_models"] = role_models
    return config
```

This is the **default** — the platform merges it with the operator-supplied bundle config,
and operator values take precedence. So to change the model, override `role_models` in the
bundle registry; you do not need to modify this file.

### `execute_core()` — Graph execution

```python
async def execute_core(self, *, state, thread_id, params):
    return await self.graph.ainvoke(
        state,
        config={"configurable": {"thread_id": thread_id}},
    )
```

Called by the SDK lifecycle after the platform validates the request and populates
`BundleState`. The thread ID is used by LangGraph for checkpointing (even though this
bundle does not use multi-turn memory).

---

## Class: `OpenRouterDataWorkflow`

Located in `orchestrator/workflow.py`. Handles the actual API call.

### Constructor

```python
def __init__(self, *, comm, config, comm_context=None)
```

| Param | Type | Description |
|-------|------|-------------|
| `comm` | `ChatCommunicator` | SSE event emitter (from `BaseEntrypoint.comm`) |
| `config` | `Config` | Resolved bundle config including `role_models` |
| `comm_context` | `ChatTaskPayload` | Full task payload (optional, for future use) |

### `_resolve_model()` — Model resolution

```python
def _resolve_model(self) -> str:
    role_models = getattr(self.config, "role_models", {}) or {}
    spec = role_models.get("data-processor") or {}
    return spec.get("model") or DEFAULT_MODEL
```

Resolution order:
1. `config.role_models["data-processor"]["model"]` — operator override
2. `DEFAULT_MODEL = "google/gemini-2.5-flash-preview"` — hardcoded fallback

### `process()` — Main execution

```python
async def process(self, payload: dict) -> dict
```

Full execution sequence:

```
1. Extract user_text from payload["text"]
2. Resolve model via _resolve_model()
3. Emit step(step="processing", status="running", ...)
4. Build messages: [system_prompt, user_message]
5. with_accounting("data-processor", metadata={"openrouter_model": model}):
       result = await openrouter_completion(model, messages, temperature=0.3, max_tokens=4096)
6. If result["success"] is False:
       Emit step(status="error") → return {"answer": f"Processing failed: {error}", "followups": []}
7. Emit step(status="done", data={model, usage})
8. Return {"answer": result["text"], "followups": []}
```

#### System prompt

```
You are a precise data-processing assistant.
Follow the user's instructions exactly.
When asked to extract, classify, tag, summarize, or generate schemas,
produce clean, structured output.
Prefer JSON output when the task is structured.
```

Override this by subclassing `OpenRouterDataWorkflow` and overriding the `process()` method
or extracting `SYSTEM_PROMPT_DATA_PROCESSOR` as a constructor parameter.

#### OpenRouter call parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `temperature` | `0.3` | Low — for deterministic structured output |
| `max_tokens` | `4096` | Sufficient for most data-processing responses |
| `model` | resolved at runtime | See `_resolve_model()` |
| `messages` | `[system, user]` | Single-turn, no history |

---

## Accounting integration

Every OpenRouter call is wrapped in:

```python
with with_accounting("data-processor", metadata={"openrouter_model": model}):
    result = await openrouter_completion(...)
```

`openrouter_completion()` is itself decorated with `@track_llm`, so token usage is
captured automatically. The `with_accounting` context associates the usage with the
`"data-processor"` role and attaches the model slug as metadata for cost attribution.

---

## Event filter

`BundleEventFilter` in `event_filter.py` is a pass-through:

```python
class BundleEventFilter:
    def filter(self, event: dict) -> bool:
        return True  # forward all events
```

To suppress specific event types (e.g., internal debug steps), override `filter()`:

```python
class BundleEventFilter:
    def filter(self, event: dict) -> bool:
        # suppress internal step events, forward only final answer
        return event.get("type") != "chat.step"
```

---

## BundleState fields used

| Field | Direction | Description |
|-------|-----------|-------------|
| `request_id` | read | Passed to workflow payload |
| `tenant` / `project` | read | Passed to workflow payload |
| `user` / `user_type` | read | Passed to workflow payload |
| `session_id` / `conversation_id` / `turn_id` | read | Passed to workflow payload |
| `text` | read | User message content |
| `attachments` | read | Optional file/image attachments |
| `final_answer` | write | LLM response text |
| `followups` | write | Always `[]` in current implementation |

---

## Extending this bundle

### Change the system prompt

Subclass `OpenRouterDataWorkflow` and override `SYSTEM_PROMPT_DATA_PROCESSOR` or pass it as a constructor argument:

```python
class MyWorkflow(OpenRouterDataWorkflow):
    async def process(self, payload):
        # inject a custom system prompt before calling super
        ...
```

### Add structured output (JSON schema)

Pass a `response_format` parameter to `openrouter_completion()`. Not all OpenRouter models
support it — check the model's capabilities on the OpenRouter model catalog.

### Add follow-up suggestions

Populate `followups` in the return dict. The platform forwards them to the UI:

```python
return {
    "answer": answer,
    "followups": ["Extract as CSV instead", "Show only top 5 results"],
}
```

### Use streaming

Replace `openrouter_completion()` with `openrouter_stream()` (if available in
`kdcube_ai_app.infra.service_hub.openrouter`) and emit tokens via
`self.comm.token(chunk)` inside the stream loop.

---

## Relevant implementation files

- `kdcube_ai_app/apps/chat/sdk/examples/bundles/openrouter-data@2026-03-11/entrypoint.py`
- `kdcube_ai_app/apps/chat/sdk/examples/bundles/openrouter-data@2026-03-11/orchestrator/workflow.py`
- `kdcube_ai_app/apps/chat/sdk/examples/bundles/openrouter-data@2026-03-11/event_filter.py`
- `kdcube_ai_app/infra/service_hub/openrouter.py`
- `kdcube_ai_app/infra/accounting.py`
- `kdcube_ai_app/apps/chat/sdk/solutions/chatbot/entrypoint.py` — `BaseEntrypoint`
- `kdcube_ai_app/infra/plugin/agentic_loader.py` — `@agentic_workflow`

## Related docs

- Bundle overview: [openrouter-data-README.md](openrouter-data-README.md)
- Bundle developer guide: [bundle-dev-README.md](../bundle-dev-README.md)
- Bundle interfaces (widgets, ops): [bundle-interfaces-README.md](../bundle-interfaces-README.md)