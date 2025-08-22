# Agentic App Bundle — Integration Guide

This doc shows how to plug **your** agentic app (LangGraph/LangChain/etc.) into the host chat service by shipping a **bundle**. You’ll see:

* The required **directory layout**
* The **decorators** that mark your workflow for the loader
* The **interface contract** your workflow should implement
* The **env vars** you must set
* How to **stream**, emit **step** updates, and **serialize** results
* A minimal **Hello World** bundle you can copy

> **Important:** The **decorator is the only way** to mark a workflow (class or factory). If nothing in your module is decorated, the loader won’t find you.

---

## 1) What’s a “bundle”?

A bundle is a small Python package the host can import dynamically. It contains your workflow class (or factory), optional integrations (RAG, tools, etc.), and any local services you need.

**Example layout:**

```
my_cool_bundle/
├── __init__.py
├── agentic_app.py        # ← your decorated workflow lives here
├── inventory.py          # optional local service/model glue
└── integrations/
    ├── __init__.py
    └── rag.py            # optional RAG integration
```

---

## 2) How the loader finds you

The host sets these **env vars**:

* `AGENTIC_BUNDLE_PATH`
  Absolute filesystem path to the **root folder** of your bundle (the dir containing `__init__.py`).

* `AGENTIC_BUNDLE_MODULE`
  The **import path inside that folder** that contains the decorated workflow/factory.

    * If your workflow is in `agentic_app.py` at the top level, set: `AGENTIC_BUNDLE_MODULE=agentic_app`.
    * If it’s in a subpackage, e.g. `app_backend/agent.py`, set: `AGENTIC_BUNDLE_MODULE=app_backend.agent`.

* `AGENTIC_SINGLETON`
  “true”/“false” (case-insensitive). Overrides bundle’s default and forces the loader to keep **one instance per process** (true) or **new per request** (false), when supported by your decorator options.

> If the loader logs:
> **“No decorated workflow found in module 'X'. Use @agentic\_workflow or @agentic\_workflow\_factory.”**
> Check that `AGENTIC_BUNDLE_PATH` and `AGENTIC_BUNDLE_MODULE` point to the right place **and** your item is decorated.

---

## 3) The decorators (required)

Import the decorators from the host:

```py
from kdcube_ai_app.infra.plugin.agentic_loader import (
    agentic_workflow,
    agentic_workflow_factory,
    agentic_initial_state,
)
```

### 3.1 `@agentic_workflow` (decorate a **class**)

```py
@agentic_workflow(
    name="chat-workflow",   # unique ID within your module
    version="1.0.0",
    priority=150,                # higher wins if multiple are present
    singleton=False              # preferred default (can be overridden via env)
)
class MyWorkflow:
    ...
```

### 3.2 `@agentic_workflow_factory` (decorate a **factory function**)

Use this if you prefer to build/return an instance yourself.

```py
@agentic_workflow_factory(
    name="my-workflow-factory",
    version="1.0.0",
    priority=200,
    singleton=True
)
def create_workflow(config, step_emitter=None, delta_emitter=None):
    return MyWorkflow(config, step_emitter, delta_emitter)
```

> You can use **either** class **or** factory. The loader will pick the highest-priority decorated item it finds.

### 3.3 `@agentic_initial_state` (optional)

Provide a function the host can call to make your first state dict.

```py
@agentic_initial_state(name="my-initial-state", priority=100)
def create_initial_state(user_message: str) -> dict:
    return {
        "context": {"bundle": "my_cool_bundle"},
        "user_message": user_message,
        "is_our_domain": None,
        "classification_reasoning": None,
        "rag_queries": None,
        "retrieved_docs": None,
        "reranked_docs": None,
        "final_answer": None,
        "error_message": None,
        "format_fix_attempts": 0,
        "search_hits": None,
        "execution_id": f"exec_{int(time.time()*1000)}",
        "start_time": time.time(),
        "step_logs": [],
        "performance_metrics": {},
    }
```

---

## 4) Workflow interface contract

Your decorated workflow **must** be constructible with:

```py
__init__(self,
         config,                        # host Config object
         step_emitter: Optional[StepEmitter] = None,
         delta_emitter: Optional[DeltaEmitter] = None)
```

…and should provide at least **one** of these async entry points:

* **Preferred:**

  ```py
  async def process_message(
      self,
      user_message: str,
      thread_id: str = "default",
      seed_messages: Optional[List[Any]] = None
  ) -> Dict[str, Any]
  ```

  Return a **JSON-serializable dict** (see §6).

* **Optional low-level:**

  ```py
  async def run(self, session_id: str, state: dict) -> dict
  ```

  Useful if you expose your graph directly.

> The host may inject:
>
> * `delta_emitter(chunk: str, idx: int)` for **streaming tokens**
> * `step_emitter(step_name: str, status: Literal["started","completed","error"], payload: dict)` for **step timeline**

---

## 5) Emitting streaming & steps (for the host UI)

### 5.1 Streaming tokens

Call the `delta_emitter` whenever you get a chunk:

```py
idx = -1
async def on_token(txt: str):
    nonlocal idx
    if not txt: return
    idx += 1
    await delta_emitter(txt, idx)
```

### 5.2 Step events

Wrap your nodes/operations and emit:

```py
await step_emitter("query_writer", "started", {"message": "Generating RAG queries..."})
# ...do work...
await step_emitter("query_writer", "completed", {"query_count": 5, "queries": ["..."]})
# or, on error:
await step_emitter("query_writer", "error", {"error": str(e)})
```

**Common step names** used by the UI (suggested):
`workflow_start`, `classifier`, `query_writer`, `rag_retrieval`, `reranking`, `answer_generator`, `workflow_complete`

---

## 6) What you should return (JSON-serializable)

Return a plain dict (no LangChain objects) containing at least:

* `final_answer: str` — what the UI shows
* `error_message: Optional[str]` — if something failed

Recommended optional fields the UI knows how to use:

* `is_our_domain: Optional[bool]`
* `classification_reasoning: Optional[str]`
* `retrieved_docs: Optional[List[Dict]]`
* `reranked_docs: Optional[List[Dict]]`
* `step_logs: List[Dict]` — your own per-step timeline
* `performance_metrics: Dict[str, Any]`
* `execution_id: str`, `start_time: float`

> If you run a LangGraph and receive a `GraphState`/rich object back, **serialize** it by picking the above fields (and anything else your app cares about) and converting non-JSON types to primitives/strings.

---

## 7) Configuration your bundle can rely on

The host passes a `Config` object into your workflow constructor. Typical fields (check your host app):

* `selected_model` (e.g., `"gpt-4o"`)
* `openai_api_key`, `claude_api_key`, …
* `embedding_model`, `custom_embedding_endpoint`
* `kb_search_endpoint` (if you use RAG)
* `log_level`, `provider`, etc.

Your bundle can import types/utilities from the host, e.g.:

```py
from kdcube_ai_app.apps.chat.inventory import Config, AgentLogger, _mid
from kdcube_ai_app.apps.chat.emitters import StepEmitter, DeltaEmitter
```

---

## 8) Debugging your bundle

If the API exposes a debug endpoint (e.g. `GET /landing/debug/agentic`), it will show:

* Which module was loaded
* Which decorated workflow/factory was selected
* Whether singleton mode is active
* Any import or decoration errors

**Typical pitfalls:**

* Missing `__init__.py` (module not importable)
* Wrong `AGENTIC_BUNDLE_MODULE` (points to a package, not the file containing the decorator)
* Decorator not executed (guarded by `if __name__ == "__main__":` or code exceptions on import)
* Returning non-serializable objects in your result

---

## 9) Minimal bundle you can copy

### `my_cool_bundle/__init__.py`

```py
# Keep it empty or put package version here
```

### `my_cool_bundle/agentic_app.py`

```py
import time
from typing import Optional, List, Dict, Any

from kdcube_ai_app.infra.plugin.agentic_loader import (
    agentic_workflow,
    agentic_initial_state,
)
from kdcube_ai_app.apps.chat.emitters import StepEmitter, DeltaEmitter
from kdcube_ai_app.apps.chat.inventory import Config

BUNDLE_ID = "my_cool_bundle"

@agentic_initial_state(name="my-initial-state", priority=100)
def create_initial_state(user_message: str) -> dict:
    return {
        "context": {"bundle": BUNDLE_ID},
        "user_message": user_message,
        "is_our_domain": None,
        "classification_reasoning": None,
        "rag_queries": None,
        "retrieved_docs": None,
        "reranked_docs": None,
        "final_answer": None,
        "error_message": None,
        "format_fix_attempts": 0,
        "search_hits": None,
        "execution_id": f"exec_{int(time.time()*1000)}",
        "start_time": time.time(),
        "step_logs": [],
        "performance_metrics": {},
    }

@agentic_workflow(name="hello-workflow", version="1.0.0", priority=100, singleton=False)
class HelloWorkflow:
    def __init__(self, config: Config,
                 step_emitter: Optional[StepEmitter] = None,
                 delta_emitter: Optional[DeltaEmitter] = None):
        self.config = config
        self.emit_step = step_emitter or (lambda *_a, **_k: None)
        self.emit_delta = delta_emitter or (lambda *_a, **_k: None)

    async def process_message(self, user_message: str, thread_id: str = "default",
                              seed_messages: Optional[List[Any]] = None) -> Dict[str, Any]:
        await self.emit_step("workflow_start", "started", {"message": "Starting..."})

        # Simulate streaming
        answer = "Hello! You said: "
        idx = -1
        for token in ["Hello", ", ", "world", "!"]:
            idx += 1
            await self.emit_delta(token, idx)
            answer += token

        await self.emit_step("answer_generator", "completed", {"answer_length": len(answer)})
        await self.emit_step("workflow_complete", "completed", {"message": "Done"})

        return {
            "final_answer": answer,
            "is_our_domain": True,
            "retrieved_docs": [],
            "reranked_docs": [],
            "error_message": None,
            "step_logs": [],
            "execution_id": f"exec_{int(time.time()*1000)}",
            "performance_metrics": {},
        }
```

### `my_cool_bundle/inventory.py` (optional)

```py
# Put your local model/service glue here (or leave empty if not needed)
```

### `my_cool_bundle/integrations/rag.py` (optional)

```py
class RAGService:
    def __init__(self, config):
        self.endpoint = getattr(config, "kb_search_endpoint", None)

    async def retrieve_documents(self, queries):
        # Replace with your real KB call
        return []
```

---

## 10) Configure the host

Set env vars so the host can import your bundle:

```bash
export AGENTIC_BUNDLE_PATH=/absolute/path/to/my_cool_bundle
export AGENTIC_BUNDLE_MODULE=agentic_app
export AGENTIC_SINGLETON=false

# plus any model keys your workflow needs:
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

Restart the host service and hit your chat endpoint. If the UI supports the debug route, check `/landing/debug/agentic`.

---

## 11) FAQ

**Q: Can I decorate my class instead of a factory?**
A: Yes. Use `@agentic_workflow` on the class (see §3.1). That’s the simplest path.

**Q: What if I have multiple decorated items?**
A: The loader picks the **highest priority**. Use `priority` to control the winner.

**Q: My state contains LangChain messages and crashes JSON serialization.**
A: Don’t return raw graph state. Return a **clean dict** with primitives only (see §6). Keep your own `step_logs` and metrics if useful.

**Q: Do I need `__init__.py`?**
A: Yes, your bundle root must be a Python package.

