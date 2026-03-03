---
id: ks:docs/sdk/tools/custom-tools-README.md
title: "Custom Tools"
summary: "Define and register bundle‑local tools, including runtime selection and descriptors."
tags: ["sdk", "tools", "custom", "bundle", "runtime", "python"]
keywords: ["tools_descriptor", "tool id", "local_tools.py", "runtime mode", "in‑proc", "docker", "fargate"]
see_also:
  - ks:docs/sdk/tools/tool-subsystem-README.md
  - ks:docs/sdk/bundle/bundle-index-README.md
  - ks:docs/sdk/tools/mcp-README.md
---
# Custom Tools (Bundle‑Local)

This guide explains how to define **bundle‑local tools**, register them, and control
their runtime (in‑proc/local/docker/fargate).

For full subsystem details, see:
- `docs/sdk/tools/tool-subsystem-README.md`
- `docs/sdk/tools/mcp-README.md`

---

## 1) Write a tool module

Tools are just async Python functions. Use `@kernel_function` so the runtime can
introspect names/descriptions/params.

Example (bundle‑local):
```python
# tools/local_tools.py
from typing import Annotated, Optional
import semantic_kernel as sk
try:
    from semantic_kernel.functions import kernel_function
except Exception:
    from semantic_kernel.utils.function_decorator import kernel_function

@kernel_function(
    name="search",
    description="Search the KDCube site (kdcube.tech) for product info."
)
async def search(
    query: Annotated[str, "Search query"],
    n: Annotated[int, "Max results"] = 5,
):
    ...
```

Tool IDs are formed as:
```
<alias>.<function_name>
```

---

## 2) Register tools in `tools_descriptor.py`

Use `TOOLS_SPECS` to declare tool modules (SDK or bundle‑local):

```python
TOOLS_SPECS = [
    # SDK module
    {"module": "kdcube_ai_app.apps.chat.sdk.tools.web_tools", "alias": "web_tools", "use_sk": True},

    # bundle‑local module (relative to bundle root)
    {"ref": "tools/local_tools.py", "alias": "doc", "use_sk": True},
]
```

Notes:
- `module` = importable Python module path.
- `ref` = file path relative to bundle root.
- `alias` defines the tool id prefix.
- `use_sk=True` tells the runtime to use semantic‑kernel metadata.

---

## 3) Set runtime per tool (optional)

Use `TOOL_RUNTIME` to route specific tools to isolated runtimes:

```python
TOOL_RUNTIME = {
    "doc.search": "local",        # subprocess
    "web_tools.web_search": "local",
    "rendering_tools.write_pdf": "docker",
    "exec_tools.python_exec": "fargate",
}
```

Valid values: `none`, `local`, `docker`, `fargate`

---

## 4) MCP tools (optional)

Use `MCP_TOOL_SPECS` to expose tools from MCP servers:

```python
MCP_TOOL_SPECS = [
    {"server_id": "web_search", "alias": "web_search", "tools": ["web_search"]},
    {"server_id": "docs", "alias": "docs", "tools": ["*"]},
]
```

---

## 5) Using tools in generated code (ISO runtime)

Generated code must **not** import built‑in tool modules directly.  
It should call tools via `agent_io_tools.tool_call(...)`.

Example:
```python
resp = await agent_io_tools.tool_call(
    fn=rendering_tools.write_pdf,
    params={"path": "report.pdf", "content": html},
    call_reason="Render PDF",
    tool_id="rendering_tools.write_pdf",
)
```

See:
- `apps/chat/sdk/skills/instructions/shared_instructions.py` (`ISO_TOOL_EXECUTION_INSTRUCTION`)

---

## 6) Skills → tools mapping (optional hint system)

This is **optional**. Use it to give the agent **contextual hints**:
- When a skill is loaded via `react.read`, the skill payload includes the tool hints from `tools.yaml`.
- The agent can then treat those tools as **recommended** for the current task.
- You can also mention **recommended skills** in the tool description or docstring so the tool catalog itself suggests which skills to load.

Example `tools.yaml` inside a skill:
```yaml
tools:
  - id: doc.search
    role: product discovery
    why: Search the product site for authoritative details
```

This is **not required** for tools to work; it only improves tool selection.

---

## References (code)

- Example bundle: `services/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/with-isoruntime@2026-02-16-14-00`
- Descriptor: `services/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/with-isoruntime@2026-02-16-14-00/tools_descriptor.py`
- Local tools: `services/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/with-isoruntime@2026-02-16-14-00/tools/local_tools.py`
- Product search tool: `services/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/react@2026-02-10-02-44/tools/local_tools.py`
