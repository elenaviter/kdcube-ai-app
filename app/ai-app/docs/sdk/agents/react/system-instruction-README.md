---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/react/system-instruction-README.md
title: "React System Instruction"
summary: "How React decision system instructions are composed and cached, including the always-present harness context, selectable instruction bodies, and appended admin customization."
tags: ["sdk", "agents", "react", "instructions", "system-prompt", "lite", "configuration"]
keywords: ["React system instruction", "React lite instructions", "instruction_body", "instruction_blocks", "default_lite_system_instruction", "React prompt composition"]
updated_at: 2026-08-18
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/tools/tool-catalog-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/skills/instruction-blocks-and-signals-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/react/context-caching-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/react/react-round-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/react/react-tools-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/react/context-layout.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/react/context-progression.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/react/micro-agents-and-cache-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/react/react-announce-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/react/shared-timeline-event-bus-steer-followup-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/react/memory-recovery-path-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/react/external-exec-README.md
---
# React System Instruction

This page explains the system instruction seen by the React decision agent.
It is also the checklist for deciding whether a Lite or explicit complete instruction body is
complete for the tools and runtime surfaces exposed by a bundle.

## Composition

The full decision system text is assembled in
`kdcube_ai_app.apps.chat.sdk.solutions.react.decision_prompt`.

```text
v2/v3 decision agent
  -> version-specific strict channel protocol
  -> always-present tool-availability signal
  -> always-present React harness context and artifact-access model
  -> selected instruction body
       1. instruction_body, if supplied
       2. composed instruction_blocks, if supplied
       3. extended/default body, otherwise
  -> optional tool catalog
  -> optional skill catalog
  -> optional appended agent-admin customization block
```

The strict channel protocol, tool-availability signal, and harness
context/artifact-access block are always present. The runtime parser depends on
the protocol. `REACT_HARNESS_TOOL_AVAILABILITY` tells the agent that the
current catalogs are the authority for configured capabilities, including
optional tool families. `REACT_HARNESS_CONTEXT_AND_ARTIFACT_ACCESS` explains
the model input shape, the universal artifact-URI model, distributed turn
locality, the current-turn pull precondition for materializable artifacts,
namespace-based access, and shape-driven inspection. Both blocks are inserted
by `compose_decision_system_text(...)` before the selected body, so Full, Lite,
Extra Lite, and an explicit complete body receive the same harness facts.

`instruction_body` is the explicit complete-body selector shown in the diagram;
it does not mean the appended agent-admin customization. Admin customization is
passed as `additional_instructions` and wrapped by
`append_agent_admin_customization(...)` after the standard body and catalogs.

The protocol teaches the BASE channels (`thinking`, `action`, optional
`summary`) and states that a connected tool may extend the protocol with a
channel of its own. The `<channel:code>` teaching is such an extension: it
renders — together with the exec instruction blocks, the exec sections of
the extended body, and the code-snippet-only tool catalog section — only
when `exec_tools.execute_code_python` is in the effective adapter roster.
The effective roster is the single authority for capability teaching: an
agent whose exec tool is off receives an instruction with no
code-execution teaching anywhere (protocol, body, or catalogs).

The tool catalog and skill catalog are not substitutes for the instruction
body. They tell the model what is currently available. The instruction body
explains how to behave with the React timeline, ANNOUNCE, logical paths,
workspace, recovery paths, tools, memory, and finalization. How the tool
catalog itself is composed — its sections, the full/compact forms, and the
configuration behind them — is owned by
[the tool-catalog doc](../../tools/tool-catalog-README.md).

### Cache Consequence

The composed system instruction is part of the exact model-input prefix. It is
not a timeline block, but it still affects the prompt prefix that React sends
before the rendered timeline.

```
[strict protocol]
[tool availability]
[always-present harness context and artifact access]
[selected instruction body]
[tool/skill catalog]
[appended admin or runtime customization]
[rendered timeline prefix]
[ANNOUNCE / current tail]
```

Changing any bytes before the rendered timeline creates a different downstream
prompt prefix. In the current ReAct decision path, the tool catalog and skill
catalog are text rendered in this system instruction, usually near the bottom.
Changing them changes the system prefix before the rendered timeline. A
per-user customization suffix therefore prevents cross-user cache sharing for
that agent after that suffix. A subagent with a different instruction has its
own cache story. Put volatile current state in ANNOUNCE or another tail block
instead of appending it to the instruction body.

React maps this prefix layout to explicit cache controls for Anthropic/Claude.
For other providers, the same prompt layout still controls token shape and
semantic stability, but React does not currently assume equivalent
provider-side cache-control behavior.

The important boundary is the first changed segment:

```
[strict protocol]                         stable for all compatible agents
[tool availability]                       stable for all compatible agents
[harness context and artifact access]     stable for all compatible agents
[default React instruction]               stable for all compatible agents
[shared bundle/domain instruction]        stable for this bundle/config

--- first variable segment below this line limits cache sharing ---

[per-user instruction suffix]             differs by user
[selected tool catalog]                   differs if user/runtime changes tools
[selected skill catalog]                  differs if user/runtime changes skills
[rendered timeline prefix]                downstream of the variable segments
[ANNOUNCE / current tail]                 intentionally uncached
```

If the tool catalog or skill catalog is rendered inside the instruction
envelope, it is part of the cache prefix. Letting a user select tools or skills
therefore partitions the cache by that exact selection. Changing the selection
between turns invalidates the downstream cache for that user. Changing it
between rounds is worse: the same turn can no longer reuse cache points after
the changed catalog segment.

Keep the instruction envelope stable when cache sharing matters. Put changing
current state in ANNOUNCE. Only put tool/skill catalogs in the instruction
envelope when the model really needs that catalog for the current call, and
expect the cache to be keyed by the exact catalog text.

Be explicit about the reuse scope:

- Same user reuse: later turns, later rounds, or repeated calls to the same
  configured subagent can reuse cache only while that user's instruction/catalog
  prefix stays identical.
- Cross-user reuse: multiple users can share the common prefix only before any
  per-user instruction, per-user data, or user-selected catalog segment. This is
  valuable on Anthropic because traffic from multiple users is more likely to
  keep a short-lived cache entry hot.
- Subagent reuse: a subagent configured dynamically by the main agent is usually
  a cache story inside that user's work. A ready-made/static subagent can share
  its common prefix across users until the first user-specific segment.

## Use The Extended Default

If a bundle does not pass body-selection fields, React uses the extended
default body from `shared_instructions.py`.

```python
tool_config = agent_tool_config_from_bundle_props(
    self.bundle_props,
    "main",
    bundle_root=BUNDLE_ROOT,
)
react = self.build_react(
    scratchpad=scratchpad,
    mod_tools_spec=tool_config.tool_specs,
)
```

This is the broadest current instruction set. It is the right default for
general-purpose, full-capability agents.

## Use A Lite Profile

Lite instruction blocks live in `shared_instructions_lite.py`. The helper below
returns an instruction body only. The version-specific protocol, tool catalog,
skill catalog, and admin customization are still added by React.

```python
from kdcube_ai_app.apps.chat.sdk.skills.instructions.shared_instructions_lite import (
    default_lite_system_instruction,
)

tool_config = agent_tool_config_from_bundle_props(
    self.bundle_props,
    "main",
    bundle_root=BUNDLE_ROOT,
)
react = self.build_react(
    scratchpad=scratchpad,
    mod_tools_spec=tool_config.tool_specs,
    instruction_body=default_lite_system_instruction("workspace_exec"),
    include_tool_catalog=True,
    include_skill_gallery=True,
)
```

Available profiles:

| Profile | Intended Use |
| --- | --- |
| `core` | Minimal React operation: timeline, ANNOUNCE, live events, paths, read recovery, workspace model, skills, attachments, citations, finalization. |
| `workspace` | Core plus common React workspace tools: write, memsearch, rg, pull/checkout, patch, and plan. |
| `workspace_exec` | Workspace plus isolated exec guidance. |
| `document` | Workspace plus rendering-tool guidance. |
| `web` | Workspace plus web search/fetch guidance. |
| `all_capabilities` | All lite blocks, including internal notes and durable user memory. Use only when those tools and policies are enabled. |

### Exec Artifact Output Paths

Every built-in instruction tier that exposes `exec_tools.execute_code_python`
teaches the same two-part path contract:

1. The action's `contract[].filepath` is an `OUTPUT_DIR`-relative string such
   as `turn_<current>/files/report/report.xlsx`. Keep this `artifact_rel`
   value relative.
2. Generated code resolves that value under the runtime artifact root, creates
   its parent, and passes the resolved path to the writer:

```python
from pathlib import Path

artifact_rel = "turn_<current>/files/report/report.xlsx"
artifact_path = Path(OUTPUT_DIR) / artifact_rel
artifact_path.parent.mkdir(parents=True, exist_ok=True)
wb.save(artifact_path)
```

The contract `filepath` must equal `artifact_rel` byte-for-byte. It is not the
path passed directly to `open()`, `wb.save()`, or another writer. Writing
`artifact_rel` directly would create a bare `turn_<current>/...` tree relative
to the process working directory, where the artifact collector will not find
the file.

This rule lives in the full body's `CODEGEN_BEST_PRACTICES_V2` and
`EXEC_SNIPPET_RULES`, the moderate `REACT_LITE_EXEC_TOOL`, and the extra-lite
`REACT_XLITE_EXEC` block. The exec tool's parameter description repeats it, so
the rule also survives the compact tool catalog. Every tier is
roster-conditional: Lite and Extra-lite profiles without an effective exec
tool omit their exec blocks, and the legacy Full body drops its exec
sections (`CODEGEN_BEST_PRACTICES_V2`, `EXEC_SNIPPET_RULES`, the exec
subsection of the paths guide) the same way. Generic surfaces — the react
tool docs, the protocol, paths/workspace guidance — speak in class rules
(file-processing tools, computation tools, capped tool outputs) and never
name the exec tool; everything exec-specific renders only with the tool
present. An explicit complete body that hides the tool catalog must carry its own
complete exec contract.

Exec code itself is preserved as a Python module body and evaluated with
top-level `await` enabled. Do not generate an event-loop runner or a separate
`main()` entrypoint.


## The Blocks Vocabulary, Signals, And Completeness

The `blocks` vocabulary, the signal table for every default block, the
completeness checklist, and the audit method for Lite/explicit complete bodies live in
[the instruction-blocks-and-signals doc](../../skills/instruction-blocks-and-signals-README.md)
— they are agent-neutral and serve every harness that composes an
instruction body from blocks. This page keeps what is React-specific: the
composition order, the cache consequences, and the build calls above.
