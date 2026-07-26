---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/tools/tool-catalog-README.md
title: "The Tool Catalog"
summary: "The tool catalog as a reusable concept: how an agent's effective tool roster becomes the catalog rendered into its instruction — sections, the full and compact forms, the never-cut contract rule — and how the roster conditions the instruction itself (capability blocks, protocol extensions) with the configuration that drives all of it."
tags: ["sdk", "agents", "react", "tools", "tool-catalog", "instructions", "configuration", "capabilities"]
keywords: ["tool catalog", "compact tool catalog", "tool_catalog_detail", "capability instruction exclusions", "effective roster", "channel:code", "tools configuration", "tool_traits"]
updated_at: 2026-07-26
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/skills/instruction-blocks-and-signals-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/react/system-instruction-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/react/react-tools-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/tools/tool-subsystem-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/agentic-config/agentic-config-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/kdcube_for_agents/named-services-mcp-README.md
---
# The Tool Catalog

The tool catalog is the rendered inventory of an agent's effective tool
roster — the part of the instruction that tells the model what it can call,
with each tool's full contract. It is a reusable concept: any harness that
binds an agent's tools renders a catalog from the same roster and the same
contracts (the ReAct decision agent renders it into its system instruction;
bridged agents on MCP or other bindings receive the same tool docs through
their binding). The companion concept:
[instruction blocks and their signals](../skills/instruction-blocks-and-signals-README.md)
own the instruction BODY; this page owns the catalog — and the way the
**roster conditions the instruction itself**: which capability blocks
render, which protocol channels are taught, which catalog sections exist.
One principle carries it all: **the effective roster is the single
authority for capability teaching** — what renders is decided by what is
wired.

## Where the tools come from

An agent's tool roster is declared in its app config
(`surfaces.as_consumer.agents.<agent>.tools` in the deployment descriptor),
one entry per connection:

```yaml
tools:
  - name: web
    alias: web_tools
    kind: python
    module: kdcube_ai_app.apps.chat.sdk.tools.web_tools   # SDK module…
  - name: user_memory
    alias: user_memory
    kind: python
    ref: tools/user_memory_tools.py                        # …or app-local file
  - name: exec
    alias: exec_tools
    kind: python
    module: kdcube_ai_app.apps.chat.sdk.tools.exec_tools
    allowed: [execute_code_python]                         # name allow-list
    tool_traits:
      execute_code_python:
        strategy: [exploration, exploitation]              # multi-action traits
  - kind: named_service
    alias: named_services
    namespaces:
      slack: { allowed: ["provider.about", "object.search", "object.action"] }
```

- `kind: python` connections load a module (dotted SDK path) or an app-local
  `ref`; each function the module exposes becomes a tool id
  `<alias>.<function>`.
- `allowed` narrows a connection to named tools; `runtime` maps per-tool
  execution placement; `tool_traits` attaches per-tool metadata — the
  `strategy` trait feeds the multi-action compatibility matrix.
- `kind: named_service` connections expose the `named_services.*` operation
  tools scoped to the listed namespaces; the catalog renders each tool's
  `namespaces applicable` from this scope.
- The **effective roster** at a turn = these admin defaults ⊕ the user's
  per-agent tool selection (the inventory user-pick). Runtime binding turns
  the roster into *adapters* — `{id, doc, call_template}` records — and the
  adapter ids are what every conditional decision below reads.
- `react.*` tools (read, pull, write, patch, rg, memsearch, hide, plan,
  checkout) belong to the react loop itself and are always present;
  `react.delegate` joins for a subagent parent, `react.contribute` for a
  subagent child.

## The catalog inside the instruction

In the ReAct decision instruction the order is protocol → instruction body
→ **tool catalog → skill catalog** → admin customization (assembly in
`decision_prompt.py`; rendering in `layout.py`). The catalog renders as up
to three sections:

| Section | Holds | Present |
| --- | --- | --- |
| `[AVAILABLE REACT-LOOP TOOLS]` | `react.*` plus web/exec-classed tools the loop calls directly | always |
| `[AVAILABLE COMMON TOOLS]` | the rest of the roster (rendering, named services, app tools) | always |
| `[TOOLS AVAILABLE ONLY IN CODE SNIPPET]` | tools callable only from generated code (`io_tools.tool_call`, `ctx_tools.fetch_ctx`) | only when `exec_tools.execute_code_python` is in the roster |

Each entry carries the tool's contract: purpose, parameters (name, type,
default, description), return description, constraints, `namespaces
applicable` and per-namespace traits for named-service tools, and the
`strategy` trait for the multi-action matrix.

### Full and compact forms

`tool_catalog_detail` selects the rendering form:

- **full** — banner layout, prose-wrapped entries, example sections
  rendered.
- **compact** — dense flat lines, example sections left out.

**The contract text is identical in both forms.** Purpose, every parameter
description, the return description, and every constraint render in full —
compactness comes from format, never from cutting (a truncated description
is a lost contract: the model acts on the missing half). Measured on a
7-tool SDK roster, compact is ~83% of full — the two forms converge for
tools whose documentation *is* their contract.

The `skills_form` facet is the companion knob for the skill catalog, and
`include_skill_gallery` switches the skill gallery on or off.

## The roster conditions the instruction

Beyond listing tools, the roster decides what the instruction *teaches*.
The design intent: **the instruction teaches a capability exactly when the
agent has it.** Teaching a capability whose tool is absent makes the model
believe it can use it (the concepts exist only through this text — the
model has no prior about them), and an instruction written for one fixed
roster goes stale the moment an administrator or user changes the tools.

The shared foundation: at turn start the runtime binds the effective
roster into adapters, and the set of **adapter ids** is the single input
every mechanism below reads. The id-to-consequence mapping is
`capability_instruction_exclusions()` in `decision_prompt.py`:

| Adapter ids | When absent, excluded teaching |
| --- | --- |
| `exec_tools.execute_code_python` | the exec blocks (`REACT_LITE_EXEC_TOOL`, `REACT_XLITE_EXEC`) |
| any `rendering_tools.*` | the rendering blocks (`REACT_LITE_RENDERING_TOOLS`, `REACT_XLITE_DOCUMENTS_RENDERING`) |
| any `web_tools.*` | the web blocks (`REACT_LITE_WEB_TOOLS`, `REACT_XLITE_WEB`) |

Mechanisms 1–4 key off this one function or the same id test — there is no
second switch to keep in sync.

### 1. Capability instruction blocks

The lite/extra-lite bodies are composed from named blocks, and the composer
receives the exclusion set. An excluded block is dropped **even when a
profile or an admin `blocks` list names it** — a profile is an admin
convenience, not authority to advertise a tool the user disabled. Effect:
a workspace agent on `xlite:workspace_exec` with exec switched off gets a
body with no `REACT_XLITE_EXEC` — no OUTPUT_DIR contract, no channel-code
binding rules, no contract-files teaching.

### 2. The protocol's code channel

The strict channel protocol (v3 single-action, v3 multi-action, v2 — in
each version's `agents/decision.py`) always teaches the base channels —
`thinking`, `action`, optional `summary` — plus one sentence: *a connected
tool may extend the protocol with a channel of its own; when such a tool
is available, this instruction carries that channel's rules.*
`<channel:code>` is that extension. With exec in the adapters the protocol
says "4 channel types, three required every round", shows round shapes
with `<channel:code>`, and carries the code-channel details ("code goes
only in `<channel:code>`, the tool has no `code` param"). Without it the
same sections say "3 channel types, two required", the worked examples use
`react.write`, and the code channel appears nowhere. The runtime parser
accepts both response shapes: it requires only that the first channel is
`thinking`, and reads the code channel with a default when absent.

### 3. The extended (full) body's exec sections

The legacy full/default body is assembled from named constants; three are
exec teaching: `CODEGEN_BEST_PRACTICES_V2`, `EXEC_SNIPPET_RULES`, and the
`#### External owner namespace browsing in exec` subsection of the paths
guide. `build_default_decision_instruction_body(include_exec=...)`
includes them only with exec in the roster — and `include_exec` is derived
from the same exclusion set, so the full body and the block tiers cannot
disagree about whether exec exists.

### 4. The code-snippet catalog section

`[TOOLS AVAILABLE ONLY IN CODE SNIPPET]` lists tools callable only from
generated code (`io_tools.tool_call`, `ctx_tools.fetch_ctx`). Those
helpers can be in the roster while exec is not — but without exec there is
no code to call them from, so the section renders only when
`execute_code_python` is in the catalog. Advertising them without exec
reads as "I can run code" (this exact misread is what surfaced the whole
mechanism family).

### 5. Generic surfaces speak class rules

The inverse mechanism: the always-present text is written to be
roster-proof instead of being gated. The `react.*` tool docs, the
protocol's visibility rule, the causality constants, ANNOUNCE, and the
shared paths/workspace blocks describe capabilities as CLASSES —
*file-processing tools* (operate on local physical files),
*computation tools* (create smaller derived artifacts), *the producing
tool's external visibility*, *every tool output is capped* — and each
opt-outable tool's own teaching travels with that tool (mechanism 1). A
concrete tool that is present matches its class rule through its own
catalog entry; a tool that is absent is simply never named. This is what
keeps the instruction truthful under any roster, and a regression test
(`v3/test_exec_conditional_instruction.py`) fails on any exec signal
appearing in an exec-less composition.

### 6. The named-services block

`[NAMED SERVICES — NAMESPACE OBJECT OPERATIONS]` (with its CONTRACT FIRST
schema rule) is composed by `compose_named_service_agent_instructions`
only when the agent's tools config contains `kind: named_service`
connections, and it closes with the roster of exactly the connected
namespaces (e.g. ``- `slack` — Team messaging``). No named-service
connections: no block, no namespace vocabulary in the instruction at all.

### The taxonomy, in one line each

- Mechanisms 1–4: **conditional rendering** — one authority (adapter ids)
  decides what appears.
- Mechanism 5: **roster-proof text** — unconditional surfaces phrased in
  classes so they hold under any roster.
- Mechanism 6: **composition from the roster** — the block is built out of
  the configuration itself.

Because the forge's token projection composes through these same
functions, toggling one tool in a draft moves all of them at once —
removing exec shrinks the instruction body AND the catalog, exactly as the
live agent would see it.

## The configuration that drives it

| Knob | Where | Effect |
| --- | --- | --- |
| `tools` (per agent) | `surfaces.as_consumer.agents.<agent>.tools` | the roster: catalog entries, capability blocks, protocol extensions |
| `allowed` (per connection) | tools entry | narrows which of the connection's tools exist for this agent |
| `tool_traits` (per tool) | tools entry | `strategy` feeds the multi-action matrix; traits render with the entry |
| user tool selection | per-user agent selection | ⊕ with admin defaults = the effective roster |
| `tool_catalog_detail` | `react…instructions` facet; profile option facets; user picker | full or compact catalog form (user pick > profile default > agent config > full) |
| `skills_form` | same facet family | full or compact skill bodies |
| `include_skill_gallery` | `react…instructions` | skill gallery on/off |
| `multi_action_mode` | `react.<agent>` (quoted `"off"`/`"on"`) | which protocol variant wraps the catalog's strategy traits |
| `subagents` | `react.<agent>` | adds `react.delegate` to the catalog and its teaching |
| `instruction_profiles` / `instructions.blocks` | `react.<agent>` | the body the catalog accompanies — see the system-instruction doc |

The Agent Forge (agentic-config) edits every knob in this table from one
interface and projects the token cost of the result — the projection
composes this same catalog through the same functions, so its numbers are
the real ones.
