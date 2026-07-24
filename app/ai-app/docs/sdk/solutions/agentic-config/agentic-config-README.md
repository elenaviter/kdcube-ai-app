---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/agentic-config/agentic-config-README.md
title: "Agentic Config: Stored Instruction Sets, The Constructor, And The Agent Forge"
summary: "Agent instructions and configuration as managed artifacts: versioned stored sets wired by instr:custom:<id>:<version>, a governed instr namespace, block signals and token weights, the constructor widget (block library, segmented composed rendering, immutable versions, Assign), and the agent forge — the full per-agent configuration edited as a staged draft with a live token projection of the real system instruction, applied in one merge."
status: active
tags: ["sdk", "solutions", "agentic-config", "instructions", "agents", "admin", "widget", "named-services", "forge", "tokens"]
updated_at: 2026-07-25
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/react/system-instruction-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/kdcube_for_agents/named-services-mcp-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/configuring-agent-service-access/configuring-agent-service-access-README.md
---
# Agentic Config: Stored Instruction Sets, The Constructor, And The Agent Forge

Agents are configured, extended, reconfigured, and tuned — and their SYSTEM
INSTRUCTION moves as fast as their tools and skills. `agentic_config`
(`kdcube_ai_app/apps/chat/sdk/solutions/agentic_config`) makes instruction
sets **managed artifacts**: authored from blocks, stored in immutable
versions with provenance, previewed exactly as an agent receives them, and
wired to any agent by ref.

The composition vocabulary itself (what a `blocks` list may contain, the
predefined profiles, capability-conditional exclusions) is owned by
[the ReAct system-instruction doc](../../agents/react/system-instruction-README.md)
— this page owns the STORE, the namespace, and the authoring surface.

## Stored instruction sets

A stored set is an ordered item list in the composition vocabulary, saved
under a slug id:

- **Ref = wiring ref.** `instr:custom:<id>[:<version>]` is both the object
  ref in the `instr` namespace and the token a descriptor wires. Version
  omitted = latest active.
- **Versions are immutable.** An edit inserts the next version; a ref pinned
  to a version always resolves to the same content. Retiring flips status —
  a PINNED ref keeps resolving even when retired (running agents never
  break); only the unpinned "latest" read filters to active.
- **Provenance is first-class.** Every version records who created it; every
  retire records who and when.
- **Description and tags** make units distinguishable and findable: listing
  supports a `q` substring (id/name/description) and tag containment.
- Storage: the `agentic_instructions` table in the project schema
  (tenant/project-scoped; no cross-project sharing), `items` as JSONB.

At runtime, custom refs expand **asynchronously from the store before
composition** — recursively (stored sets may reference stored sets),
cycle-safe, and fail-open: a ref that cannot resolve is dropped with a
warning, never leaked into a prompt as literal text.

## The block library

The constructor composes from three kinds of units:

1. **Predefined sets** — `instr:profile:full | lite | extra-lite`.
2. **Built-in blocks** — the moderate (`REACT_LITE_*`) and extra-lite
   (`REACT_XLITE_*`) registry blocks. `builtin_block_catalog()` serves each
   with its MEANING: curated **signals** (the behaviors the block protects
   or teaches, from `block_signals.py`) and semantic **tags** reflecting
   them — tier and profile memberships are separate fields, never tags —
   plus the full block text and its **token weight** (cl100k, counted
   server-side). The signal table in the system-instruction doc remains the
   authoritative long-form purpose map.
3. **Stored units** — including "blocks" you author yourself: a stored unit
   whose items are one literal text IS a custom block, composable into other
   sets by ref. Authors fill signals and keyword tags at save time, so
   custom units are distinguishable the same way built-ins are.

## The `instr` namespace and its governance

`kdcube-services@1-0` registers the `AgenticInstructionsNamedService`
provider: `provider.about`, `object.list` (with `q`/`tags` filters),
`object.get` (one version + history), `object.upsert` (next version),
`object.delete` (retire). Reads are open to the surface's callers; **writes
are admin-gated in the provider** — a widget is never the only gate.

Delegated governance mirrors that: the `instr:read` grant is delegable to
signed-in roles, `instr:write` to super-admin only, with per-operation
grants on the named-services door. **The namespace is deliberately absent
from every agent's `as_consumer` roster** — it serves administrators,
widgets, and governed external clients; an agent sees it only if a roster
explicitly names it.

## The operations facade

The same provider answers the widget through
`kdcube-services@1-0`'s `agentic_instructions` operation
(`body.data.action`):

| action | payload | returns |
| --- | --- | --- |
| `list` | `{include_retired?, q?, tags?}` | latest version per id |
| `blocks` | — | the built-in block catalog (signals, tags, profiles, text, token weight) |
| `get` | `{ref}` | one version + version history |
| `save` | `{instruction_id, name, description?, tags?, signals?, items}` | the next immutable version (admin) |
| `retire` | `{ref}` | retire pinned version / whole id (admin) |
| `preview` | `{items, workspace_implementation?}` | the composed body exactly as the runtime builds it, per-item `segments`, token counts (total + per segment) |
| `project` | `{draft: {react?, consumer?}, include_text?}` | the token breakdown of the REAL system instruction a DRAFT agent config would produce (see the forge below) |

The operations route wraps results under the op alias
(`{status: "ok", …, agentic_instructions: {ok, …}}`).

## The admin surface: one package, one place

The views live in the shared package
`@kdcube/components-react/agentic-config` (transport-injected, so any admin
host mounts them): `AgenticConfigProvider` + `AgenticConfigTabs` with three
views — **Instruction sets**, **Agents** (the forge), **App settings**. The
app-config widget hosts them as its "Agents & Instructions" tab beside the
structured app-config panel — one surface for the whole app configuration;
the dedicated widget served by `kdcube-services@1-0`
(`sdk://solutions/agentic_config/ui/widget`, admin surface) is a thin shell
over the same package.

### Instruction sets (the constructor)

- **Block library** — searchable by name, signal, or tag across built-in
  blocks and stored sets. Clicking a card SHOWS the block (signals, tags,
  profile memberships, full text, token weight); the `+` adds it to the
  item list. Built-in sets sit in the sidebar, selectable and assignable
  exactly like stored ones; every sidebar set shows the token weight of its
  EXPANDED composition.
- **Composed instruction** — rendered CONTINUOUSLY beside the editor
  (debounced server-side compose), SEGMENTED per source item: each section
  carries a clickable source label that jumps back to its block, its token
  weight, and the view shows the total.
- **Save as v(n+1)** — immutable versions, provenance echoed back;
  **Retire** per version.
- **Assign** — wires a set (built-in or stored) to an application agent:
  pick the app and the agent, and the widget adds/updates an
  instruction-profile OPTION (id = the slug, blocks = the pinned ref) via
  the platform admin props write — user-pickable immediately, optionally as
  the profile default. The write lands live; the descriptor file remains
  the restart-time source of truth.

### Agents (the forge)

The full per-agent configuration from one interface, edited as a **staged
draft** — nothing reaches the application until one explicit **Apply**:

- An agent is the union of its two config roots — the `react` runtime block
  (`react.agents.<key>` / direct `react.<key>`, `default_agent` fallback)
  and the `surfaces.as_consumer.agents.<key>` inventory — and the list
  badges which roots each agent occupies.
- **Quick controls** stage the scalars: tool catalog full/compact, skills
  form, skill gallery, multi-action mode, event pipeline, story snapshots,
  serving `max_tokens`.
- **Section editors** (summary line + YAML) stage each area wholesale:
  instruction profiles, instructions, supported models, subagents (react
  root); tools & traits, skills, event sources, model, capabilities
  (inventory root). Staged sections carry chips and per-section Unstage.
- **Apply** merge-writes BOTH roots in a single admin props call to the
  agent's real containers; **Discard** drops the draft; the draft bar shows
  the exact YAML Apply will write. Adding an agent creates a draft-only
  profile that materializes on apply.
- **The projection meter** shows what the TOTAL system instruction would
  weigh under the draft — computed server-side (`project`) by composing the
  REAL decision system text with the draft's levers: instruction sets
  (stored refs expanded), catalog detail full/compact, the CONNECTED tool
  roster (tool modules genuinely loaded, `allowed` narrowing honored — the
  catalog inside the instruction grows with the roster, and
  capability-conditional blocks appear exactly when their tools are
  present), skill gallery, subagents (adds the delegation tool), and the
  action protocol. The breakdown (base+instructions / tool catalog / skill
  gallery) comes from diffs of real compositions; tool kinds the projection
  cannot load in-process (MCP, named-service connections) are listed as
  not projected — never silently counted as zero.

### App settings

The effective stored config as YAML, a YAML/JSON merge-patch editor over
the platform admin props route (roles, visibility, anything), and secrets
set/clear through the admin secrets route — keys listed redacted, values
write-only.

Presentation facets (`tool_catalog`, `skills_form`) are the companion picker
surface — profile defaults the user overrides — documented with the
composition vocabulary in the system-instruction doc.
