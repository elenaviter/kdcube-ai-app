---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/react-object-materialization-README.md
title: "Namespace Services: ReAct Object Materialization"
summary: "ReAct adapter diagram over the shared runtime harness for pulling, reading, owner-projecting, and rendering named-service objects."
status: current
tags: ["sdk", "namespace-services", "react", "pull", "read", "block-production", "events"]
updated_at: 2026-07-27
keywords:
  [
    "react.pull",
    "react.read",
    "named service object get",
    "block.produce",
    "block.render",
    "owner projection",
    "source_namespace",
  ]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/ecosystem-component/components-ecosystem-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/react-object-policy-bridge-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/providers-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/clients-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/integration-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/events/event-subsystem-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/events/namespaces-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/agents/react/artifact-discovery-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/events/artifact-resolution-and-materialization-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/workspace/references-and-paths-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/timeline/provider-projection-README.md
---
# Namespace Services: ReAct Object Materialization

This page is the ReAct-adapter diagram for named-service objects. Generic ref
resolution, byte identity, and workspace placement are owned by the shared
agentic [runtime harness](../../runtime/harness/README.md), with the canonical
materialization contract in
[Artifact Resolution And Materialization](../../runtime/harness/events/artifact-resolution-and-materialization-README.md).
This page covers what ReAct adds on top:
`react.pull -> react.read -> Timeline.render`.

Current implementation state:

- `react.pull` calls the provider through `object.get(response_mode=stream)`.
- `react.read` calls owner block production through `block.produce` when that
  policy is declared. Pull-only providers remain valid: their exact
  materialized bytes are read through the generic artifact path.
- `Timeline.render()` renders stored timeline blocks, applies local ReAct
  projection policies, then runs the optional named-service `block.render`
  projection for providers whose objects are present in the visible timeline.
- `block.render` is called concurrently per relevant provider event source.
  The provider receives a bounded timeline snapshot and returns patches. The
  central merge accepts patches only for blocks owned by that provider
  namespace.
- `block.render` may also support direct-client responses such as
  `{format, markdown, blocks}`. The automatic `Timeline.render()` path consumes
  timeline-mode patches or indexed blocks.
- A provider response with `named_service_operation_not_supported` is cached as
  an unsupported render hook for the process lifetime, so later renders skip
  that provider until the runtime restarts.

## Runtime Boundary Diagram

```text
1. Object ref is visible
   executor: model, chat context, canvas card, search-result widget, or timeline
   runtime: consumer/client runtime
   data:
     object_ref = mem:record:mem_123

        |
        v

2. Agent requests materialization
   executor: ReAct tool runner
   runtime: consumer ReAct runtime
   call:
     react.pull(paths=["mem:record:mem_123"])
   boundary:
     local model/tool runtime -> namespace rehoster
   latency:
     local dispatch plus one provider lookup/call when the ref is not already
     a local conv:fi: artifact

        |
        v

3. Harness resolution calls the owner provider
   executor: shared harness resolver + named-service artifact rehoster
   runtime: consumer backend, using Named Service Discovery
   provider call:
     object.get(
       namespace="mem",
       object_ref="mem:record:mem_123",
       response_mode="stream"
     )
   boundary:
     consumer backend -> provider bundle/backend
     transport is bundle_registry, bundle_operation, API, or another configured
     named-service transport
   provider responsibility:
     auth check, object lookup, byte stream production, compact sidecar
     response
   latency:
     provider discovery + provider object lookup + streamed byte transfer

        |
        v

4. Harness workspace receives bytes
   executor: named-service artifact rehoster / shared harness workspace
   runtime: agentic harness, with ReAct as this call's adapter
   work:
     write streamed bytes under OUTPUT_DIR
     mint a local conv:fi: artifact path for the current turn
     preserve the provider-returned canonical object_ref in pull state and
     artifact metadata
   result:
     logical_path  = conv:fi:turn_1.files/mem_123.json
     physical_path = turn_1/files/mem_123.json
     object_ref    = mem:record:mem_123
     source_namespace = mem
   latency:
     local filesystem write

        |
        v

5. Agent reads the materialized artifact
   executor: ReAct tool runner
   runtime: consumer ReAct runtime
   call:
     react.read(items=[{"path": "conv:fi:turn_1.files/mem_123.json"}])
   work:
     read local bytes
     construct a read target whose metadata includes:
       meta.object_ref       = mem:record:mem_123
       meta.source_namespace = mem
       meta.materialized_path = conv:fi:turn_1.files/mem_123.json
   latency:
     local filesystem read before owner projection

        |
        v

6. ReAct resolves the owner event source
   executor: EventSourceSubsystem
   runtime: consumer ReAct runtime
   work:
     resolve object_ref to an event source id
     ordinary path uses registered named_services.<source_namespace>
     richer path may call provider event.resolve for URI-specific routing
   traces:
     react.read.owner_projection status=...
   boundary:
     local registry lookup, plus optional consumer -> provider event.resolve
     call
   latency:
     local lookup on the ordinary path
     one lightweight provider call when event.resolve is used

        |
        v

7. ReAct asks the owner to produce model-visible blocks
   executor: named-service block-production adapter
   runtime: consumer ReAct runtime
   provider call:
     block.produce(
       namespace="mem",
       object_ref="mem:record:mem_123",
       target=<read target with materialized fi path and owner metadata>
     )
   boundary:
     consumer ReAct runtime -> provider bundle/backend
   provider responsibility:
     convert the owner object into bounded ReAct blocks for model context
     according to the namespace's object semantics and read/render policy
   latency:
     one provider call, plus provider-side object read if the provider needs it

        |
        v

8. ReAct stores the read result blocks
   executor: react.read
   runtime: consumer ReAct runtime
   result:
     if provider returned blocks:
       timeline block path may remain owner-oriented or include owner metadata
       block.meta.owner_projected = true
       block.meta.object_ref = mem:record:mem_123
       block.meta.materialized_path = conv:fi:turn_1.files/mem_123.json
     fallback:
       generic text block for the local conv:fi: artifact
   latency:
     local timeline append

   owner policy may also:
     emit a compact read fact instead of a large object body
     attach original_object_stats for stats_only reads
     attach owner metadata that a later ANNOUNCE policy can render for a
     bounded number of prompt renders

        |
        v

9. ReAct renders prompt context
   executor: Timeline.render()
   runtime: consumer ReAct runtime
   work:
     render stored blocks
     apply local ReAct timeline, announce, and compaction projection policies
     scan visible blocks for provider-owned object_ref values
     call relevant provider block.render hooks concurrently
     apply valid provider patches to provider-owned blocks
     expose the resulting message stream to the model
   boundary:
     local timeline projection plus optional consumer -> provider block.render
     calls for provider namespaces present in the visible blocks
   latency:
     local timeline rendering plus the slowest provider render call when
     at least one provider implements block.render
```

## Operation Map

| Stage | Operation | Caller | Callee | Automatic in ReAct path |
| --- | --- | --- | --- | --- |
| Pull bytes | `object.get(response_mode=stream)` | harness namespace resolver used by `react.pull` | named-service provider | Yes |
| Resolve owner source | `event.resolve` | EventSourceSubsystem / resolver bridge | named-service provider | Optional |
| Produce read blocks | `block.produce` | ReAct owner-projection adapter | named-service provider | Yes, when owner projection is registered and `object_ref` is present |
| Render provider representation | `block.render` | ReAct timeline projection or explicit custom client | named-service provider | Optional, only when provider-owned blocks are present and the provider implements it |
| Render model prompt | `Timeline.render()` | ReAct runtime | local timeline and local policies | Yes |

## Materialization Scope

`react.pull` returns local artifact rows. A row may include a `scope` such as
`snapshots`:

```json
{
  "object_ref": "cnv:main",
  "logical_path": "conv:fi:turn_1.git/snapshots/cnv/main.json",
  "physical_path": "turn_1/git/snapshots/cnv/main.json",
  "scope": "snapshots",
  "mime": "application/json"
}
```

`scope` describes where the bytes were materialized in the ReAct workspace. It
is not the owner namespace and must not be used for named-service routing.
Routing and owner policy selection use the preserved `object_ref`.
The path grammar itself is canonical in
[Harness References And Workspace Paths](../../runtime/harness/workspace/references-and-paths-README.md).

## Owner Identity Field

`object_ref` is the canonical owner identity field throughout this path. It is
the provider-returned canonical ref when the provider normalizes an alias such
as `mem:<id>` to `mem:record:<id>`. A materializer may retain the requested ref
as diagnostic metadata when it differs, but read/projection/render policy
selection uses `object_ref`.
`react.read` is called with the materialized `conv:fi:` path. The carried
`object_ref` is the owner identity used by owner event-source routing and
`block.produce`.

## Projection And Rendering Ownership

Provider storage and projection code must return canonical structured state,
not model-facing prose. Storage may return fields such as `object_ref`,
`revision`, `bounds`, `legend`, `cards_count`, `namespace`, and
`object_kind`; it must not hide ReAct instructions, edit protocols, or prompt
framing inside those storage objects.

The generic `react.pull` / `react.read` tools must remain provider-agnostic.
If a provider needs stats, owner revision, refresh guidance, or rendering
metadata, its block-production policy should place that metadata on the block
it produces, using stable fields such as `original_object_stats`. The generic
tool may read documented metadata fields, but it must not branch on concrete
namespaces such as `cnv`, `mem`, or `task`. See
[Object Refs, Presentation, And Actions](object-ref-presentation-and-actions-README.md).

Model-facing text is owned by one of these layers:

- `block.produce`, when a concrete object is read through `react.read`;
- local timeline/announce/compaction projection policies, when runtime-local
  volatile state such as a canvas board needs bounded prompt visibility;
- provider `block.render`, when provider-owned visible blocks need final
  prompt-time patches.

Instruction strings used by those policies belong in the namespace or solution
instruction module, not in storage. For example, canvas storage returns the
board projection facts, while the canvas announce policy decides whether the
board is visible and renders `[CANVAS BOARD]` text using canvas instructions.
This keeps ReAct generic: `react.read` does not branch on `cnv`, `mem`, `task`,
or any future namespace, and storage remains reusable by UI, API, event, and
model clients.

## Provider Contract

For an object namespace that supports ReAct materialization:

- `object.get(response_mode=stream)` provides the exact bytes that become the
  local `conv:fi:` artifact.
- The stream response sidecar carries compact identity and diagnostics. The
  large object body is streamed as bytes, not embedded in the JSON response.
- `event.resolve` provides lightweight URI-to-event-source routing when the
  namespace needs more than the default `named_services.<namespace>` event
  source.
- `block.produce` optionally replaces the generic artifact read with
  owner-shaped model-visible blocks.
- `block.render` optionally patches provider-owned blocks during model prompt
  rendering. It receives a bounded block snapshot and a render context with
  `phase`, `audience`, event-source id, trigger refs, and limits.

## Owner-Projected Snapshot Pattern

Google Sheets is a concrete snapshot provider. Discovery and ontologic tools
return canonical spreadsheet or tab refs, for example:

```text
sheets:google:<account_id>:spreadsheet:<spreadsheet_id>
sheets:google:<account_id>:spreadsheet:<spreadsheet_id>:tab:<sheet_id>
```

When the consumer declares a provider-backed pull policy for `sheets`, this
sequence is available without any Sheets branch in ReAct:

```text
react.pull(paths=[<sheets ref>])
  -> sheets object.get(response_mode=stream)
  -> kdcube.sheets.snapshot.v1 JSON artifact
  -> conv:fi:.../named_services/sheets/.../*.sheets.json

react.read(paths=[<returned conv:fi ref>])
  -> sheets block.produce
  -> compact [SHEETS SNAPSHOT] inventory without cell values

react.rg(root=<returned conv:fi ref>, pattern=...)
  -> read_items for matching local JSON lines

react.read(items=[<returned read_item>])
  -> exact JSON range, with ordinary read caps
```

The snapshot contains the canonical owner ref, spreadsheet/tab metadata, and
used values for selected grid tabs. KDCube does not cap read ranges or returned
cells. A provider-side failure remains explicit; the consumer can request
smaller A1 ranges when Google requires the work to be split.

Sheets declares `event.resolve` and `block.produce` so a whole-file read stays
small and useful: the model sees workbook identity, tabs, dimensions,
materialized range/count summaries, completeness, and the logical/physical
snapshot paths. The cell matrices remain in `values.ranges[].values` for code.
An explicit ranged `react.read` is different: the provider returns no owner
block, so the generic reader exposes that exact line/symbol chunk. This keeps
`react.rg -> read_items -> react.read` available for textual exploration.

Sheets does not need `block.render` or ANNOUNCE behavior for this snapshot.

## Volatile Object Pattern

Some objects are live collaborative state rather than static files. Canvas is
the reference pattern:

```text
react.pull(paths=["cnv:main"])
  -> writes conv:fi:turn_1.git/snapshots/cnv/main.json

react.read(paths=["conv:fi:turn_1.git/snapshots/cnv/main.json"])
  -> canvas block-production emits a compact [CANVAS TOOL RESULT] fact
  -> canvas announce policy renders [CANVAS BOARD] for N render rounds
```

The compact timeline fact is durable. The ANNOUNCE board map is not. The fact
should tell the model how to refresh:

```text
announce_effect: board projection refreshed in ANNOUNCE for 3 render rounds
refresh_rule: use react.pull(paths=['cnv:main']) and react.read on the returned conv:fi: path if you need an updated or prolonged board view
```

This text is produced by the canvas owner policy, not by generic `react.read`.
Other providers can implement the same pattern for volatile objects by keeping
their own refresh instructions in their block-production and ANNOUNCE policies.

## Provider Render Merge Contract

The central ReAct render adapter discovers provider renderers from the timeline
it is about to expose to the model.

```text
Timeline.render visible blocks
  |
  | scan block.meta.object_ref / block.object_ref
  v
group by owning event source, for example named_services.mem
  |
  | asyncio.gather(...)
  v
provider block.render calls run concurrently against the same input snapshot
  |
  | validate returned patches
  v
merge accepted patches into the visible block list
```

The provider request payload has this shape:

```json
{
  "blocks": [
    {
      "index": 42,
      "type": "react.tool.result",
      "text": "...",
      "meta": {
        "object_ref": "mem:record:mem_123"
      }
    }
  ],
  "render_context": {
    "phase": "timeline_projection",
    "audience": "model",
    "event_source_id": "named_services.mem",
    "trigger_object_refs": ["mem:record:mem_123"],
    "limits": {
      "max_blocks": 64,
      "neighbor_radius": 4
    }
  }
}
```

`blocks` is a bounded snapshot. It includes provider-owned blocks and nearby
context blocks. Nearby blocks are available as context for rendering decisions.
The merge contract is provider-owned-block mutation: returned patches are
accepted for indexes whose block belongs to the provider namespace/event source.

Supported patch forms:

```json
{ "op": "replace_block", "index": 42, "block": { "type": "react.tool.result", "text": "..." } }
{ "op": "patch_block", "index": 42, "fields": { "text": "...", "meta": { "render_policy": "..." } } }
{ "op": "append_block_after", "index": 42, "block": { "type": "text", "text": "..." } }
```

Provider calls are independent. Every provider sees the same input snapshot.
Merge order does not create data dependencies between providers because a
provider patch can change only blocks owned by that provider namespace.

Direct-client `block.render` calls can return a rendered representation such as
`{format, markdown, blocks}`. The prompt-time render adapter uses the
timeline-mode response: `patches[]`, or `blocks[]` entries that carry
`index`/`target_index`/`block_index`.

## Consumer Runtime Contract

For a consumer ReAct runtime:

- Pull state preserves the owner URI:
  `logical_path -> object_ref/source_namespace`.
- Read targets carry that owner metadata into block production.
- Owner projection uses owner metadata, not the local `conv:fi:` prefix.
- Read traces are emitted under `react.read.owner_projection`.
- The prompt renderer consumes stored blocks and may call provider
  `block.render` hooks for visible provider-owned blocks.
- Provider render traces are emitted under `named_services.block_render`.

For the field-level owner policy contract, including top-level
`original_object_stats` for `react.read(stats_only=true)`, see
[ReAct Object Policy Bridge](react-object-policy-bridge-README.md).
