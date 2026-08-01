---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/object-ref-presentation-and-actions-README.md
title: "Object Refs, Presentation, And Actions"
summary: "Canonical contract for ecosystem objects: object_ref is the universal handle, namespace presentation config owns visual identity, and provider resolvers own actions and subtype semantics."
status: active
tags: ["sdk", "namespace-services", "object-ref", "presentation", "resolvers", "scene", "canvas", "chat"]
updated_at: 2026-08-02
keywords:
  [
    "object_ref",
    "namespace presentation",
    "object_kind",
    "object action",
    "resolver",
    "canvas pins",
    "scene overlay",
    "chat context"
  ]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/providers-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/clients-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/react-object-materialization-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/scene/generic-scene-contract-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/canvas/canvas-sdk-solution-README.md
---
# Object Refs, Presentation, And Actions

This is the ecosystem object contract used by scene hosts, canvas pins, chat
context chips, ReAct materialization, and named-service providers.

The rule is:

```text
object_ref is the universal handle.
Presentation comes from namespace presentation config.
Actions come from the object resolver owned by the provider.
```

No scene, widget, canvas board, chat chip, or generic component should parse an
object URI to decide behavior. They pass the full `object_ref` through.

There is one narrow exception: a scene or surface registry may use declarative
compatibility selectors supplied by scene config, a component claim, or provider
metadata to decide whether a surface is a candidate drop target. That selector
layer is not behavior resolution. It is equivalent to a route hint. The
provider resolver still owns open/download/preview semantics.

## Boundary

Scope note: this doc is about presenting one OBJECT (a search hit, a pinned
card, a viewer row). The realm-level presentation a service card renders
(purpose, works-with, operation/action labels) is a different declaration —
see [Providers — The Presentation Layer](providers-README.md).

```text
component / scene / canvas / chat
  owns: mount, display, drag/drop, selected action UI
  passes: object_ref exactly as received
       |
       v
object resolver registry
  owns: route request to the owner resolver
  input: { object_ref, action, payload }
       |
       v
provider resolver
  owns: URI parsing, object kind, auth, capabilities, open/download/preview
       |
       v
result
  capabilities/actions -> UI buttons and behavior
  ui_event             -> scene surface command / open request
  object bytes/data    -> preview, download, ReAct materialization
```

The only code allowed to inspect URI structure is:

- the centralized resolver/router, only to select a registered owner resolver;
- the provider resolver, because the provider owns that URI grammar.

Everything else treats the URI as opaque.

## Identity Fields

| Field | Owner | Meaning |
| --- | --- | --- |
| `object_ref` | provider/object owner | Canonical URI for the object. This is the handle for resolve, action, pull, read, and render. |
| `namespace` | source component or provider | Owner or scoped presentation namespace such as `mem`, `task`, `cnv`, `conv`, or `conv:fi`. This is a presentation/lookup key, not behavior. |
| `object_kind` | provider/source component | Optional presentation subtype such as `task:issue`, `task:attachment`, or `cnv:user:attachment`. This is a presentation/lookup key, not behavior. |
| `kind` | source component | Legacy/display label on canvas cards. It must not drive behavior. |

`namespace` and `object_kind` exist so generic UIs can choose a configured
visual identity without learning each provider's object grammar. They are not
mandatory for object identity: `object_ref` is still the handle. When these
keys are missing, generic UIs should use the neutral unknown presentation until
the provider resolver returns metadata.

For future provider expansion, the root namespace remains stable:

```text
task:issue:ticket_123
task:snapshot:daily/2026-06-22
task:attachment:ticket_123/attachments/a1/v000001/report.md
```

All of these route to the `task` provider. The provider may report
`object_kind` values such as `task:issue`, `task:snapshot`, and
`task:attachment`. The generic UI still passes the full `object_ref`.

## Presentation

Visual identity is not resolver behavior. It comes from runtime/app
configuration exposed by the public endpoint:

```text
POST /api/integrations/bundles/<tenant>/<project>/<app>/public/namespace_presentation_config
```

The response is a map keyed by presentation keys. All components should use
the shared resolver exported by `@kdcube/components-core`
(`namespacePresentationCandidates` + `namespaceStyleForContext`) so chat,
Pinboard, scene drag/drop, and future components resolve identically.

The lookup order is:

1. exact `object_kind`, when present;
2. exact explicitly declared `namespace`, when present;
3. normalized scoped style key (`conv:<first-subnamespace>` for conversation
   refs, such as `conv:fi`; otherwise the root namespace);
4. root namespace;
5. neutral unknown fallback.

Object-ref parsing is used only to derive a presentation candidate when
explicit namespace metadata is absent. It does not choose behavior.

Example:

```yaml
namespace_styles:
  task:
    label: task
    color: "#2563eb"
    border: "#2563eb"
    focus: "#60a5fa"
    background: "#eff6ff"
    icon_svg: '<svg viewBox="0 0 24 24">...</svg>'
  task:attachment:
    label: task file
    color: "#0f766e"
    icon_svg: '<svg viewBox="0 0 24 24">...</svg>'
  mem:
    label: memory
    color: "#16a34a"
    icon_svg: '<svg viewBox="0 0 24 24">...</svg>'
  "conv:fi":
    label: file
    color: "#ca8a04"
    icon_svg: '<svg viewBox="0 0 24 24">...</svg>'
```

The same map is shared by:

- chat context chips;
- canvas cards and pin flyouts;
- scene drag/drop overlay highlights;
- any standalone component that renders namespace refs.

Components may ship a neutral fallback for unknown refs. They must not ship a
hardcoded table like "memory = green search icon" or "task = blue check icon".

## Actions

Actions are provider-owned.

```json
{
  "object_ref": "task:issue:ticket_123",
  "action": "capabilities"
}
```

Expected resolver shape:

```json
{
  "ok": true,
  "object_ref": "task:issue:ticket_123",
  "namespace": "task",
  "object_kind": "task:issue",
  "capabilities": {
    "preview": true,
    "open": true,
    "download": false,
    "rehost": false
  },
  "actions": ["preview", "open"]
}
```

The generic UI decides which buttons to show from `capabilities`/`actions`.
It does not infer that `task:*` can open, that `conv:fi:*` can download, or that
`mem:*` should use a memory viewer. The provider resolver returns those facts.

`open` can return a scene-routable UI event:

```json
{
  "ok": true,
  "object_ref": "task:issue:ticket_123",
  "ui_event": {
    "type": "kdcube.ui.object.open.requested",
    "target_surface": "task_tracker.issue_editor",
    "action": "open",
    "object_ref": "task:issue:ticket_123"
  }
}
```

The scene host routes by `target_surface`; it does not parse the object ref to
choose a widget.

When the provider owns the browser experience, `open` can instead return an
explicit external URL:

```json
{
  "ok": true,
  "object_ref": "docs:google:account_1:document:doc_123",
  "ui_event": {
    "type": "kdcube.ui.object.open.requested",
    "action": "open",
    "object_ref": "docs:google:account_1:document:doc_123",
    "external_url": "https://docs.google.com/document/d/doc_123/edit"
  }
}
```

The provider performs the current identity, account, claim, and consent checks
for `object.action(open)` before returning this URL. A generic client opens only
an explicit HTTP(S) `external_url` from that action result. Display metadata
such as `object.web_url` from an earlier list or search result is not an
authorization result and does not drive navigation.

## Surface Compatibility

Surface compatibility is declarative and separate from object behavior. It may
be authored in host config for a concrete scene, emitted by a component as part
of its surface claim, or returned by provider/component metadata. It must not be
implemented as scene-core code branches.

Example:

```json
{
  "surface_ref": "website.memory",
  "target_surfaces": ["sdk.memory.viewer"],
  "dropTargets": [
    {
      "effect": "open",
      "accepts": { "open": ["mem:*"] },
      "requestedTargetSurface": "sdk.memory.viewer"
    }
  ]
}
```

This says:

```text
Refs that match this declared route hint are candidates for this surface.
```

It does not say:

```text
All mem:* refs can be opened without checking the provider.
```

The open flow remains:

```text
drag/drop candidate
  -> scene sees declared compatibility match
  -> scene calls object.action(open, object_ref, requestedTargetSurface)
  -> provider validates object_ref and returns ui_event.target_surface
  -> scene dispatches kdcube.surface.command to the registered surface
```

Selectors may be declared by scene host config, component claims, or provider
metadata. They are matched by a generic string matcher. The matcher understands
only exact values and prefix wildcards; it does not know task, memory, file,
conversation, or canvas semantics. Prefer provider/component-declared
compatibility for reusable components; use host config only to compose a
specific scene profile.

Good selectors:

```text
*
mem:*
task:issue:*
task:attachment:*
cnv:*
```

Bad selectors:

```text
all Judo cancellation tasks
latest memory for this user
cards on the main board
```

Those are provider/domain queries, not scene compatibility selectors.

## Canvas-Owned Objects

Canvas can host its own objects under `cnv:`.

Examples:

```text
cnv:main
cnv:canvas/users/<user>/attachments/<id>/v000001/report.md
```

Canvas-owned cards may use `object_kind` values such as:

```text
cnv:user:text
cnv:user:attachment
cnv:agent:text
```

Those values are presentation keys and schema hints for the canvas provider.
They are not a reason for other components to inspect URI tails.

## ReAct

ReAct uses the same identity through the shared harness:

- `react.pull(paths=["cnv:main"])` materializes the current object into the
  turn workspace through harness resolution while preserving `object_ref`;
- `react.read(conv:fi:...)` uses preserved `meta.object_ref` to ask the owner
  block-production policy for model-facing blocks;
- block-production policies can include `original_object_stats` or other
  provider metadata directly on blocks so generic ReAct code stays provider
  agnostic.

Neither the shared harness nor the generic ReAct read tool may contain
namespace-specific branches for canvas, task, memory, or any other provider.

## Anti-Patterns

Do not:

- parse `object_ref` in scene/widget/canvas UI code to decide behavior;
- treat selector matches as authorization or capability checks;
- keep local icon/color tables for provider namespaces;
- make `kind` decide open/download/preview behavior;
- add fake handoff resolvers for namespaces owned elsewhere;
- let canvas know provider object types such as task issue vs task attachment;
- put provider-specific block rendering logic into generic ReAct tools.

Do:

- pass the full `object_ref`;
- resolve actions through the registered provider resolver;
- load presentation from `namespace_presentation_config`;
- let providers own `object_kind`, schema, capabilities, actions, and bytes.
