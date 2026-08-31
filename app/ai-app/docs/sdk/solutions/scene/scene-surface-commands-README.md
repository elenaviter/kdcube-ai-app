---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/scene/scene-surface-commands-README.md
title: "Scene Surface Commands"
summary: "How one surface directs another through the scene host: contract declaration, the kdcube.surface.command envelope, receiver-ready delivery, both ack shapes, and the emitter pattern with a standalone fallback."
status: current
tags: ["sdk", "solutions", "scene", "surfaces", "surface-command", "widgets", "postmessage"]
updated_at: 2026-08-31
keywords:
  [
    "scene surface commands",
    "kdcube.surface.command",
    "kdcube.surface.command.ack",
    "kdcube.surface.ready",
    "command listener readiness",
    "surfaceCommandContracts",
    "target_surface",
    "ui_event payload",
    "command_id ack",
    "connections.hub.open",
    "summon widget",
  ]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/scene/generic-scene-contract-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/scene/scene-surface-registry-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/scene/scene-event-orchestration-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/scene/cross-surface-context-drag-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/components/scene-surface-command-README.md
  - repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/npm/packages/components-core/src/scene/runtime.ts
---
# Scene Surface Commands

A surface command is one surface directing another through the scene host:
"open the Connection Hub at the Slack card", "focus this task in the editor",
"refresh the usage card". The host routes the command to the declared target,
summons the target's window, and both sides acknowledge. This document is the
mechanism reference: declaration, envelope, routing, acks, and the emitter
pattern.

Boundaries with the sibling documents:

- The overall scene design and the full runtime message inventory live in
  [Generic Scene Contract](generic-scene-contract-README.md).
- Provider-resolved object opens (drop a ref, the provider picks the surface)
  live in [Scene Surface Registry](scene-surface-registry-README.md); a surface
  command is the local hop that mechanism also ends with.
- Persistent event relay lives in
  [Scene Event Orchestration](scene-event-orchestration-README.md).
- Moving object refs between surfaces by drag lives in
  [Cross-Surface Context Drag](cross-surface-context-drag-README.md).

The rule:

```text
Being routable on a scene is DECLARED in scene configuration.
An emitter treats the capability as absent until the host acks a command —
and then falls back to its own standalone path.
```

## Declaration

Two declaration styles exist; both key routing off `target_surface` values.

**Contract map (plain-script hosts).** The scene config carries
`scene.surfaceCommandContracts`: a map of contract id → routing entry. The
contract id names the interaction; the entry names the mounted component alias
and the target surfaces it answers for:

```json
{
  "scene": {
    "surfaceCommandContracts": {
      "connections.hub.open": {
        "alias": "connection_hub",
        "targetSurfaces": ["connection_hub.connections", "connection_hub.settings"],
        "action": "open"
      }
    }
  }
}
```

Optional entry fields include `readyType` / `closeType` for widget lifecycle
messages and `pending: true` for a contract-specific pending lane. A component
that receives commands should declare message readiness; in website-style
configuration that is:

```json
"ready": {
  "type": "message",
  "messageType": "kdcube.surface.ready",
  "fallbackDelayMs": 6000
}
```

**Component target surfaces (package hosts).** A scene built on
`@kdcube/components-react/scene` declares `targetSurfaces` directly on the
component spec; the scene runtime registers each one in its surface registry
and routes commands through it:

```ts
{
  alias: 'connection_hub',
  bundleId: 'connection-hub@1-0',
  widgetAlias: 'connections_settings',
  targetSurfaces: ['connection_hub.connections', 'connection_hub.settings'],
  gated: true,
  placement: 'floating',
  // ...
}
```

Both styles express the same fact: this scene routes these `target_surface`
values to this mounted component. A component's own config (the website-style
`components.<alias>.targetSurfaces`) should list the same surfaces, so the
declaration reads consistently on both sides of the mount.

## The Envelope

```json
{
  "type": "kdcube.surface.command",
  "target_surface": "connection_hub.connections",
  "action": "open",
  "command_id": "connhub_1751900000000_ab12cd",
  "widget": "chat",
  "source": "consent-card",
  "ui_event": {
    "tab": "provider_connections",
    "provider": "slack",
    "tiers": ["read", "write"],
    "account_id": "slack_workspace_user"
  }
}
```

- `target_surface` selects the route; `action` names the verb (contract default
  applies when omitted).
- `command_id` (optional) makes the command ack-able; the host echoes it.
- `widget` / `source` identify the emitter for diagnostics.
- `ui_event` carries `tab` plus the target's deep-link query params verbatim —
  the same keys the target widget's URL deep-link path parses (the example
  above shows the provider-connections params; a delegated consent plan rides
  `provider_id` / `connector_app_id` / `claims` the same way).
- **The command payload rides in `ui_event`.** Hosts rebuild the command before
  forwarding and copy only the enumerated envelopes — `ui_event` (verbatim),
  `context`, `object_ref`, `response`, `source`, `view`, `x`/`y`, `reason`.
  A target widget should read its payload from `ui_event` first and accept the
  same keys at the message top level, which covers hosts that relay the raw
  emitter message.

## Host Routing

```text
emitter frame posts kdcube.surface.command to the host
        |
        v
host matches target_surface against declared contracts / registered surfaces
        |
        v
host summons the target: opens the component window, or brings an
already-open window to the front
        |
        v
host keeps the rebuilt command queued
        |
        v
target installs its command listener, then posts kdcube.surface.ready
        |
        v
host marks that source frame ready and forwards the queued command unchanged
```

A matched contract whose component is absent from the current profile is a
visible no-op: the host logs it and, when the command carried a `command_id`,
acks `ok: false` with a code — the emitter's fallback then takes over.

## Receiver Readiness

Readiness means the target can receive the command now. It is an explicit
action by the target widget, not a property inferred from the iframe object:

```json
{
  "type": "kdcube.surface.ready",
  "target_surfaces": [
    "connection_hub.connections",
    "connection_hub.settings"
  ]
}
```

The widget emits this message only after installing its
`kdcube.surface.command` listener. The host validates the source frame and its
origin, marks that frame alias ready, and flushes commands queued for its
declared target surfaces.

`iframe.contentWindow` exists while the frame still contains `about:blank`, so
it proves only that a transport object exists. The iframe `load` event proves
that a document loaded, but not that application code installed its command
listener. A config handshake is a valid readiness policy only when that
component installs its command receiver before completing the handshake.

Hosts may retain a delayed load fallback for older widgets. The shipped hosts
wait six seconds, log that the explicit signal was absent, then release the
queue. This is compatibility recovery; the normal command path is the
receiver's explicit ready message.

## Acks

Two acknowledgements with different jobs:

**Host → emitter** (routing ack; sent only when the command carried a
`command_id`). It tells the emitter the scene owns the open:

```json
{
  "type": "kdcube.surface.command.ack",
  "command_id": "connhub_1751900000000_ab12cd",
  "target_surface": "connection_hub.connections",
  "ok": true,
  "code": ""
}
```

`ok: false` codes: `target_not_mounted` (contract matched, component absent in
this profile), `command_rejected` (the host declined to build the command).
An `ok: true` routing ack may be sent once the host safely owns the queued
command; it does not claim that the target has already applied it.

**Widget → host** (applied ack; diagnostics). After the target widget applies
the command it reports back to its parent:

```json
{
  "type": "kdcube.surface.command.ack",
  "target_surface": "connection_hub.connections",
  "action": "open",
  "reason": "applied",
  "ts": "2026-07-07T18:00:00.000Z"
}
```

When the forwarded command still carries `command_id`, the widget echoes it
(plus `ok`). Hosts log applied acks; routing decisions ride on the host →
emitter ack only.

## Emitter Pattern

The emitter never assumes it is on a routing scene — the capability is
declared, and the ack is the proof:

```text
1. Compose the command with a unique command_id.
2. Post it to the parent frame and arm a short ack timer (~600 ms).
3. Ack with ok:true  -> done; the scene owns the open.
4. Timeout or ok:false -> run the widget's standalone path:
   window.open(<served widget URL, deep-linked>).
```

The standalone URL carries the same intent as the `ui_event` payload (as query
parameters of the served widget), so both paths land the user on the same
state. Reference emitter: `postConnectionsCommandAndAwaitAck` /
`openConnectionsSurface` in
`app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/solutions/chat/ui/widget/src/host.ts`.

## Receiver Pattern

The target widget installs its `kdcube.surface.command` listener, announces
`kdcube.surface.ready`, filters commands by its own `target_surface` values,
applies the payload at runtime (the same state its URL deep-link path
produces), and posts the applied ack. Reference receiver:
[Connection Hub surfaceCommand.ts](https://github.com/elenaviter/app-ecosystem/blob/main/products/connection-hub/apps/connection-hub%401-0/ui/widgets/connections/src/api/surfaceCommand.ts).

## Reference Implementations

| Piece | Where |
| --- | --- |
| Scene runtime routing + readiness queue | `app/ai-app/src/kdcube-ai-app/npm/packages/components-core/src/scene/runtime.ts` (`queueSurfaceCommand`) |
| Package-host scene readiness + routed component | `app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/workspace@2026-03-31-13-36/ui/scene/src/main.tsx` and `sceneConfig.ts` |
| Emitter with ack-wait + standalone fallback | `app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/solutions/chat/ui/widget/src/host.ts` |
| Receiver ready signal + runtime apply + applied ack | Connection Hub `ui/widgets/connections/src/api/surfaceCommand.ts` and `App.tsx` in the app-ecosystem repository |

The plain-script host variant (contract map, ack sender, summon/focus) ships in
the KDCube website scene (`kdcube.config.json` `scene.surfaceCommandContracts`
and component `ready`, `scene-summon.js`, `scene-windows.js`).

The step-by-step walkthrough for app authors — declaring a contract for your
component, receiving, emitting, and the worked consent → hub flow — is the
[Scene Surface Command recipe](../../../recipes/components/scene-surface-command-README.md).
