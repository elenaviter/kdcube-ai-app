---
id: user-memories@2026-06-26
title: "User Memories App"
summary: "A memory-only app: serves the SDK user-memories widget and the `mem` named service, so other apps and scenes embed memories by iframe / consume them as a named service instead of each republishing the memory module."
status: active
tags: ["app", "bundle", "memory", "memories", "widget", "named-service", "economics", "iframe"]
module: entrypoint
singleton: false
primary_surfaces:
  - "memories widget (iframe) — search/add/manage the signed-in user's memories"
  - "named_service `mem` — cross-app memory object contract (search/get/create/update/delete)"
  - "MCP endpoint `memories` — delegated external-client access to memory_search/memory_get"
links:
  config: config/bundles.template.yaml
  secrets: config/bundles.secrets.template.yaml
  interface: interface/README.md
  design: docs/README.md
---

# User Memories App

This app exposes the user's memories **once**, as a standalone surface, so other
apps don't each embed and republish the memory module. It is the cleaner
paradigm: one app owns the memory widget + the `mem` named service; everyone
else **embeds the widget by iframe** and **consumes `mem` as a named service**.

It is intentionally tiny. The whole behavior comes from deriving the SDK
**memories + economics** mixin; this app only enables the memory widget and
points the build at the shared SDK widget source. It ships no UI of its own.

## What it serves

- **`memories` widget** — the SDK memories widget, built from
  `sdk://context/memory/ui/widget/memories`. Reachable at
  `…/bundles/{tenant}/{project}/user-memories@2026-06-26/widgets/memories`.
- **`mem` named service** — this owner app explicitly enables
  `memory.named_service.enabled`; enabling embedded memory alone does not
  publish the namespace. Other apps resolve/search/mutate memory objects
  through this provider without republishing the module.
- **`memories` MCP endpoint** — a public proc-served MCP endpoint guarded by
  Connection Hub delegated credentials. It requires authority `delegated_client`; each
  exposed MCP tool declares its required grant (`memories:read` for
  `memory_search` and `memory_get`) and must also be selected during consent.
  This is the reference target for connecting an external Claude client on
  behalf of a regular KDCube user. The `KDCubeMCPServer` app is served with
  `stateless_http=True`, because the proc bridge can dispatch MCP requests
  through fresh app instances. It accepts modern discovery and legacy
  initialization clients through the same surface.

Memory writes are economics-guarded (reconciliation reserves budget) via the
economics half of the mixin.

## How it works

The entrypoint derives `BaseEntrypointWithEconomicsAndMemory`
(`kdcube_ai_app.apps.chat.sdk.solutions.chatbot.entrypoint_with_memory`), which
already wires the memory widget operations, the `mem` named-service provider,
reconciliation, snapshots, and the economics guard. `configuration_defaults()`
turns the widget on, explicitly enables named-service publication, and sets its
scope; `memory_configuration_defaults()` from the mixin fills in everything
else.

There is **no `ui/` folder** — the memories widget is built from the SDK source
shared by every app, so there is a single copy to maintain.

## Layout

```text
user-memories@2026-06-26/
  README.md
  AGENTS.md
  release.yaml
  entrypoint.py            # derives BaseEntrypointWithEconomicsAndMemory; enables the widget
  memory_mcp_tools.py      # MCP server exposing memory_search and memory_get
  __init__.py
  config/
    bundles.template.yaml          # non-secret deployment props
    bundles.secrets.template.yaml  # secret keys (no values)
  interface/
    README.md              # operations, widget, named service, storages + dataflows
  docs/
    README.md              # design + storages + dataflows rationale
```

## Run it locally

Add the descriptor item from `config/bundles.template.yaml` to your runtime
`bundles.yaml`, then `kdcube bundle reload`. The platform builds the memories
widget from the SDK source and serves it at the widget URL above.

## Roadmap (step by step)

1. **This app** — serve the memories widget + `mem` named service. ✅
2. **`workspace`** — stop embedding the memory module; consume `mem` as a named
   service only, and embed this app's widget by iframe (incl. the Telegram web
   app) instead of republishing it.
3. **Site scene** (`website/index.html` + `kdcube.config.json`) — point the
   memories component at this app instead of `workspace`.

> Terminology: this is an **app** (the platform calls the deployable unit a
> "bundle" in code — `bundle_id`, `bundles.yaml`, `@bundle_entrypoint`). The two
> mean the same thing during the rebrand.
