# KDCube Example Apps

This directory contains versioned, runnable KDCube app packages. The runtime
and descriptor APIs still call these packages **bundles**, so in this directory:

```text
KDCube app = runtime bundle = one deployable app package
```

The examples are worked implementations, not a substitute for the SDK
contracts. Start with the canonical guidance, then choose the narrowest example
that matches the surface you are building.

## Read first

1. [What I Should Know Before Writing a KDCube App](../../../../../../../../docs/recipes/what-i-should-know-about-app-README.md)
   - compact map of app surfaces, runtime boundaries, identity, context,
     storage, delivery, economics, UI, and maintenance.
2. [How To Write A KDCube Bundle](../../../../../../../../docs/sdk/bundle/build/how-to-write-bundle-README.md)
   - authoritative package shape, lifecycle, entrypoint rules, async runtime
     contract, configuration, access control, and validation.
3. [How To Assemble A Bundle With SDK Building Blocks](../../../../../../../../docs/sdk/bundle/build/how-to-assemble-bundle-with-sdk-building-blocks-README.md)
   - choose existing SDK capabilities before implementing another copy.
4. [How To Configure And Run A Bundle](../../../../../../../../docs/sdk/bundle/build/how-to-configure-and-run-bundle-README.md)
   - descriptor integration, local loading, reload, and runtime checks.
5. [How To Avoid Common Bundle Integration Failures](../../../../../../../../docs/sdk/bundle/build/how-to-avoid-common-bundle-integration-failures-README.md)
   - recurring package, identity, configuration, UI, and runtime mistakes.
6. [How To Release Bundle Content](../../../../../../../../docs/sdk/bundle/build/how-to-release-bundle-content-README.md)
   - release metadata and content-release procedure.

## Production-shaped references

### Workspace

[workspace@2026-03-31-13-36](workspace%402026-03-31-13-36/README.md) is the
broad integration reference. Use it when building a conversational app that
combines several platform surfaces:

- ReAct agent and gate;
- chat widget and scene/main-view composition;
- tools, MCP consumers, and named-service consumers;
- economics and accountable execution;
- Telegram webhook and delivery integration;
- app interface, configuration templates, storage documentation, scenarios,
  release metadata, and builder-agent guidance.

Workspace is intentionally large. Copy only the surfaces your app owns; do not
use its all-features shape as the minimum package.

### Ported LangGraph Agents

[ported-langgraph-agents@2026-07-13](ported-langgraph-agents%402026-07-13/README.md)
is the reference for bringing existing LangGraph or LangChain agents into one
KDCube app while keeping solution code separate from platform integration. It
demonstrates:

- one app hosting multiple agents selected by `agent_id`;
- framework-specific stream adapters into the KDCube conversation record;
- bound identity, shared Postgres, economics, tools, files, and Telegram;
- rebuildable per-turn graph state suitable for scaled processors;
- run-to-completion event handling at the start of a hosted-agent turn.

Use the accompanying
[settling recipe](../../../../../../../../docs/recipes/apps/settle-your-solution-in-kdcube-README.md)
as the contract; use this app as the worked implementation.

### User Memories

[user-memories@2026-06-26](user-memories%402026-06-26/README.md) is a focused
provider app. It owns the reusable memory experience once and exposes it to
other apps through:

- the shared SDK memories widget;
- the `mem` named-service provider;
- a delegated MCP surface for external clients;
- economics-guarded memory mutation.

Use it to understand a small app that primarily provides reusable services and
UI, rather than a chat agent.

### Website

[website@2026-07-12](website%402026-07-12/README.md) is the reference for an
app-owned public website and `ui.main_view`. It demonstrates host/default site
routing, platform frontend configuration, a scene shell, and separation between
app routes and platform/API routes.

## Platform service references

These are production-shaped apps, but they implement KDCube platform services.
Read them when building the corresponding infrastructure; do not use them as
generic starter apps.

### Connection Hub

[connection-hub@1-0](https://github.com/elenaviter/app-ecosystem/tree/main/apps/connection-hub%401-0) owns user-scoped identity
and connection boundaries:

- external identity links and principal resolution;
- request authenticators;
- accounts delegated to KDCube and access delegated by KDCube;
- OAuth callbacks, grants, and consent-demand completion;
- the `connections` named-service provider;
- the Connections settings widget.

### KDCube Services

[kdcube-services@1-0](kdcube-services%401-0/README.md) owns KDCube-managed
surfaces that may be exposed to delegated external clients:

- conversations and configured named services over MCP;
- signed file transfer;
- the storage browser and platform administration widgets;
- Connection Hub delegated-credential enforcement.

## Focused fixtures

These examples isolate one mechanism. They are useful for implementation and
regression testing, but they predate parts of the current package standard and
must not be copied as complete app skeletons.

| Fixture | Use it for | Do not infer |
| --- | --- | --- |
| [node.bridge.mcp@2026-04-24](node.bridge.mcp%402026-04-24/README.md) | A Python-owned app surface calling a process-local Node/TypeScript sidecar and exposing selected logic through API/MCP. | That every Node integration needs a new public transport or that its compact package is the current full app contract. |
| [with-isoruntime@2026-02-16-14-00](with-isoruntime%402026-02-16-14-00/README.md) | Direct isolated-execution diagnostics without ReAct. | General app structure, normal conversational execution, or permission to pass host state into the untrusted executor. |
| `echo.ui@2026-03-30` | Smoke-testing a custom main-view build, a simple API, cron registration, and a no-LLM echo graph. | Production package documentation, current configuration completeness, or an all-purpose starter app. |

For Node-specific architecture, also read
[Node Backend Sidecar](../../../../../../../../docs/sdk/node/node-backend-sidecar-README.md)
and
[Bundle Node Backend Bridge](../../../../../../../../docs/sdk/bundle/bundle-node-backend-bridge-README.md).

## How to inspect an example

For a production-shaped app, read the package in this order:

```text
README.md                 product purpose and owned surfaces
AGENTS.md                 builder/maintainer onboarding
release.yaml              release identity and history
interface/                external operations and data contracts
config/                   descriptor and secret templates
entrypoint.py             composition root only
agents/, services/, ...   modular implementation
docs/storage/             ownership, retention, and cleanup
docs/journal/             significant implementation decisions
tests/                    executable behavior and regression coverage
```

Not every app needs every optional folder, but every declared surface must be
represented consistently in code, interface documentation, configuration,
release metadata, and tests.

## Rules that apply to every example

- App handlers run in the concurrent proc environment. I/O must be async;
  synchronous database, network, filesystem, or subprocess work must not block
  the shared event loop.
- Identity, tenant, project, user, app, conversation, and delegation context
  come from bound runtime context. Do not accept authoritative identity from a
  model or arbitrary request payload.
- Use descriptor-backed app properties and secrets through SDK helpers. Do not
  introduce app configuration through environment variables, mutable globals,
  or hardcoded descriptor paths.
- Treat `singleton` as instance reuse, not as global-state safety or operation
  exclusivity.
- Keep provider declarations under `surfaces.as_provider` and consumed
  capabilities under `surfaces.as_consumer`.
- Choose the correct delivery contract: conversation event bus, Data Bus,
  background job, cron, webhook, and direct API calls have different ownership
  and ordering semantics.
- Reuse SDK widgets, tools, named-service contracts, MCP support, storage,
  economics, telemetry, and communication primitives before adding parallel
  implementations.
- Validate the real configured app through its actual API, widget, MCP, event,
  job, or chat surface. Import-only tests are not sufficient.
