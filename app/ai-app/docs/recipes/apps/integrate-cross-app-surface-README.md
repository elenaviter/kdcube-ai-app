---
id: repo:kdcube-ai-app/app/ai-app/docs/recipes/apps/integrate-cross-app-surface-README.md
title: "Integrate One KDCube App With Another"
summary: "Builder recipe for choosing and implementing the correct same-KDCube app-to-app contract: direct API operations, named services, Data Bus, jobs, conversation ingress, MCP, REST, and UI composition."
status: active
tags: ["recipes", "apps", "interoperability", "api", "named-services", "data-bus", "mcp"]
updated_at: 2026-08-26
keywords:
  [
    "KDCube app integration",
    "call_bundle_operation",
    "cross app API",
    "named service app integration",
    "Data Bus app integration",
    "same KDCube MCP",
    "cross KDCube app call",
  ]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/cross-app-surface-interoperability-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/cross-kdcube-app-surface-interoperability-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/apps/govern-provider-surfaces-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/apps/consume-mcp-service-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/apps/expose-mcp-service-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/providers-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/service/comm/data-bus-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/service/streams/background-jobs-README.md
---
# Integrate One KDCube App With Another

Use this recipe when app A needs a capability owned by app B. Both apps may
provide several independent surfaces. Choose the surface whose delivery and
authority contract matches the work, then call that surface through its native
KDCube path.

This recipe implements the **same-KDCube** case: both apps run under one
effective tenant/project runtime. Calls to an app in another KDCube use a
network surface and a credential accepted by the target deployment; that
boundary is covered in [Call An App In Another
KDCube](../../runtime/cross-kdcube-app-surface-interoperability-README.md).

KDCube retains **bundle** in code and identifiers such as `bundle_id`,
`bundles.yaml`, and `@bundle_entrypoint`. In this recipe, app and bundle refer
to the same deployable unit.

```text
app A needs app B
       |
       v
choose the work contract
       |
       +-> immediate result          -> local app operation
       +-> reusable object language  -> named service
       +-> durable domain mutation   -> Data Bus
       +-> ready background work     -> job stream
       +-> ordered agent turn        -> conversation ingress
       +-> MCP compatibility         -> MCP transport
       +-> HTTP compatibility        -> REST transport
       +-> browser composition       -> scene/widget
```

## 1. Choose The Contract Before Writing The Caller

| What app A needs | App B provides | App A uses |
| --- | --- | --- |
| One bounded result during the current request | `@api` | `call_bundle_operation(...)` |
| A file or byte stream during the current request | streaming `@api` | `call_bundle_operation_stream(...)` |
| A provider-neutral vocabulary for refs, search, schemas, objects, and actions | named-service provider | configured named-service discovery and client |
| A durable command or event that survives the caller | `@data_bus_handler` | `data_bus_publish(...)` or `data_bus_publish_and_wait(...)` |
| Ready background work owned by app B | `@on_job` | background job stream addressed to app B |
| A new ordered event for app B's agent | reactive entrypoint | conversation ingress |
| MCP discovery and tool-call compatibility | `@mcp` | configured MCP client transport |
| An HTTP contract, webhook shape, or cross-deployment endpoint | `@api` | authenticated REST through OpenResty |
| A browser surface owned by app B | `@ui_widget` | scene/widget composition |

The full runtime reasoning behind this table lives in [Cross-App Surface
Interoperability](../../runtime/cross-app-surface-interoperability-README.md).
This recipe focuses on the builder steps.

## 2. Call A Bounded App Operation

Use the local operation bridge when app A is handling a request with a bound
caller and needs one immediate result from app B.

App B declares the operation on its entrypoint and delegates domain work to a
focused service:

```python
from __future__ import annotations

from typing import Any

from kdcube_ai_app.infra.plugin.bundle_loader import api


class TaskTrackerEntrypoint:
    @api(method="POST", alias="issue_get", route="operations")
    async def issue_get(
        self,
        *,
        issue_id: str,
        **request_fields: Any,
    ) -> dict[str, Any]:
        return await self.issues.get_visible_issue(
            issue_id=issue_id,
            request_fields=request_fields,
        )
```

App A calls the alias, not app B's Python method name:

```python
from __future__ import annotations

from typing import Any

from kdcube_ai_app.apps.chat.sdk.infra.bundle_operations import (
    call_bundle_operation,
)


async def load_issue(issue_id: str) -> dict[str, Any]:
    result = await call_bundle_operation(
        bundle_id="task-tracker@1-0",
        operation="issue_get",
        route="operations",
        http_method="POST",
        data={"issue_id": issue_id},
    )
    return dict(result)
```

The runtime resolves app B's current version and effective props, checks the
target app and operation policy, creates app B's request context from the bound
caller, awaits the operation, and returns its result. App B still applies its
domain and record-level authorization.

Normally omit `tenant` and `project`; the bridge inherits the current
tenant/project scope. Treat `data` as operation input only. Caller identity is
provided by the target request context, so app A does not copy cookies, bearer
tokens, user ids, roles, or credentials into the payload.

The bridge supplies `user_id` and `fingerprint` compatibility fields to the
target method. Accept them explicitly or keep `**request_fields` in the
operation signature. Product authorization should read the bound request
context rather than trust those keyword values as proof.

`call_bundle_operation(...)` is available while KDCube has bound a local
operation caller to the current request or task. A free-standing script has no
such caller and receives:

```text
No request-bound bundle operation caller is available
```

Headless work carries explicit saved authority through its job or service
contract. A call to another KDCube uses REST or MCP instead of this local
bridge.

### Stream A File Or Byte Response

When app B's operation returns a platform file, binary, or stream response,
app A uses the stream bridge:

```python
from kdcube_ai_app.apps.chat.sdk.infra.bundle_operations import (
    call_bundle_operation_stream,
)


result = await call_bundle_operation_stream(
    bundle_id="reporting@1-0",
    operation="report_download",
    route="operations",
    data={"report_id": "report-2026-08"},
)

async for chunk in result.chunks:
    await consume(chunk)
```

The result also carries `filename`, `media_type`, response headers, and status.
Keep large content streamed or referenced rather than converting it to inline
JSON.

## 3. Call An Owner-Defined Named Service

Use a named service when several consumers should work with app B through an
owner-defined vocabulary such as `object.search`, `object.get`,
`object.schema`, or `object.action`. App B owns the namespace, refs, schemas,
capabilities, and authorization. App A owns which namespace operations its
agent, UI, or service may use.

App B registers a provider through `@named_service_provider(...)` and exposes
it from `_named_service_providers()`. App A configures the relevant consumer
surface under `surfaces.as_consumer`: for example an agent tool, a canvas
resolver, or an event source. The platform publishes and resolves provider
discovery for the effective tenant/project.

Request-bound app code can call the resolved provider endpoint:

```python
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers import (
    NamedServiceAdmission,
    NamedServiceEndpoint,
    call_named_service_endpoint,
)


response = await call_named_service_endpoint(
    NamedServiceEndpoint(namespace="task"),
    {
        "operation": "object.search",
        "namespace": "task",
        "query": "blocked authentication issues",
        "limit": 20,
    },
    admission=NamedServiceAdmission.application(
        source="task_dashboard.blocked_issue_search",
    ),
)

if not response.ok:
    return response.to_dict()

items = response.ret.get("items", [])
```

`admission` is required and names who owns authority for this invocation.
Application code selects `NamedServiceAdmission.application(...)` positively at
its trusted call site. A delegated entrance, including a hosted-agent tool or a
managed MCP bearer, constructs delegated admission from its current Connection
Hub card; missing delegated state never falls back to application authority.
Admission is separate from `NamedServiceRequest.context`, provider input, and
caller identity.

This is a required SDK signature change. Existing two-argument calls raise a
`TypeError`; migrate each trusted call site by selecting its positive
application or delegated admission mode.

Discovery selects the current provider and one of the configured local
transports:

```text
bundle_registry   load app B and call its named_services() registry
bundle_operation  call app B's @api(alias="named_service") facade
module            load an explicit provider module in the same runtime
```

An isolated executor reaches the same capability through the trusted Data Bus
relay. Provider credentials and storage clients remain on the trusted side.

See [Named Service Providers](../../sdk/namespace-services/providers-README.md)
for provider implementation, consumer configuration, operation schemas, and
streaming named-service responses.

## 4. Send Durable Work Through Data Bus

Use Data Bus when app B must own a mutation after app A's request, worker, or
browser has gone away. Delivery is at least once, so app B owns durable
idempotency and revision checks.

App B declares the subject and handler:

```python
from __future__ import annotations

from typing import Any

from kdcube_ai_app.apps.chat.sdk.runtime.data_bus import data_bus_handler


class TaskTrackerEntrypoint:
    @data_bus_handler(
        subject="task.issue.rename",
        partition_by="object_ref",
        ordering="serial_per_partition",
        idempotency="required",
    )
    async def rename_issue(self, ctx, message) -> dict[str, Any]:
        result = await self.issues.rename_once(
            issue_id=message.payload["issue_id"],
            title=message.payload["title"],
            idempotency_key=message.idempotency_key,
        )
        await ctx.reply.ok({"revision": result.revision})
        return {"status": "ok", "data": {"revision": result.revision}}
```

App A publishes through its current communicator. Derive `message_id` and
`idempotency_key` from a stable domain request id so a retry identifies the
same mutation:

```python
from kdcube_ai_app.apps.chat.sdk.runtime.comm_ctx import (
    data_bus_publish_and_wait,
)


async def rename_issue(
    *,
    request_id: str,
    issue_id: str,
    title: str,
) -> dict:
    return await data_bus_publish_and_wait(
        bundle_id="task-tracker@1-0",
        subject="task.issue.rename",
        object_ref=f"task:issue:{issue_id}",
        message_id=f"task_issue_rename_{request_id}",
        idempotency_key=f"task_issue_rename_{request_id}",
        reply=True,
        timeout_ms=20_000,
        payload={"issue_id": issue_id, "title": title},
    )
```

`publish_and_wait(...)` waits for the correlated handler result; the message
remains a durable Data Bus record. `publish(...)` returns after admission when
app A does not need the completed result. A live reply may update a connected
UI, while app B's durable state remains the authority.

See [Data Bus](../../service/comm/data-bus-README.md) for handler visibility,
partitioning, retries, dead-letter behavior, and client ingress.

## 5. Use MCP When MCP Is The Product Contract

Use MCP when the consumer needs MCP discovery, tool schemas, `tools/call`, or
compatibility with MCP clients. The consuming app configures a real MCP
transport; the providing app declares `@mcp(...)`.

```text
app A MCP client
  -> configured streamable HTTP, SSE, or stdio transport
  -> target MCP authentication and current grant checks
  -> app B @mcp surface
```

For two apps in one KDCube, a streamable-HTTP connection can use the
deployment-provided private OpenResty address. Keep the URL descriptor-owned;
each deployment supplies its own service address. Managed MCP also has a
protected `resource` identity, which must match the canonical request path
observed by the target guard.

The same-KDCube SDK currently keeps MCP as a protocol path. The MCP consumer
uses the configured transport, while `call_bundle_operation(...)` supplies the
short local path for an ordinary bounded app operation.

Follow these two focused recipes:

- [Expose An MCP Service From A KDCube App](expose-mcp-service-README.md)
- [Connect An MCP Service To A KDCube Agent](consume-mcp-service-README.md)

## 6. Use REST When HTTP Semantics Matter

Use the target app's REST surface when app A needs HTTP status, headers,
webhook compatibility, app-owned authentication, or the same transport used
by an external caller.

Inside one deployment, the trusted application runtime reaches OpenResty
through the deployment's configured private address. OpenResty authenticates
the request and constructs app B's context again. The network route supplies
reachability; app B's API policy and domain authorization supply permission.

Across KDCube deployments, REST and MCP enter through KDCube B's accepted
ingress with a credential accepted by KDCube B. The current implemented
baseline supports calls with a preconfigured target credential. Automatic
source-side remote discovery, browser consent, OAuth/PKCE exchange, per-agent
credential storage, and retry are tracked as the cross-KDCube delegated-client
feature in [issue #223](https://github.com/kdcube/kdcube/issues/223).

## 7. Route Jobs, Agent Turns, And Browser Surfaces Deliberately

Three other contracts complete app composition:

| Intent | Integration path | Completion meaning |
| --- | --- | --- |
| App B owns ready background work | enqueue the background job with app B's `bundle_id` | app B's async `@on_job` owns execution and durable result |
| App B's agent should receive an ordered event | submit through conversation ingress for app B and its conversation | submission returns lane admission; a later worker runs the turn |
| App B owns part of the browser experience | mount app B's `@ui_widget` through scene configuration | the widget calls app B's authenticated backend surfaces |

Use conversation ingress for an agent event rather than calling an agent loop
as a Python method. Use the job stream for ready app-owned work rather than
holding an HTTP request open. Use communicator events for live delivery to a
connected client; keep durable work in Data Bus, jobs, or the conversation
lane.

See [Background Jobs](../../service/streams/background-jobs-README.md),
[Servicing Interfaces](../../service/servicing-interfaces-README.md), and
[Scene Composition](../../sdk/solutions/scene/scene-composition-README.md).

## 8. Preserve Identity And Recheck Authority

Each path carries caller facts according to its own contract:

```text
local operation / named service
  current bound caller -> target app context -> target checks

Data Bus / job / conversation ingress
  explicit durable actor and authority metadata -> worker rebuilds context

MCP / REST
  target-accepted credential -> target ingress authenticates -> target checks
```

Follow these rules in app code:

1. Read identity from the bound request, Data Bus, job, or ingress context.
2. Let app B enforce its endpoint policy and domain ownership checks.
3. Keep provider secrets in server-side secret references and trusted services.
4. Make durable and side-effecting handlers idempotent.
5. Treat a provider id, object ref, endpoint URL, or resource handle as an
   identifier; authorization remains a separate check.

## 9. Verify The Integration Over Its Real Path

1. Run the shared bundle contract suite for app A and app B.
2. Start or refresh a local KDCube with both app versions and their effective
   descriptors.
3. Invoke app A through its real browser, API, job, or agent entrypoint.
4. Verify app B receives the expected caller identity and app-local context.
5. Disable or narrow app B's target surface and confirm the next call is
   rejected before domain work runs.
6. For Data Bus, submit the same idempotency key twice and confirm the durable
   mutation is applied once.
7. For MCP, verify discovery and one allowed tool call through the configured
   transport, then verify an ungranted tool is denied.
8. For streams or files, verify media type, filename, chunk delivery, and the
   final consumer artifact.
9. Inspect app and runtime logs for the source app id, target app id, operation
   or subject, and the represented caller.

The integration is complete when the real target surface, current descriptor
policy, identity projection, failure response, and delivery semantics all
match the contract selected in step 1.
