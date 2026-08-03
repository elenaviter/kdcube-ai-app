---
id: repo:kdcube-ai-app/app/ai-app/docs/recipes/components/named-service-README.md
title: "Build A Complete Named Service"
summary: "End-to-end recipe for publishing an app-owned domain as a discoverable, governed named service that external MCP clients, hosted agents, UI surfaces, and harness materializers can use progressively."
status: active
tags: ["recipes", "apps", "bundles", "named-services", "agents", "mcp", "react", "connections", "consent"]
updated_at: 2026-08-03
keywords:
  - named service provider
  - ontology-guided agent interface
  - progressive schema projection
  - provider discovery
  - object refs
  - complete object retrieval
  - connected account claims
  - external MCP agent
  - ReAct pull materialization
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/providers-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/clients-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/discovery-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/react-object-materialization-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/index/hybrid-index-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/apps/named-services-mcp-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/apps/consume-mcp-service-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/create-delegated-automation-access-README.md
---
# Build A Complete Named Service

A named service gives agents and user interfaces a common way to discover and
operate a domain. The provider app (bundle) owns the domain vocabulary, object
refs, schemas, actions, authorization requirements, and complete-data path.
Consumers use that contract without importing the provider's storage or
provider-specific SDK.

This is KDCube's ontology-guided interface for connected domains:

```text
provider.about          what this domain is for
provider.capabilities   what this deployment can do now
object.schema           browse/search catalog -> exact operation contract
object.list/search      discover objects and receive stable refs
object.get              inspect or retrieve an object by ref
object.action           run one named, bounded effect
object.upsert/delete     mutate when the provider supports it
object.host_file         move a caller-owned file into provider ownership
event.resolve            route a provider ref into the agentic harness
block.produce/render     project provider-owned state for model/UI readers
```

An app can also expose REST, MCP, widgets, jobs, Event Bus, or Data Bus
surfaces directly. Use a named service when unrelated clients should learn a
domain progressively and operate it through stable refs.

## 1. Design The Agent Journey First

An unfamiliar agent should be able to follow this sequence without prior
KDCube or provider knowledge:

```text
about -> schema root -> browse/search capabilities -> exact contract
      -> list/search objects -> get -> bounded action
```

The provider must answer five questions:

1. What domain is this?
2. What object kinds and operations exist here?
3. How do I find an object and retain its stable ref?
4. How do I retrieve all of the selected object's data when the normal result
   is intentionally compact?
5. Which caller grant and connected-account consent does each operation need?

The generic client tools stay small because the schema carries the domain
knowledge. Adding another namespace does not require adding another flat tool
catalog to every agent.

## 2. Put The Provider In The Owning App

Keep the app entrypoint thin. A typical provider package is:

```text
entrypoint.py                         composition and surface declarations
services/<domain>/                    domain and provider API adapters
services/<domain>/named_service.py    provider spec + async operations
interface/                            published contracts
docs/                                 app-specific behavior and storage
tests/                                provider, policy, transport, consent tests
```

The owner app contributes its provider explicitly:

```python
def _named_service_providers(self) -> list:
    return [
        *super()._named_service_providers(),
        self._domain_provider(),
    ]
```

The app publishes its complete current registry during load. Removing the
provider contribution withdraws that app's stale discovery record. A reusable
mixin may implement provider helpers, but the concrete owner decides whether
to publish them. See [Discovery Registry](../../sdk/namespace-services/discovery-README.md).

All provider methods are `async`. Do not run synchronous network, database,
filesystem, or provider-SDK work in the proc event loop. Use an async client,
an app-owned `@venv` operation, or another existing asynchronous KDCube
boundary. Keep runtime state in descriptors, user properties/secrets, or
documented storage; do not use environment variables or process globals.

## 3. Declare One Authoritative Provider Spec

The provider spec is both runtime metadata and agent education. Keep the
decorator metadata and the instance spec synchronized when both are present;
the instance returned by the owner app is the runtime authority.

```python
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers import (
    NamedServiceContext,
    NamedServiceProvider,
    NamedServiceProviderSpec,
    NamedServiceRequest,
    NamedServiceResponse,
    named_service_provider,
)


def domain_named_service_spec(*, bundle_id: str | None = None):
    return NamedServiceProviderSpec(
        provider_id="acme.records",
        bundle_id=bundle_id,
        namespace="records",
        refs=("records:*",),
        object_kinds=("records.collection", "records.item"),
        operations={...},
        label="Records",
        description="Search and operate approved business records.",
        intro="Start with capabilities and schema, then search for a ref.",
        metadata={
            "canonical_refs": {
                "collection": "records:collection:<id>",
                "item": "records:item:<id>",
            },
            "grant_hints": {...},
            "presentation": {...},
            "object_kinds": {...},
            "actions": {...},
        },
    )


@named_service_provider(
    provider_id="acme.records",
    namespace="records",
    refs=("records:*",),
    object_kinds=("records.collection", "records.item"),
    operations={...},
    label="Records",
    description="Search and operate approved business records.",
    intro="Start with capabilities and schema, then search for a ref.",
    metadata={...},
)
class RecordsNamedServiceProvider(NamedServiceProvider):
    def __init__(self, *, bundle_id: str | None = None) -> None:
        super().__init__(domain_named_service_spec(bundle_id=bundle_id))

    async def provider_about(self, ctx, request): ...
    async def provider_capabilities(self, ctx, request): ...
    async def object_schema(self, ctx, request): ...
    async def object_search(self, ctx, request): ...
    async def object_get(self, ctx, request): ...
    async def object_action(self, ctx, request): ...
```

Use SDK absolute imports. Inside an app package, use package-relative imports
for the app's own modules.

## 4. Make Refs Round-Trip

Every result intended for later use carries a canonical `object_ref`:

```text
records:collection:<collection_id>
records:item:<item_id>
```

The provider owns the grammar. Consumers treat refs as opaque handles and
pass them back unchanged. A ref returned by search must work in `object.get`,
applicable actions, UI resolution, and materialization without hidden ambient
state.

Search results should be lean and selectable:

```json
{
  "ref": "records:item:42",
  "object_ref": "records:item:42",
  "object_kind": "records.item",
  "title": "Quarterly forecast",
  "summary": "Updated 2026-07-27",
  "score": 0.91
}
```

Put the full object envelope on `object.get`, not on every search hit. Return
`next_cursor` whenever more provider results exist. Accept that cursor on the
next `object.list` or `object.search` request. With multiple eligible provider
accounts, return an actionable `account_required` response and labeled account
candidates; do not silently pick one.

## 5. Teach Both The Agent And The User

`provider.about` and `object.schema` are the agent reader. Spec presentation
metadata is the human reader shown in capability and consent surfaces.

At minimum declare:

```text
metadata.presentation.about
metadata.presentation.works_with or third_party
metadata.presentation.operations.<operation>.label + description
metadata.presentation.actions.<action>.label + description
metadata.object_kinds.<kind>
metadata.grant_hints.<operation or exact action>
```

`provider.capabilities` reports what is actually wired in this deployment.
Do not advertise an operation that will return "not implemented." The schema
should explain filters, defaults, ref shapes, action payloads, pagination,
revision/idempotency behavior, and the complete retrieval path.

### Project A Large Schema Before It Reaches The Agent

Keep one complete provider schema as the source of truth, then assign its
parts to kinds in `schema_projection_index`:

```python
RECORDS_SCHEMA_PROJECTION = {
    "catalog": {
        "id": "records",
        "label": "Records",
        "children": [
            {
                "id": "browse",
                "label": "Find and inspect records",
                "children": [
                    {
                        "id": "collections",
                        "label": "Collections",
                        "object_kind": "records.collection",
                        "operations": ["object.list", "object.get"],
                    },
                    {
                        "id": "items",
                        "label": "Items",
                        "object_kind": "records.item",
                        "operations": ["object.search", "object.get"],
                    },
                ],
            },
            {
                "id": "lifecycle",
                "label": "Create and manage records",
                "keywords": ["publish", "archive"],
                "object_kind": "records.item",
                "operations": [
                    "object.upsert",
                    "object.action:publish",
                    "object.action:archive",
                ],
            },
        ],
    },
    "kinds": {
        "records.collection": {
            "refs": ["collection"],
            "operations": {
                "object.list": {},
                "object.get": {"sections": ["get"]},
            },
        },
        "records.item": {
            "refs": ["item"],
            "selectors": ["item_selector"],
            "related_kinds": ["records.collection"],
            "operations": {
                "object.search": {"sections": ["search"]},
                "object.get": {"sections": ["get", "materialization"]},
                "object.upsert": {
                    "sections": ["upsert"],
                    "section_keys": {"upsert": ["item"]},
                },
            },
            "actions": ["publish", "archive"],
        },
    },
}


class RecordsNamedServiceProvider(NamedServiceProvider):
    schema_projection_index = RECORDS_SCHEMA_PROJECTION
```

Implement `schema_object_kind_from_ref()` with the provider's canonical ref
parser. The shared dispatcher then validates the index and applies these views
to both `provider.about` and `object.schema`:

```text
namespace only                  -> root catalog node
schema_path="/browse"           -> one recursive catalog node
query="publish a record"        -> matching capability declarations
object_kind or object_ref       -> one kind and operation summaries
schema_operation               -> one exact executable contract
schema_view="full"              -> complete schema, explicitly requested
```

This projection reduces model context; it does not invent the domain model.
Map the backing provider's endpoint inventory into user-meaningful objects and
bounded actions first. Several API calls may implement one action. Distinct
effects that need separate policy remain distinct actions.

The `catalog` tree is provider-owned and may be nested to any useful depth.
Node ids form stable `schema_path` values. Labels, descriptions, and keywords
make capability search useful without placing every payload contract in the
model context. A query such as
`object.schema(namespace="records", query="publish a record")` searches those
capability declarations and returns `catalog_path`, `object_kind`, and
`schema_operation` selectors. The caller then requests the exact operation
contract before invoking it.

Capability search and object search are separate:

```text
object.schema(query=...)   find what this provider knows how to do
object.search(query=...)   find the provider-owned objects to operate on
```

The bundle that publishes the provider owns index preparation. Its base
entrypoint sees only the providers contributed by that bundle, binds the
bundle's shared storage and embedding service, and prepares projection-enabled
catalogs during `on_bundle_load`. Consumer bundles and agents query the
provider; they do not build or write its index. Providers without
`schema_projection_index` keep their existing full-schema behavior.

When an embedding service is bound, the provider persists a compact hybrid
capability index in its bundle storage. Use the persistent production backend
in the publishing bundle's descriptor:

```yaml
config:
  named_services:
    schema_search:
      vector_backend: faiss-local
```

This produces a profile-specific SQLite file containing declarations, FTS,
and cached vectors, plus a derived `.faiss` sibling shared by workers. Set
`vector_backend: bruteforce` explicitly only for a dependency-free development
or test runtime; it reconstructs the vector view in each process and does not
write a `.faiss` file. A deterministic catalog signature lets other workers
reuse the index and refreshes it when the provider declaration changes.
Without an embedding service, no persistent hybrid index is required: lexical
matching runs directly over the declared projection. The response reports the
requested and effective mode, selected vector backend, and any fallback reason.

Only capability declarations enter this index. Provider objects remain in the
provider realm. Embedding provider, model, vector-dimension, and vector-backend
changes also change the cached capability-index identity without changing the
provider's object-search path. Profile-specific shared paths keep
rolling-deployment workers from rewriting one another's vectors. Do not add an
LLM role to a provider app solely for schema discovery: the lexical path
remains available when no embedding service is bound.

## 6. Keep Normal Results Compact And Data Reachable

Compact model results and complete retrieval are separate concerns. A provider
may summarize a large message, workbook, history, or file in `object.get`, but
it must not make the rest of the authorized data unreachable.

Use this hierarchy:

```text
small structured object
  -> return inline

long collection
  -> return a page + next_cursor

large complete object or binary
  -> return metadata + a short-lived signed KDCube delivery URL

harness materialization request
  -> object.get(response_mode=stream)
  -> NamedServiceStreamResult(response=<compact sidecar>, chunks=<async bytes>)
```

The client decides how much retrieved data enters its model context. The
provider controls provider/API safety limits per request, reports those limits,
and offers continuation or complete delivery when the backing provider allows
it.

### External Client Delivery

A signed URL for a connected provider is normally a live KDCube delivery
proxy, not a previously hosted artifact:

```text
external client GET signed KDCube URL
  -> verify signature, expiry, exact ref, user, tenant, and project
  -> resolve the user's current connected-account credential server-side
  -> enforce current provider consent
  -> fetch current provider data
  -> stream bytes or a complete JSON snapshot
```

The provider token never enters the URL, MCP result, or downloaded content.
The outer delegated-client grant is checked when KDCube mints the URL. The URL
then acts as a short-lived bearer capability for that exact ref and bound
identity. On every GET, KDCube resolves the connected provider account and
claim again, so connected-account revocation or claim removal takes effect
immediately. Revoking only the outer delegated grant prevents minting another
URL but does not invalidate an already-issued URL before its expiry. A repeated
fetch can observe newer provider state. The external client may save the
response locally when it needs an immutable working copy.

A KDCube-hosted file uses a different implementation: the signed route streams
already hosted bytes. The schema and response should say whether delivery is a
live provider read or a hosted artifact.

Do not put ReAct-only instructions such as `react.read` or `react.rg` into an
external MCP response. Return provider-neutral metadata: MIME type, size when
known, schema, tabs/ranges or parts, completeness, cursor, and download URL.

### Harness Materialization

For a resident agent using the KDCube agentic harness:

```text
provider ref
  -> react.pull
  -> object.get(response_mode=stream)
  -> complete bytes written into the current turn workspace
  -> conv:fi: materialized ref
  -> react.read or code works on the stable turn snapshot
```

The materialized `conv:fi:` path identifies the local copy. Its metadata keeps
the canonical provider `object_ref`, which lets owner `block.produce` render a
compact inventory while code or exact reads can still inspect the complete
local content. The worked flow is in
[ReAct Object Materialization](../../sdk/namespace-services/react-object-materialization-README.md).

Wrapped LangGraph or other agents can use the same mechanism when the app
equips them with the harness pull/workspace adapter. See the
[ported LangGraph app](../../../src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/ported-langgraph-agents@2026-07-13/README.md).

## 7. Give Every Action A Stable Name

`object.action` is the operation family. Authorization and presentation use
the exact variant:

```text
object.action.post_message
object.action.download_file
object.action.append_rows
```

The generic MCP bridge dispatches the provider request as `object.action` with
`request.action=<name>`, while it authorizes the exact
`object.action.<name>` policy. This lets a user allow reading and one mutation
without granting every action in the namespace.

Each action schema states:

- accepted object kind/ref;
- payload fields and limits;
- required grants and connected-account claims;
- idempotency or `outcome_unknown` behavior;
- returned refs, UI events, or hosted files.

Keep the legacy parent `object.action` policy only for descriptors that have
not yet declared exact variants. Once exact variants exist, unknown actions
fail closed.

## 8. Model The Two Authorization Gates

For an internal realm, the provider receives the current KDCube auth context
and enforces its domain ownership/sharing rules.

For a provider-backed realm, calls may cross two independent gates:

```text
Gate 1 - Delegated by KDCube
  may this external client or hosted agent use this KDCube resource,
  operation, and exact action on behalf of the user?

Gate 2 - Delegated to KDCube
  may KDCube use this user's connected provider account for the required
  provider claim right now?
```

Declare gate 2 in `metadata.connected_accounts`:

```python
"connected_accounts": [
    {
        "provider_id": "google",
        "provider_label": "Google",
        "claims": ["records:read", "records:write"],
        "claim_labels": {
            "records:read": "read records",
            "records:write": "edit records",
        },
        "claims_by_operation": {
            "object.search": ["records:read"],
            "object.get": ["records:read"],
            "object.action.publish": ["records:read", "records:write"],
        },
    }
]
```

Connection Hub resolves the credential on the trusted server side. The named
service never returns provider access or refresh tokens. A missing gate emits
the structured consent demand for that gate; the agent can explain exactly
what approval is needed.

## 9. Configure Provider And Consumer Surfaces

Publishing a provider and allowing an agent to use it are separate choices.

Provider app:

```text
_named_service_providers()
  -> service discovery record for tenant/project
```

Consumer app (bundle descriptor):

```yaml
surfaces:
  as_consumer:
    agents:
      main:
        tools:
          - id: records_service
            kind: named_service
            alias: named_services
            namespaces:
              records:
                allowed:
                  - provider.about
                  - provider.capabilities
                  - object.schema
                  - object.search
                  - object.get
                  - object.action
```

Add an event source only when provider refs should be pullable/renderable by a
harness agent:

```yaml
        event_sources:
          - kind: named_service
            namespace: records
            enabled: true
            discovery: {mode: service_discovery}
            policies:
              pull: {mode: provider, operation: object.get}
              block_production: {mode: provider, operation: block.produce}
```

An external client connects to a managed MCP surface such as
`kdcube-services@1-0/public/mcp/named_services`. Its Connection Hub resource
declares namespace tools and exact operation/action grants. A hosted agent and
an external MCP client then see the same provider ontology through different
client adapters.

## 10. Add UI And Event Semantics When The Domain Needs Them

Generic UI surfaces pass the full `object_ref` to the provider:

```text
object.resolve
  -> label, object kind, actions, default_open_effect_action

object.action(open)
  -> ui_event.target_surface + bounded payload

scene
  -> routes target_surface to a configured widget/component
```

The scene does not infer what a ref means. The provider owns actions and the
consumer scene owns where the returned command is routed.

Provider events should carry canonical identity:

```json
{
  "type": "records.item.changed",
  "metadata": {"object_ref": "records:item:42"}
}
```

Use `event.resolve` and `block.produce` when those events or pulled refs should
enter the harness timeline with provider-owned rendering.

## 11. Verification Checklist

### Provider contract

- provider publication and withdrawal work through service discovery;
- all provider operations are async and do not block the proc loop;
- `about`, `capabilities`, and `schema` are useful without source knowledge;
- every returned ref round-trips into the next applicable operation;
- search/list pagination returns and accepts `next_cursor`;
- multiple accounts produce an explicit account choice;
- exact actions are schema-declared and independently authorized;
- projection indexes cover every declared object kind/action; recursive
  catalog paths, capability queries, kind, exact-operation, ref-inferred,
  full, and invalid-selector cases are tested;
- capability search results identify `catalog_path`, `object_kind`, and
  `schema_operation`; lexical fallback reports its effective mode;
- normal reads stay bounded, while complete authorized data remains
  retrievable by continuation, signed delivery, or stream materialization.

### Authorization

- the external/hosted agent grant is denied independently at Gate 1;
- the connected-account claim is denied independently at Gate 2;
- revoking either gate stops the next protected call;
- no provider credential appears in tool output, URLs, logs, timeline, or
  model input.

### Client behavior

- an external MCP client can discover, search, fetch complete content, and run
  one bounded action without ReAct-specific guidance;
- a resident harness agent can pull the same ref into `conv:fi:` and inspect
  the stable snapshot;
- UI resolve/open/download follows provider-owned actions;
- large payloads stream asynchronously and are not forced through model JSON.

### Package maintenance

- `entrypoint.py`, provider code, descriptor defaults/templates, interfaces,
  app docs, storage docs, tests, journal, and release metadata describe the
  same surface;
- the app's own test suite and the shared bundle contract suite pass;
- the real deployed transport is tested after refresh/reload.

## Worked Providers

- [Mail named service](../connections/integrations/mail-named-service-README.md):
  provider-neutral mail refs, complete message snapshots, attachments, send,
  and forward through connected accounts.
- [Slack integration](../connections/integrations/slack-README.md): channels,
  messages, files, pagination, signed file delivery, and exact actions.
- [Google Services (Gmail, Sheets)](../connections/integrations/google-service-README.md):
  the Sheets service adds typed productivity tools plus the `sheets` namespace,
  live complete snapshots, and harness materialization.
- [Named services over MCP](../apps/named-services-mcp-README.md): external and
  hosted client usage of the generic MCP surface.

Field-level SDK contracts remain in
[Namespace Service Providers](../../sdk/namespace-services/providers-README.md)
and [Namespace Service Clients](../../sdk/namespace-services/clients-README.md).
