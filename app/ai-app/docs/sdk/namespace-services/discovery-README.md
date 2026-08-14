---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/discovery-README.md
title: "Namespace Services: Discovery Registry"
summary: "The per-tenant/project, Redis-backed discovery table where explicitly owned named-service providers publish an authoritative per-app registry snapshot so consumers across processes can find them."
status: current
tags: ["sdk", "namespace-services", "discovery", "registry", "redis", "named-service-provider", "providers", "namespaces", "cross-process"]
updated_at: 2026-07-18
keywords:
  [
    "named service discovery",
    "RedisNamedServiceDiscovery",
    "discovery registry",
    "discovery entry",
    "provider record",
    "redis key schema",
    "register_provider",
    "register_registry",
    "authoritative provider snapshot",
    "provider withdrawal",
    "discovery reconciliation",
    "service discovery read path",
    "namespace roster intros",
    "cross-process provider table",
  ]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/providers-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/clients-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/integration-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/cross-runtime-context-README.md
---
# Namespace Services: Discovery Registry

The discovery registry is a per-tenant/project, Redis-backed table where
named-service providers announce themselves. Consumers — agents, named-service
clients, canvas/chat resolvers, and the ReAct namespace roster — read this table
to find which provider owns a namespace, **across processes**.

The in-process `NamedServiceRegistry` (`registry.py`) is process-local: it holds
the live provider objects loaded in one runtime and is the fastest local
dispatch path. The discovery registry is the shared routing source a consumer
uses when the provider may live in another app package, another process, or
behind a transport. A provider app publishes its registry into discovery on
load; any consumer with the same tenant/project then resolves the provider from
Redis. The provider remains the semantic authority for the realm.

The implementation is `RedisNamedServiceDiscovery` in
`kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.discovery`. One
instance is scoped to one `(tenant, project)` pair.

```python
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers import (
    RedisNamedServiceDiscovery,
)

discovery = RedisNamedServiceDiscovery(redis, tenant=tenant, project=project)
```

## Ownership And Publication Invariant

Discovery records what an app **currently publishes**. It does not decide which
domain an app owns and it does not turn inherited code into a provider surface.

It also does not make a surface **askable**. Being discoverable and being
grantable are different decisions: no user can be asked to approve access to an
endpoint until the deployment lists it in the Connection Hub's delegable
catalog, which is a subset of what apps expose and is never populated
automatically — see
[Named Service Clients](clients-README.md#consent-on-the-named-service-path).

```text
provider implementation
  describes one realm and its operations

app registry
  explicitly chooses which provider instances this app publishes

discovery
  indexes that current registry for cross-process lookup
```

The `@named_service_provider(...)` decorator describes a provider class. An app
publishes an instance only by contributing it to its `NamedServiceRegistry`,
normally through `_named_service_providers()` on `BaseEntrypoint`. A reusable
base class or mixin may implement provider support, but support alone is not
ownership. Any provider switch on reusable infrastructure must default off; the
dedicated owner enables it explicitly.

For example, `user-memories@2026-06-26` explicitly enables and publishes the
`sdk.memory` provider for `mem`. Task Tracker publishes its task provider only.
Using memory helpers or inheriting memory-capable infrastructure in another app
must not make that app a second memory provider.

Each publication is authoritative only for one app identity (`bundle_id`):

- providers present in the app's current registry are upserted;
- previous records for that same app that are absent now are withdrawn;
- publishing an empty registry withdraws every previous provider for that app.

This does **not** make discovery a global one-namespace/one-provider map.
Different apps may intentionally publish providers for the same base namespace
when operations, refs, or object kinds are partitioned. That is an explicit
architecture choice; accidental duplicate ownership through inheritance is
not. Request resolution ranks the eligible entries.

## The Discovery Record Per Provider

Each provider stored in discovery is a `NamedServiceDiscoveryEntry`: a provider
`spec` plus a transport/retention envelope. The spec is a
`NamedServiceProviderSpec`; its fields (from `to_dict()` / `from_dict()`) are:

| Field | Meaning |
| --- | --- |
| `provider_id` | Provider registry identifier, for example `task.issue`. |
| `bundle_id` | Owning bundle id, for example `task-tracker@1-0`. |
| `namespace` | Primary base namespace (first of `namespaces`). |
| `namespaces` | Sorted base namespaces the provider owns/operates on. |
| `refs` | Owned ref patterns, for example `task:issue:*`. |
| `object_kinds` | Concrete object kinds the provider exposes. |
| `search_scopes` | Bounded searchable scopes (`namespace`, `label`, `object_kind`, `description`, `filters_schema`). |
| `operations` | Operation → `{transports}` capability map. |
| `label` | Short human label. |
| `description` | Human description. |
| `intro` | Roster intro string for the namespace (see the read path below). |
| `metadata` | Provider metadata; `metadata.priority` participates in ranking. Also carries the realm's human layer for catalog consumers: `presentation` (purpose, works-with, entry labels/descriptions), `object_kinds` one-liners, an `actions` name→description map, and — for provider-backed realms — `connected_accounts` requirements (with `claims_by_operation`/`claim_labels` where real). Declared in the spec, published verbatim (see [Providers](providers-README.md)). |

Compact reference for one entry:

```text
provider_id · bundle_id · namespace(s) · label · description ·
search_scopes:[…] · operations:[…] · refs:[…] · object_kinds:[…] · intro:"…"
```

The `NamedServiceDiscoveryEntry` envelope wraps the spec:

| Entry field | Meaning |
| --- | --- |
| `spec` | The `NamedServiceProviderSpec` above. |
| `endpoint` | Transport/routing payload: `transport`, `bundle_id`, `provider`, `registry_method`. |
| `registered_at` | `time.time()` at registration. |
| `expires_at` | Absolute expiry (`registered_at + ttl`), or `0.0` when persistent. |
| `schema` | `NAMED_SERVICE_DISCOVERY_SCHEMA` = `"kdcube.named_service.discovery.v1"`. |

The `endpoint` written by `register_provider` defaults to
`transport="bundle_registry"` and `registry_method="named_services"`, so a
consumer that resolves the entry knows to call the owner bundle's
`named_services()` registry method (callers may pass extra endpoint fields).

## Redis Key Schema

Every key is namespaced by tenant and project. The base prefix is:

```text
kdcube:named_services:{tenant}:{project}
```

(`tenant`/`project` are sanitized by `_key_part`, which keeps alphanumerics and
`- _ . @ :`.) From that base:

| Key | Type | Contents |
| --- | --- | --- |
| `kdcube:named_services:{tenant}:{project}:provider:{bundle_id}::{provider_id}` | string | JSON of one `NamedServiceDiscoveryEntry`. |
| `kdcube:named_services:{tenant}:{project}:providers` | set | All provider keys for this tenant/project. |
| `kdcube:named_services:{tenant}:{project}:namespace:{ns}` | set | Provider keys that own namespace `{ns}`. |

The per-provider uid is `{bundle_id}::{provider_id}` (`_provider_uid`), so two
bundles registering the same `provider_id` get distinct keys. The `:providers`
set is the all-providers index; one `:namespace:{ns}` set is written per base
namespace in `spec.namespaces`.

### TTL And Expiry

TTL is `ttl_seconds`, defaulting to `DEFAULT_DISCOVERY_TTL_SECONDS = 0`
(persistent). When `ttl > 0`:

- the per-provider key is written with `ex=ttl` (`expires_at = now + ttl`);
- the `:providers` and `:namespace:{ns}` index sets are given `expire(ttl * 2)`,
  so the index outlives the records it points at.

When `ttl == 0`, records and index sets are persistent: `register_provider`
calls `redis.persist(...)` on the index sets (when the client supports it) so a
previously-expiring set does not silently drop. Expired or missing per-provider
records are tolerated on read: `providers()` skips index keys whose record no
longer resolves and logs a `missing_records` count.

## Registration (Write Path)

A provider app publishes its current registry into discovery when the app is
loaded — typically through `BaseEntrypoint.on_bundle_load`. Two methods write,
with deliberately different lifecycle semantics:

- `register_registry(registry, *, bundle_id, transport="bundle_registry",
  registry_method="named_services", ttl_seconds=None)` — reconciles the app's
  complete current provider set. It upserts current providers and withdraws
  records previously published by the same `bundle_id` that are now omitted.
- `register_provider(spec, *, bundle_id, transport="bundle_registry",
  registry_method="named_services", endpoint=None, ttl_seconds=None)` —
  upserts one spec without reconciling the rest of the app's registry. Requires
  a resolvable `bundle_id` (from the argument or `spec.bundle_id`), else it
  raises `ValueError`.

`register_provider` stores the **full** spec — including `intro`, `search_scopes`,
`operations`, `refs`, and `object_kinds` — as the JSON entry under the
per-provider key, then adds that key to the `:providers` set and to each
`:namespace:{ns}` set. It returns the stored `NamedServiceDiscoveryEntry` and
logs a structured registration line (scope, provider, bundle, namespaces,
endpoint, retention, refs, object_kinds, search_scopes, operations).

When an existing provider changes namespaces, registration removes its key from
the old namespace indexes before adding the new memberships. When
`register_registry` withdraws a provider, it removes the provider key from the
global index, every namespace index recorded in the old entry, and Redis
itself. Persistent discovery (`ttl == 0`) therefore remains correct when an app
stops publishing a provider; operators should not need routine manual Redis
cleanup.

```python
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers import (
    publish_registry_discovery,
)

await publish_registry_discovery(
    self.named_services(),  # may be empty: empty is an authoritative snapshot
    redis=self.redis,
    tenant=tenant,
    project=project,
    bundle_id=self.BUNDLE_ID,
    logger=self.logger,
)
```

`publish_registry_discovery(...)` is the normal app-level entry point. It must
still call `register_registry` for an empty registry so previous records can be
withdrawn. It no-ops only when Redis or tenant/project/app identity is
unavailable.

The discovery scope itself is portable runtime context. `bind_named_service_discovery(...)`
binds an instance to a `ContextVar` and publishes a JSON-safe
`{schema, backend, tenant, project}` descriptor through `comm_ctx`;
`get_current_named_service_discovery()` restores a `RedisNamedServiceDiscovery`
from that descriptor (or the current request actor) without passing a live Redis
client through tool registries. See
[Cross-Runtime Context](../../runtime/cross-runtime-context-README.md).

## Reading: Go Through The Discovery Module

**The discovery module is the single entry point for routing reads.** Consumers resolve
providers through its read methods; they do not SCAN raw Redis keys and do not
treat the process-local `NamedServiceRegistry` as the authoritative provider
table. The process-local registry only knows providers loaded in the current
runtime; the discovery table is the cross-process routing source. Reading
through the module keeps record decoding, missing-record tolerance, namespace
scoping, and ranking in one place.

`RedisNamedServiceDiscovery` exposes these reads:

| Method | Returns | Use |
| --- | --- | --- |
| `providers(*, namespace="")` | `list[NamedServiceDiscoveryEntry]` | Read all entries, or only those in one base namespace. |
| `entries_for_namespace(namespace)` | `list[NamedServiceDiscoveryEntry]` | Live provider entries that own one base namespace (normalized). |
| `list_entries()` | `list[NamedServiceDiscoveryEntry]` | Every entry in the tenant/project table. |
| `namespace_intros(namespaces=None)` | `dict[str, dict[str, str]]` | `{base_namespace: {"intro", "label"}}` for the ReAct namespace roster. |
| `resolve(request, *, namespace="", provider_id="")` | `NamedServiceDiscoveryEntry \| None` | Select the best provider for a `NamedServiceRequest` by namespace, `provider_id`, operation support, ref match, and object kind, ranked. |

`providers()` reads keys from the `:namespace:{ns}` set (when a namespace is
given) or the `:providers` set, loads each record, and drops unreadable/missing
ones. `resolve()` filters candidates by `provider_id`, `supports_operation`,
`matches_ref`, and `object_kinds`, then ranks by ref match score, object-kind
match, namespace match plus `metadata.priority`, and `provider_id`.

`namespace_intros(...)` is the canonical roster read. It pulls entries via
`entries_for_namespace` (or `list_entries` when no namespaces are passed) and
maps each provider's `spec.intro` (with `spec.label` fallback) to **every** base
namespace it owns through the shared `intros_from_entries(...)` helper — reading
only the existing `:providers` / `:namespace:{ns}` sets, no raw key scan. The
module-level convenience `fetch_namespace_intros(discovery, namespaces=None)`
delegates to whatever discovery object's `namespace_intros` reader is given
(and returns `{}` when `discovery is None`), so it works against both
`RedisNamedServiceDiscovery` and the in-memory `ConfiguredNamedServiceDiscovery`.
Reading the process-local registry instead of discovery here is the failure mode
the rule prevents: the roster would render only providers loaded in the current
process, missing cross-process providers.

`ConfiguredNamedServiceDiscovery` is the in-memory discovery view for explicit
provider config when Redis discovery is not available. It exposes the same
`resolve(...)` read contract, so consumers depend on the discovery surface, not
on the backend.

## Cross-Links

- [Providers](providers-README.md) — how a provider and its
  `NamedServiceProviderSpec` are defined.
- [Clients](clients-README.md) — how consumers call the resolved provider.
