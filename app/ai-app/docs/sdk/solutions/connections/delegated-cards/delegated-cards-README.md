---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/delegated-cards/delegated-cards-README.md
title: "Delegated Access Cards: Storage, Rendering, And Enforcement"
summary: "Canonical lifecycle of Connection Hub Delegated by KDCube cards: what each card stores, which live catalogs render its editor, how changes reach runtime enforcement, and how descriptor drift must be reconciled."
status: active
tags: ["sdk", "solutions", "connections", "connection-hub", "delegated-access", "cards", "grants", "mcp", "named-services"]
keywords: ["Delegated by KDCube", "AutomationAccessRecord", "resource_grants", "named_service_operations", "account_scope", "registry_access_id", "card authority", "descriptor drift", "grant lifecycle"]
updated_at: 2026-08-11
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/connection-hub-solution-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/delegated-connections/delegated-connections-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/connection-hub-token-storage-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/configuring-agent-service-access/configuring-agent-service-access-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/delegated-credentials/oauth-delegated-credential-protocol-adapter-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/delegated-accounts/delegated-accounts-README.md
---
# Delegated Access Cards

A card under Connection Hub **Delegated by KDCube** is the user-visible form of
one server-side delegated-access record. It answers:

- which user granted access;
- which agent, connected app, or manual automation received it;
- which KDCube resources, grants, and operations that caller may use;
- which connected provider accounts and claims that caller may use;
- when the grant was created, when it expires, and how it is revoked.

The card is the user's live delegated-authority record, not a cached
illustration of a token. Pointer-backed credentials identify the card, and the
managed guard resolves its current contents on each call. Editing or revoking
the card therefore changes the next call made with an already-issued bearer.
The deployment descriptor is intended to remain the ceiling around that user
decision. The descriptor-drift section below records where the current
implementation does not yet preserve that invariant.

The card is also not a copy of the entire current service catalog. It stores
the user's selected authority. Connection Hub separately reads the live
descriptor and provider catalogs to render choices around that stored
selection.

This page covers every card in **Delegated by KDCube**: manual automations,
hosted agents, and external OAuth/MCP clients. Cards in **Delegated to
KDCube** represent connected provider accounts and use a different storage
lifecycle; see [Delegated Accounts](../delegated-accounts/delegated-accounts-README.md)
and [Connection Hub Token Storage](../connection-hub-token-storage-README.md).
Connection-edge and authenticator administration cards likewise project their
own stores; they are not delegated-access records.

```text
stored card selection                 current deployment catalogs
  what this user granted                what can be granted now
  resource_grants                       resources and grants
  named_service_operations              namespaces and operations
  account_scope                         connected accounts and claims
             \                           /
              +---- Connection Hub -----+
                         |
                         +-- read-only card: stored decision, live labels
                         +-- create/edit form: live choices + stored selection
                         +-- runtime guard: current card authority
```

## Card Families

All three families use `AutomationAccessRecord`, the same per-user index, and
the same **Delegated by KDCube** list. Their issuance and credential-retention
rules differ.

| `source` | Represents | How it is created | Credential material retained in the card record |
| --- | --- | --- | --- |
| `manual` | A script, service, or external automation whose operator copies a bearer. | `delegated_access_create`. | The raw bearer is returned once and is not retained. The record keeps `session_id` and `last_four` for revocation and identification. |
| `agent` | A hosted agent with deterministic identity `kdcube-agent:<app>:<agent>`. | Demand-driven consent or `delegated_agent_grant_create`, backed by `create_access(client_id=...)`. | The reusable access token is retained server-side so each turn can resolve the same consented bearer. It is never returned by list. |
| `oauth` | An external OAuth/MCP client. | Automatically on initial token issuance and every refresh rotation. | Current access- and refresh-token handles are retained server-side so revoke can invalidate both. They are never returned by list. |

An OAuth client and a hosted agent are both delegated callers. The source
field records how their credential lifecycle is managed; it does not create a
different authorization model.

## Stored Record

The current implementation stores each record in Redis:

```text
{tenant}:{project}:kdcube:delegated-access:automation:{access_id}
{tenant}:{project}:kdcube:delegated-access:automation-by-grantor:{subject_hash}
```

The first key holds the record with a TTL. The second is the grantor's set of
card ids. Expired or missing ids are pruned from the set during list.

### Persisted fields

| Field | Meaning | Public list response |
| --- | --- | --- |
| `access_id` | Stable id of this card. | yes |
| `label`, `client_id` | Human name and delegated caller identity. | yes |
| `grantor_subject`, `delegate_subject` | User who granted and integration principal that acts. | yes; list is also owner-scoped to the authenticated grantor. |
| `operations` | Allowed outer API/MCP operation names derived at issuance or save. | yes |
| `resource_grants` | Exact selected KDCube claims per resource. | yes |
| `named_service_operations` | Exact selected namespace operations per resource. This is the selection the UI renders. | yes when non-empty; the current public serializer drops explicit `{}`. |
| `named_services` | Materialized boundary tree derived from the descriptor and `named_service_operations`. The proc-side bridge consumes it. | no |
| `account_scope` | Provider -> account -> exact connected-account claims this caller may use. | yes when non-empty |
| `identity_scope` | Which identity boundary the delegated resource uses. | yes |
| `created_at`, `expires_at`, `last_issued_at` | Lifecycle timestamps. | yes when present |
| `last_four`, `source` | Token fingerprint and card family. | yes |
| `session_id`, `access_token`, `refresh_token` | Internal credential/revocation handles, according to source. | no |

`to_public_dict()` removes all token material, `session_id`, and the internal
`named_services` boundary. A public card exposes the selection, never the
provider credential or the materialized enforcement tree.

The current record shape has one important ambiguity. Deserialization maps a
missing selection and an explicit empty selection to the same `{}`, while the
public serializer also drops `{}`. The code can therefore distinguish
"unrestricted" from "disabled" during the update that receives the input, but
not reliably on a later update that omits the field. This is a current defect,
not part of the intended contract described below.

Consequently, after a list/refetch, absence of a public Services selection can
currently mean either "use the full descriptor policy" or "no named-service
operation is allowed." The read-only card and editor need the durable mode
described under edit semantics before they can label those states reliably.

Connected provider credentials are stored by the delegated-to-KDCube account
system, not in these cards. `account_scope` contains ids and claims only.

## What List And Rendering Read

### Configuration vocabulary

The two top-level collections under
`connections.delegated_credentials.oauth` answer different questions:

| Descriptor node | Question it answers | Card effect |
| --- | --- | --- |
| `capabilities[]` | **Which grant tokens exist, and may this signed-in user delegate each one?** Each row defines a `grant`, its display metadata, delegation roles/permissions, and optional connected-account requirements. | The current user's authority is evaluated against these definitions to produce `grant_options`. A capability is primarily vocabulary and a delegation rule; it is not itself an endpoint or callable operation. |
| `resources[]` | **Which protected doors exist, and what may be reached through each door?** Each row defines a resource URL/pattern, its grants, outer tools, optional named-service boundary, identity scope, and admin restriction. | The live resource catalog supplies the selectable doors and the claims, outer tools, namespaces, and inner operations beneath them. Selected claims persist in `resource_grants`. |

`resources[].grants` is the claim ceiling shown for that door. When it is not
written explicitly, the parser derives it from the grants required by the
resource's outer tools and nested named-service operations. Every grant token
used there should have a matching `capabilities[]` definition so Connection
Hub can decide whether the current user may delegate it and provide its label.
`capabilities[].tools` remains a compatibility/fallback source of outer tool
metadata when no resource-specific tool catalog matches; current Connection
Hub descriptors define protected operations under `resources[].tools`.

There are also two different operation layers:

| Layer | Canonical descriptor shape | Example | List/card representation |
| --- | --- | --- | --- |
| Outer surface operation | `resources[].tools.<name>` | `named_services_schema` | The list response calls these `resources[].operations`; the card stores allowed names in `operations`. They are API/MCP entry operations at the protected resource. The parser also accepts `operations`, `allowed_tools`, and `actions` as input aliases, but `tools` is the canonical descriptor spelling. |
| Inner named-service operation | `resources[].named_services.namespaces.<namespace>.tools.<tool>.operation`, or the nested `<tool>.operations.<operation>` map | namespace `linkedin`, operation `object.schema` or `object.action.publish_post` | The exact user selection is stored in `named_service_operations`; the derived bridge policy is stored internally in `named_services`. These are the ontologic operations inside the named-services door. |

The word `capabilities` can also occur as a named-service tool key, for
example `tools.capabilities.operation: provider.capabilities`. That is merely
an inner callable operation. It is unrelated to the top-level
`oauth.capabilities[]` grant vocabulary.

### Live-services source fusion

The card editor does not obtain one preassembled "live services" object from
one store. Connection Hub joins configured ceilings, current user authority,
live provider requirements, connected-account state, and any stored card
selection:

```text
CONFIGURED DELEGATION CEILING                         CURRENT USER FACTS
connections.delegated_credentials.oauth
|
+-- capabilities[]                                   signed-in roles/permissions
|     grant + label                                             |
|     delegable_roles/permissions                               v
|     optional connected_accounts -----------> PlatformAuthorityInventoryProvider
|                                                          |
|                                                          +--> grant_options[]
|                                                               grants this user
|                                                               may delegate now
|
+-- resources[] ------------------------------------------> resources[] (doors)
      resource URL/pattern
      label / identity_scope / admin_only
      grants[] -------------------------------------------> claim rows
      tools{} --------------------------------------------> outer operation rows
        named_services_schema                                 |
          grants: [named_services:use]                        +--> card.operations
      named_services
        namespaces
          linkedin
            tools
              schema.operation: object.schema --------+
              call.operations:                        |
                object.action.publish_post -----------+--> NamedServiceBoundaryCatalog
                                                            |
                                                            +--> namespace/operation rows
                                                                 |
                                                                 +--> selected exact subset
                                                                      card.named_service_operations
                                                                      card.named_services
                                                                      (derived internal tree)

LIVE NAMED-SERVICE PROVIDER DISCOVERY
provider spec.metadata.connected_accounts ----------------> requirement annotations
  provider_id / connector_app_id / claims                    for descriptor namespaces
  claims_by_operation                                        only; adds no operation
                                                             and grants no authority

DELEGATED-TO-KDCUBE CONFIG + ACCOUNT STORE
provider and connector-app labels -------------------------> account-picker vocabulary
signed-in user's connected account rows -------------------> accounts, status, held claims
                                                                 |
                                                                 +--> selected exact binding
                                                                      card.account_scope

EXISTING CARD items[] --------------------------------------> seeds edit selections
PENDING DEMAND / DEEP LINK ---------------------------------> focuses requested rows only
                                                             (never grants by itself)
```

The namespace and operation tree is descriptor-owned. Specifically,
`NamedServiceBoundaryCatalog` projects only
`resources[].named_services.namespaces`; provider discovery does not invent a
namespace, add an operation, or make an operation callable. It contributes
the live provider's `metadata.connected_accounts` requirements, including
operation-specific provider claims such as `claims_by_operation`, so the UI
can explain which connected account is needed for a selected operation.

Those requirements are not impossible to express in configuration. They are
currently provider-owned runtime metadata because they describe what the
registered provider implementation needs, and sourcing them there avoids
copying the same provider contract into every delegated-resource descriptor.
This is distinct from `capabilities[].connected_accounts`, which is a
descriptor-owned mapping from a door grant such as `mail:read` to the provider
claim that satisfies that grant. If provider discovery is unavailable, the
descriptor namespace/operation rows still render, but the account-requirement
guidance is absent; no authority is broadened.

The delegated-to-KDCube catalog then contributes provider and connector-app
labels plus the signed-in user's current account rows, held claims, and
connection status. It powers account selection and connect/reconnect/approve
guidance. Provider credentials never enter the card or its list response.

### Projection into a saved card

At create or edit time, the server combines the selections with the current
catalogs in this order:

```text
selected door + claims
  -> resource exists in current resources[]
  -> every claim belongs to that door
  -> every claim is currently delegable by this user via capabilities[]

selected namespace operations
  -> namespace and operation exist in that door's NamedServiceBoundaryCatalog
  -> operation-required grants are present in the selected door claims

selected accounts
  -> normalize explicit provider -> account -> provider-claim binding
  -> live broker later verifies the account and held claim on provider use

successful projection
  -> resource_grants             exact selected door claims
  -> operations                  eligible outer surface tool names
  -> named_service_operations    exact selected inner operations
  -> named_services              narrowed descriptor tree for the bridge
  -> account_scope               exact connected-account binding
```

Resource, grant, and named-operation membership are validated during this
projection. `account_scope` is shape-normalized here; current account
existence and provider-claim possession are enforced by the connected-account
broker when the provider operation is attempted.

The manual create form sends an explicit namespace map for every selected
resource, including `{}` when no named-service operation was checked. Its UI
therefore starts named-service access closed. At the service API level,
omitting `named_service_operations` still has legacy/full-policy semantics;
that distinction, and the current failure to persist explicit-empty durably,
are described under edit semantics below.

`delegated_access_list` is a composite response. Its three top-level parts
come from different sources:

| Response field | Source | Purpose |
| --- | --- | --- |
| `items` | Persisted `AutomationAccessRecord` rows for the authenticated grantor. | What the user already granted. |
| `grant_options` | Live `PlatformAuthorityInventoryProvider` over configured capabilities and the current user's delegable authority. | Claims the user may grant now, with labels and descriptions. |
| `resources` | Live `connections.delegated_credentials.oauth.resources` descriptor parsed by `OAuthDelegatedClientConfig`. | Resources, top-level tools, grants, admin restrictions, and named-service catalogs available now. |

For each resource with a `named_services` block, Connection Hub projects the
current namespace and operation tree through `NamedServiceBoundaryCatalog`.
It enriches namespace rows with connected-account prerequisites from live
named-service provider discovery. The separate delegated-to-KDCube catalog
supplies the user's currently connected accounts and their claims to the
account picker.

This produces two deliberately different views:

### Read-only card

- Doors and access claims come from stored `resource_grants`.
- The Services row comes from stored `named_service_operations`.
- Account bindings come from stored `account_scope`.
- Current catalogs provide friendly labels when a match still exists.
- A missing connected account is displayed as a stale binding; no provider
  token is fetched merely to render the card.

### Create or edit form

- Available resources, claims, namespaces, and operations come from the live
  descriptor-backed catalogs.
- Check states begin from the stored card selection during edit.
- Claim editing currently unions the card's stored claims with the current
  resource claim catalog, so a removed stored claim remains visible and can be
  unchecked.
- Named-service operation rows currently come only from the live catalog. A
  stored operation removed from that catalog has no row in the picker.
- Account rows come from current connected accounts. A disconnected account is
  visible in read-only mode but is not seeded into the edit picker.

The direct answer to "where does the Services list come from?" is therefore:

```text
read-only Services row    -> card.named_service_operations (stored selection)
create/edit Services rows -> resources[].named_services (live descriptor catalog)
provider prerequisites    -> live named-service discovery metadata
account choices            -> live delegated-to-KDCube account catalog
```

## Creation Lifecycle

### Manual automation

```text
authenticated user opens create form
  -> list returns live resources and grant choices
  -> user selects resource grants, namespace operations, and account scope
  -> create validates every selection against current config and authority
  -> selected named-service operations narrow the descriptor policy
  -> bearer and access-grant binding are minted with registry_access_id
  -> card record is stored and indexed
  -> raw bearer is returned once
```

### Hosted agent

```text
agent attempts a governed operation
  -> denial names deterministic kdcube-agent:<app>:<agent> client
  -> user approves the demand in Connection Hub
  -> create_access deduplicates by grantor + client + resources
  -> ordinary consent merges newly approved authority
  -> explicit edit uses replace semantics
  -> reusable agent bearer and card are stored server-side
```

### OAuth/MCP client

```text
external client completes OAuth consent
  -> token endpoint issues access/refresh material
  -> record_oauth_grant upserts one card per grantor + client + resource
  -> refresh rotation updates current token handles and last_issued_at
  -> user edits or revokes the same visible card later
```

A reconnecting dynamic client may receive a new client id. Connection Hub can
supersede a matching older card, carry its account binding forward, and revoke
the old token handles.

## Edit Semantics

The card's selected authority is changed in place. No new manual token is
issued, and a pointer-backed caller observes the change on its next request.

### Manual card

`delegated_access_update` is intended to validate and rewrite the record with
this contract:

| Input | Meaning |
| --- | --- |
| `resource_grants` | Required full replacement. No remaining grant means revoke, not update. |
| `named_service_operations` omitted | Preserve the stored narrowing. |
| `named_service_operations: {}` | Disable all named-service operations for the selected resources. |
| `named_service_operations` with content | Replace with that exact resource -> namespace -> operation selection. |
| `account_scope` omitted | Preserve the current account binding. |
| `account_scope: {}` | Bind no provider account; provider-backed use is default-closed. |
| `account_scope` with content | Replace with the exact provider -> account -> claim selection. |
| `label` | Rename without changing omitted dimensions. |

The update recomputes outer operations and the materialized `named_services`
tree from the descriptor available to Connection Hub.

At the current PR head, non-empty preservation and an immediate explicit clear
work, but the clear state is not represented durably. After
`named_service_operations: {}` stores an empty selection, a later update that
omits the field can interpret that empty value as "no narrowing" and rebuild
the full descriptor policy.

The durable model adds `named_service_selection_mode` with three explicit
values:

| Mode | Meaning |
| --- | --- |
| `full` | Use the complete currently configured named-service boundary permitted by the selected grants. |
| `none` | Permit no named-service operation. |
| `selected` | Permit exactly the entries in `named_service_operations`. |

The list response must retain the mode so the editor can render "all allowed
by grants" and "no named-service operation" differently. Existing rows cannot
be migrated from an empty `named_service_operations` map alone because older
full-policy rows serialized the same value. Lazy migration must inspect raw
field presence and the persisted materialized `named_services` boundary. If
those facts conflict, GET preserves the effective narrowed selection and asks
for an explicit choice on Save; it does not guess or mutate the row.

### Agent card

Demand-driven approval merges by default so separate approved demands
accumulate. An edit calls the same creation service with
`merge_existing=False`; submitted resource claims replace that resource's
selection, while omitted `account_scope` and `named_service_operations`
preserve their stored values. Explicit empty maps clear those dimensions.

The current ordinary agent-card editor exposes resource claims and account
scope. Pending consent can submit exact named-service operations. The backend
supports the same absent/empty/content contract when that field is supplied.
It shares the same current empty-versus-absent persistence ambiguity.

### OAuth card

`extend_client_access` merges a one-click extension or replaces one resource's
claims during edit. It also merges or replaces `account_scope`. The current
OAuth card edit path does not rewrite `named_service_operations`; its
named-service boundary remains the one established by OAuth consent.

## Runtime Enforcement Lifecycle

Pointer-backed credentials carry `registry_access_id` in the access-grant
binding. The managed guard treats that id as a pointer, not as authority by
itself:

```text
request bearer
  -> authenticate bundle/OAuth token
  -> load access-grant binding
  -> read registry_access_id
  -> resolve card from delegated-access store
  -> verify expected client, grantor, delegate, expiry, and record shape
  -> copy current card facts into the request-local grant
       resource_grants
       flattened grants/scopes
       operations
       account_scope
       named_services materialized boundary, when present
  -> enforce outer resource/tool gate
  -> enforce named-service namespace/operation boundary
  -> resolve a permitted connected account and its claims, when required
  -> call provider
```

Missing, expired, malformed, unavailable, or identity-mismatched pointer
authority fails closed. Revocation deletes the card and invalidates the
source-specific session/token records, so an in-flight bearer cannot recover
authority from its older embedded snapshot.

Legacy bindings without `registry_access_id` retain embedded-snapshot
semantics. A card written before the internal `named_services` field keeps the
binding's original named-service snapshot until its first compatible save.

## Descriptor And Catalog Drift

Stored consent and current deployment policy are different facts:

```text
stored selection       = what the user approved
current catalog        = what the deployment offers now
effective authority    = never more than both permit
```

The current implementation does not yet satisfy that last equation for every
dimension after a descriptor change.

### Current behavior

| Descriptor change | Card/list behavior | Runtime behavior |
| --- | --- | --- |
| A claim or operation is added | It appears in the live create/edit catalog. Existing cards do not select it automatically. | Existing explicit selections do not gain it. |
| A stored top-level claim is removed | Claim edit unions stored and current values, so the stale claim remains visible and removable. | The live guard currently copies stored `resource_grants` without clamping them to the current resource ceiling. |
| A stored named-service operation is removed | Read-only view still shows the stored operation. The edit picker omits it because rows come from the live catalog. Strict save then rejects the invisible stale value as unknown. | The card's materialized `named_services` snapshot still includes the operation, so the inner bridge can continue to admit it. |
| A whole stored resource is removed | The card still names it, while current resource metadata is absent. A normal save fails current-resource validation. | Pointer resolution still projects the stored resource grant. |

The named-service case makes the card a dead end: the operator cannot untick an
operation that the picker no longer renders, and unrelated edits fail until
the descriptor is restored or the card is revoked.

### Required reconciliation contract

The fix has two independent parts and both are required:

1. **Enforcement clamps immediately.** A descriptor removal must reduce
   effective authority before the user opens or saves the card. The enforcing
   path needs an authoritative current ceiling without coupling a generic
   surface guard to Connection Hub's storage or transport details. If that
   authority cannot be resolved, the governed call fails closed.
2. **The editor explains and repairs stored drift.** List/edit should return a
   server-computed drift projection. The UI shows a visible warning such as
   **Service access changed since this grant was last saved**, with details:
   removed selections are no longer effective and will be removed on save;
   newly available choices remain unchecked until explicitly granted.

Save should reconcile in this order:

```text
stored selection
  -> retain stale entries long enough to explain them
  -> prune entries absent from the current catalog
  -> validate every remaining selection strictly
  -> persist the reconciled explicit selection
  -> rebuild the materialized boundary
  -> stamp the active catalog_version_id
  -> clear the warning
```

GET must not silently rewrite the user's record. Enforcement may narrow
immediately, while the stale stored value remains available as evidence until
the user saves or revokes.

### Catalog ownership and versioning

Connection Hub owns one central history of the delegable catalog. A card never
stores a catalog copy. It stores only `catalog_version_id`, referring to the
immutable snapshot active when that card was created or last saved.

The normalized snapshot fuses only authority-relevant deployment facts:

```text
capability grant vocabulary and delegation rules
+ resource doors, claim ceilings, and outer operations
+ descriptor named-service namespaces and operations
+ provider-owned account requirements for those declared operations
```

Current user accounts, roles, and held provider claims remain live inputs and
are not copied into the shared snapshot. Display labels and ordering likewise
do not create an authority version.

Publication uses a shared critical section and an inside-section recheck:

```text
normalize current inputs
-> compute input fingerprint and authority hash
-> mark the demanded input pending when it differs from the active input
-> write one immutable snapshot and one versioned runtime projection
-> commit the active version only if no newer demanded input appeared
-> publish ready
```

The serving state records the expected input fingerprint, active input
fingerprint, active version, and `pending | ready | failed` status. A governed
call accepts the active projection only while status is `ready`, both
fingerprints match, and the versioned projection exists. If configuration has
changed but publication has not completed, the old snapshot remains available
for history while authorization returns structured `503 temporarily_unavailable`.
It never treats stale deployment authority as a fallback.

Generic app deployment after a source or effective-props change is the primary
publication trigger. Connection Hub's `on_bundle_load` performs idempotent
first-run initialization and repair. A future Access Map mutation persists the
authoritative props and invokes the same publisher before reporting success.
These are generic lifecycle contracts; the platform does not hardcode
Connection Hub into bundle loading or request routing.

To say that an operation is **new since the last grant**, list compares the
card's referenced immutable snapshot with the active snapshot. Comparing only
the current catalog with selected operations can find removed selected values,
but cannot distinguish a newly added operation from one that was already
available and deliberately left unchecked.

### Drift projection and Save concurrency

`delegated_access_list` computes drift on the server and returns one status per
card:

| Status | Meaning |
| --- | --- |
| `current` | The card references the active catalog version. |
| `changed` | A relevant resource, claim, outer operation, named operation, or account requirement changed. |
| `no_relevant_change` | The global catalog advanced without changing anything represented by this card. |
| `baseline_missing` | The card is legacy or its referenced snapshot is unavailable, so exact additions cannot be claimed. |
| `unavailable` | Current catalog authority cannot be resolved; editing authority is disabled and governed calls fail closed. |

The drift object contains ready-to-render `removed`, `added`, and
`account_requirement_changes` entries. Removed selected entries stay visible
as disabled/stale rows and are already ineffective. Added entries are current
options but remain unchecked.

An edit submits `expected_record_revision` and
`expected_catalog_version_id`. Save returns `409` with a refreshed projection
if either the card or catalog changed after the editor loaded. Otherwise the
server prunes stale selections, validates every survivor against the active
catalog, rebuilds `operations` and `named_services`, increments
`record_revision`, and stamps the active `catalog_version_id` atomically.

Runtime applies the same ceiling before Save:

```text
effective resource claims       = stored claims intersect active claims
effective outer operations      = stored operations intersect active operations
effective named operations      = stored selections intersect active namespace operations
effective account-backed access = stored account_scope checked against current requirements
```

The clamp applies to pointer-backed cards and legacy managed bindings. The
generic guard consumes a catalog-resolver interface and does not know which
storage technology or bundle produced the current projection.

There is no delegated-card selector meaning "all current and future operations
in this namespace." New operations therefore remain ungranted. Platform
administrator authority is a separate concept and must not be inferred as a
future-operation wildcard on a granular card.

An append-only card-change history would improve audit and rollback, but it is
separate from the immediate enforcement and editability requirements above.

## Revocation And Expiry

- Expired records disappear from list and cease to resolve.
- Manual revoke logs out the bound platform session and removes the access
  grant/card.
- Agent revoke removes its reusable server-side grant and bearer authority.
- OAuth revoke removes current access-grant and refresh-token state as well as
  the card.
- Every mutation publishes a best-effort
  `connection_hub.delegated_access.changed` event so open Connection Hub
  widgets refetch the authoritative list.

## Implementation Map

| Concern | Implementation |
| --- | --- |
| Capability/resource vocabulary and parser aliases | `kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_credentials.oauth.config` |
| Record, Redis keys, create/list/edit/revoke | `kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_credentials.automation_access` |
| Named-service strict narrowing and materialization | `...delegated_credentials.named_service_policy` |
| Descriptor namespace/operation projection | `...solutions.named_services_providers.boundary_policy.NamedServiceBoundaryCatalog` |
| Live provider requirement enrichment | `automation_access.AutomationAccessService._named_service_options` plus provider discovery `spec.metadata.connected_accounts` |
| Pointer-backed live card resolution | `...delegated_credentials.live_grant` |
| Managed request projection | `...delegated_credentials.oauth.surface_guard._live_grant_record` |
| Connection Hub operations and descriptor binding | `connection-hub@1-0/entrypoint.py` |
| List/create/edit UI and account-requirement fusion | `connection-hub@1-0/ui/widgets/connections/src/features/delegatedAccess` |
| Connected-account credential resolution | delegated-to-KDCube broker and provider adapters |
