---
id: repo:kdcube-ai-app/app/ai-app/docs/arch/delegated-authority-and-admission-README.md
title: "Delegated Authority And Admission"
summary: "End-to-end architecture for current delegated-card authority across managed REST/MCP surfaces, connected-account claims, native named-service tools, and direct or relayed provider invocation."
status: current
tags: ["arch", "security", "admission", "connection-hub", "delegated-access", "mcp", "rest", "named-services", "data-bus"]
keywords: ["delegated authority", "managed surface guard", "delegated access card", "access_id", "active catalog", "resource grants", "MCP tool grants", "connected account claims", "NamedServiceAdmission", "Data Bus relay"]
updated_at: 2026-08-27
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/arch/security-and-trust-model-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/cross-runtime-context-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/connection-hub-solution-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/delegated-cards/delegated-cards-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/authenticated-mcp/authenticated-mcp-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/mcp/platform-mcp-over-connection-hub-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/kdcube-services/named-services-from-isolated-runtime-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/integration-README.md
---
# Delegated Authority And Admission

This page is the whole-system map for delegated authority on managed KDCube
surfaces. It connects durable Connection Hub state, request and session
identity, managed REST/MCP guards, plain tools with connected-account claims,
and named-service calls that may execute directly or through the Data Bus.

The delegated card and catalog system is reusable beyond named services. A
managed surface author registers protected resources, outer operations/tools,
and grants. The same current card and active catalog then govern a conversation
MCP tool, a productivity MCP tool, a managed REST operation, or the outer door
of a named-service MCP surface. `NamedServiceAdmission` is the additional inner
contract used only when execution enters the common named-service dispatcher.

The focused documents remain the implementation references:

- [Delegated Access Cards](../sdk/solutions/connections/delegated-cards/delegated-cards-README.md)
  owns card/catalog storage, rendering, drift, mutation, and recovery.
- [Authenticated MCP](../sdk/solutions/connections/authenticated-mcp/authenticated-mcp-README.md)
  owns the full managed MCP configuration and connected-account consent chain.
- [Platform MCP Over Connection Hub](../sdk/solutions/mcp/platform-mcp-over-connection-hub-README.md)
  owns the reusable MCP-door pattern and its caller families.
- [Named Services From An Isolated Runtime](../sdk/solutions/kdcube-services/named-services-from-isolated-runtime-README.md)
  owns the Data Bus request/reply protocol, worker behavior, retries, and replay.
- [Named-Service Integration](../sdk/namespace-services/integration-README.md)
  owns provider discovery, request/response shapes, and provider implementation.
- [Cross-Runtime Context](../runtime/cross-runtime-context-README.md) owns the
  portable actor, routing, policy, and runtime-bootstrap context.

## The Distinct Facts

The architecture keeps these facts separate:

| Fact | Answers | Owner |
| --- | --- | --- |
| Actor identity | Who caused this work? | Ingress and carried request context / `AuthContext`. |
| Authority selector | Which delegated record applies? | Exact bearer card binding or trusted hosted-agent identity. |
| Delegated card | What did this user grant this caller? | Connection Hub delegated-card store. |
| Active catalog | What does this deployment currently expose? | Connection Hub delegated catalog published from effective descriptor connections. |
| Managed surface decision | May this resource and outer REST/MCP operation run now? | Managed surface guard for this request/tool invocation. |
| Connected-account decision | Which account and provider claims may this tool use? | Account broker using declared tool requirements and the card's `account_scope`. |
| Named-service admission | May this decoded namespace and inner operation run now? | Common named-service dispatcher, when that subsystem is used. |
| Provider request context | What diagnostics and domain context accompany a named-service request? | `NamedServiceRequest.context`, visible to provider code. |

The managed surface decision is request-local guard state. For a named-service
call, `NamedServiceAdmission` is separate platform-owned dispatch state and a
sibling of the provider request. `NamedServiceRequest.context` remains
provider-visible domain and diagnostic context.

## State That Feeds A Decision

Two durable histories produce the current delegated decision. Redis is their
serving projection:

```text
OPERATOR DEPLOYMENT                                USER AUTHORITY
effective descriptor connections                  create / edit / revoke card
              |                                                |
              v                                                v
immutable catalog version + active.json           immutable card revision + current.json
              |                                                |
              +--------- durable Connection Hub storage -------+
                                       |
                              validated read-through
                                       v
                              Redis serving projections
                              active catalog + live card
```

The durable documents contain non-secret authority and provenance. Credential
handles, provider tokens, refresh tokens, and reusable session secrets remain
in their bounded credential/session stores. A Redis miss reads the committed
durable current document, validates it, and restores the serving projection.

The same card/catalog state feeds three enforcement dimensions:

```text
managed resource/tool authority
    = current card resource grants + selected outer operations
      INTERSECT complete current active catalog

connected-account authority
    = current card account_scope
      INTERSECT current tool/provider account requirements

named-service inner authority, when used
    = current card named-service selection
      INTERSECT current active namespace/operation catalog
```

The card records the user's explicit selection. The active catalog is the
deployment ceiling. A plain account-backed tool declares its current provider
requirements and resolves an account whose approved claims satisfy them. A
named-service call adds its inner namespace/operation boundary.

## Complete Managed-Surface Flow

The managed guard is the reusable outer admission layer. The protected surface
decides what happens after that layer:

```text
external MCP/REST client or resident agent connection
                         |
                         v
                 delegated bearer proof
                         |
                         v
authenticate credential/session
bind actor + exact delegated_card_binding.access_id
                         |
                         v
Connection Hub managed surface guard
  resolve exact current card
  load complete current active catalog
  match protected resource
  check identity, expiry, outer operation/tool, and required grants
                         |
             denied with a structured reason, or
                         |
                         v
request-local delegated identity and current authority
                         |
                         v
                    surface handler
                         |
       +-----------------+-----------------+-----------------+
       |                 |                 |                 |
       v                 v                 v                 v
plain domain tool  plain account tool named-service door custom app door
Conversation MCP  Productivity MCP   generic MCP tools  REST or MCP
conversations:read tool claim policy named_services:use app-owned grants
       |            + account broker       |                 |
       |                 |                 v                 +-- app storage/domain logic
       |                 v         decoded namespace/op      |
       |          card account_scope       |                 `-- optional account claims
       |          INTERSECT current        v
       |          provider requirements NamedServiceAdmission
       |                 |          + inner card/catalog check
       |                 v                 |
       |          connected provider       v
       |          operation           provider discovery
       |                                   |
       +-------------------+---------------+-----------------+
                           |
                           v
                    surface response
```

Conversation MCP demonstrates a plain managed operation with no connected
provider account. Its `conversations_export` tool requires
`conversations:read`, and the managed guard applies the current card/catalog
decision before the conversation export facade runs.

Productivity MCP demonstrates a plain account-backed MCP surface with no
named-service registration. The managed guard applies its resource and
per-tool grants first. Each tool then applies its declared `ToolClaimPolicy`
through `enforce_tool_requirements`, which resolves the card's account binding
and current provider claims before invoking Slack, mail, Sheets, Docs, or
LinkedIn code.

The named-services MCP door first passes the same managed resource/tool guard.
Its generic tool then decodes a namespace and inner operation. The common
dispatcher requires `NamedServiceAdmission` and applies the card's current
named-service selection under the active catalog before provider discovery.

## Other Named-Service Entrances

A native hosted-agent named-service tool can enter the common dispatcher
without crossing a managed MCP door. It constructs delegated admission inside
each `_call`; the trusted source bundle, agent, client, grantor, and actor form
the selector. A direct call resolves current Connection Hub state locally. A
relayed call carries the typed selector and actor to the target worker, which
validates both and resolves current state there.

Application authority is selected positively at a trusted named-service call
site. The source bundle and caller policy establish that authority.

```text
managed named-services MCP request     native hosted-agent tool     trusted application
guarded request-local snapshot         delegated selector           application admission
                  \                         |                         /
                   +------------------------+------------------------+
                                            |
                                            v
                                 common named-service dispatcher
                                 decoded namespace + operation
                                            |
                                            v
                                 one NamedServiceAdmission decision
```

## Reusing This System For Another Service

An app author can use the same delegated-authority system for a plain managed
REST or MCP surface:

1. Declare the REST or MCP surface with descriptor-owned managed auth.
2. Publish the protected resource, outer operations/tools, and required KDCube
   grants in the Connection Hub delegated catalog.
3. Let the managed guard bind the authenticated actor and resolve the exact
   current card/catalog decision for every request or tool invocation.
4. For a plain tool backed by connected accounts, declare its provider claims
   and call the shared account-enforcement helper before domain work.
5. For a named-service tool, pass explicit delegated or application admission
   into the common dispatcher; it owns the inner namespace/operation check.

The current built-in surfaces are worked examples. A bundle author can publish
the same pattern from an app-owned resource such as:

```text
example-crm@1-0/public/mcp/customers
  customers_search -> crm:read
  customers_update -> crm:write

managed guard
  -> exact caller card + active catalog
  -> resource/tool/grant check
  -> app-owned customer service
     or declared connected-account claim enforcement
```

The architecture applies equally to app-owned REST resources. The resource URL,
tool/operation names, grants, account requirements, and domain implementation
belong to that app and its descriptors.

The current and hypothetical surfaces compare as follows:

| Surface | Outer managed guard | Connected-account enforcement | Named-service admission |
| --- | --- | --- | --- |
| Conversation MCP | Resource + `conversations_export` + `conversations:read`. | None. | None. |
| Productivity MCP | Resource + each productivity tool's exact grants. | Per-tool `ToolClaimPolicy` and account broker. | None. |
| Named-services MCP | Resource + generic named-service MCP tools + `named_services:use`. | Provider requirements resolved for the inner operation. | Required for decoded namespace/operation. |
| Native hosted-agent named-service tool | Trusted agent selector and card. | Provider requirements resolved for the inner operation. | Required for each `_call`. |
| Custom app REST/MCP surface | App-owned resource, operations/tools, and grants. | Optional declared provider claims and shared account broker. | None when the app invokes its domain logic directly. |
| Custom named-service provider | Governed by whichever outer/native entrance calls it. | Optional provider requirements for its inner operations. | Required at the common dispatcher. |

## Exact Bearer Binding

A delegated-client bearer session retains the exact authenticated card binding:

```text
identity_authority.delegated_card_binding
  access_id          exact card selected at authentication
  client_id          delegated client identity
  grantor/delegate   identity relationship
  expires_at         authenticated binding metadata
```

This is session identity, not a cached grant. The `access_id` selects one card
when a user owns several delegated cards. The next invocation uses that exact
selector to resolve the current card revision and active catalog.

## Named-Service Direct And Relayed Dispatch

After execution enters the named-service subsystem, provider discovery answers
where the provider runs and `NamedServiceAdmission` answers whether this inner
invocation may reach it. The direct and relayed paths converge on the same local
provider registry:

```text
decoded request + admission input
                 |
                 v
        local platform caller available?
             /                 \
           yes                  no
            |                    |
            v                    v
 resolve/reuse admission     Data Bus request
 bind account scope          request + actor + typed selector
 local provider registry             |
            |                        v
            |                 provider-bundle worker
            |                 restore and bind actor
            |                 validate selector against actor
            |                 resolve current admission once
            |                 bind account scope
            |                 local provider registry
            |                        |
            +-----------+------------+
                        |
                        v
                 provider operation
                        |
                        v
                 structured response
```

The relay selector contains identifiers and provenance. The target owns the
card/catalog lookup and account binding. Raw cards, catalogs, account scopes,
and credentials stay in their owning trusted services.

The Connection Hub lookup for one card by `access_id` is Redis-first. Durable
storage participates when a validated serving projection must be restored, and
decides membership when a grantor's cards are listed. Relay retries use one
message id; the target records the completed outcome, redelivery returns that
outcome, and the provider executes once per message id.

## Invocation Boundaries

One invocation is the unit of authority:

- each managed REST request receives its own guarded decision;
- each MCP tool call, including each item in a batch, receives its own guarded
  decision;
- each plain account-backed tool resolves its declared account requirements for
  that tool call;
- each native agent tool `_call` receives a fresh decision;
- one relay request is validated and resolved once at its target;
- provider streaming is admitted once before the provider returns response
  metadata and its asynchronous bytes;
- consumption of an admitted byte stream uses that invocation's result;
  authorization and scope binding occur once at provider invocation.

The current relay transports ordinary request/reply results. Direct bridges own
provider byte-stream delivery.

## Changes And Revocation

Current state takes effect at the invocation boundary:

```text
card edit / revoke / expiry / new active catalog
                         |
                         v
             durable current state + Redis projection
                         |
                         v
next invocation resolves the new decision

existing bearer/session = identity + exact selector
authorization decision  = current state resolved for this invocation
```

A card edit narrows or widens the next invocation according to the newly
committed selection and current catalog. Revocation commits a durable revoked
revision, updates live serving state, and invalidates the applicable credential
or session records. An invocation that already received its singular decision
completes under that decision.

Catalog publication changes the deployment ceiling. A capability removed from
the active catalog becomes ineffective on the next invocation even while the
stored card preserves the old selection as drift evidence.

## Failure Semantics

| Condition | Result |
| --- | --- |
| Selector is missing or does not match the restored actor/session. | Structured admission denial before provider selection. |
| Exact card is expired, revoked, malformed, or identity-mismatched. | Structured delegated-authority denial. |
| Current active catalog or required card serving state cannot be obtained or validated. | `503 temporarily_unavailable`; a valid shared-state decision is required. |
| Card selected the capability but the active catalog removed it. | `403 delegated_capability_no_longer_available`. |
| Active catalog exposes the capability but the card did not select it. | Existing missing-grant or consent denial. |
| Account binding or provider claims are incomplete. | Existing connection, account-selection, or claim-consent response. |

These outcomes preserve the distinction between identity failure, unavailable
shared authority state, deployment policy change, user-grant absence, and
connected-account consent.

## Source Map

- Card/catalog durability and editor drift:
  [Delegated Access Cards](../sdk/solutions/connections/delegated-cards/delegated-cards-README.md)
- Reusable managed MCP doors and caller families:
  [Platform MCP Over Connection Hub](../sdk/solutions/mcp/platform-mcp-over-connection-hub-README.md)
- Managed MCP configuration and connected-account claims:
  [Authenticated MCP](../sdk/solutions/connections/authenticated-mcp/authenticated-mcp-README.md)
- Isolated supervisor and Data Bus round trip:
  [Named Services From An Isolated Runtime](../sdk/solutions/kdcube-services/named-services-from-isolated-runtime-README.md)
- Request, discovery, provider, and stream contracts:
  [Named-Service Integration](../sdk/namespace-services/integration-README.md)
- Cross-runtime actor and policy restoration:
  [Cross-Runtime Context](../runtime/cross-runtime-context-README.md)
- Trust boundaries and credential ownership:
  [Security And Trust Model](security-and-trust-model-README.md)
