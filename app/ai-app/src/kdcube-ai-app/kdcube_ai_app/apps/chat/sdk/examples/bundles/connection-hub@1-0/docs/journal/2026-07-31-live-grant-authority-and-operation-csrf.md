---
id: connection-hub@1-0/journal/2026-07-31-live-grant-authority-and-operation-csrf
title: "Live Grant Authority and Operation CSRF"
summary: "Records fail-closed live-grant resolution, atomic OAuth state transitions, and exhaustive one-time CSRF protection for Connection Hub mutations."
status: implemented
tags: ["connection-hub", "delegated-credentials", "authorization", "csrf", "security"]
keywords: ["live grant lookup", "stale delegated authority", "bundle operation csrf", "atomic oauth rotation", "grant mutation"]
updated_at: 2026-08-01
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/delegated-credentials/oauth-delegated-credential-protocol-adapter-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/storage-model/storage-model-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/bundle-platform-integration-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/bundle-operation-csrf-README.md
---

# Live Grant Authority and Operation CSRF

## Problems

A delegated access or refresh record can carry a pointer to the Connection Hub
card that currently owns its authority. The previous resolution path could use
the older authority snapshot from the token record when the pointed card could
not be resolved. That made an unavailable or malformed live record weaker than
an explicit revocation or narrowing.

Connection Hub grant create, agent-grant create, and revoke operations are
state-changing browser calls. Platform session cookies authenticate the user,
but cookies are ambient browser credentials, so these operations also require
an explicit request-origin proof.

The OAuth store also had three distributed races: authorization-code consume
and consent-CSRF consume each performed `GET` followed by `DELETE`, while
refresh rotation performed `GET`, `DELETE`, and replacement `SET` separately.
Two workers could therefore accept the same single-use record or both rotate
the same refresh token even though every individual Redis command succeeded.

## Implementation

Pointer-backed credentials now resolve and validate the current card on every
managed MCP/REST call and refresh. The resolver checks schema, structure,
expiry, access id, client id, grantor subject, delegate subject, resource
grants, operations, and account scope. Its outcomes are distinct:

```text
valid card                         use current authority
absent or expired card             treat as revoked
store/JSON/schema/binding failure  deny; never use stale snapshot
legacy record without pointer      preserve snapshot contract
```

Refresh rotation persists the live scopes and operations into the replacement
record while retaining the card pointer. The pointer remains authoritative on
the next refresh and call.

Single-use OAuth records now transition through awaited Redis Lua scripts:

```text
authorization code / consent CSRF
  GET + DELETE in one script

refresh token
  read exact record -> validate live authority
  -> compare exact record + delete old + create replacement in one script
```

There is no process-local lock. Every worker and replica competes on the same
tenant/project Redis key, and the Redis transition admits one winner. Dynamic
client reads also renew their sliding TTL in one Lua action.

The bundle API surface now supports `csrf=True` for selected `POST`
operations. The decorator supplies the application default and the matching
`surfaces.as_provider.api.operations.<alias>.POST.csrf` descriptor property can
override it for a deployment. A cookie-authenticated browser obtains a
one-time token from the same operation's `/csrf` route and returns it in
`X-KDCube-CSRF-Token`. Proc
binds the token to the authenticated subject, tenant, project, bundle,
operation, and method, then consumes it atomically. Explicit bearer/ID-token
requests and authenticated internal peer calls retain their existing request
proof and do not use this browser exchange.

Connection Hub classifies every effective operation POST. Mutations covering
delegated grants, provider accounts, connection edges, authenticator admin,
DCR allowlist admin, email accounts, and the generic named-service adapter
enable the contract. Read-only POST adapters and the
inherited general agent-selection write are explicit exemptions; public POSTs
are separately inventoried under their protocol-specific request proof. A
manifest regression test requires the protected and exempt sets to cover the
complete effective surface.

Delegated state-store resolution belongs to the OAuth/authorization adapter,
below MCP and REST dispatch. It reuses the request application's shared async
client when available and otherwise uses the platform shared-client factory.
Store failures carry a non-secret operation name, are logged at OAuth route
and managed MCP/REST guard boundaries, and return
`503 temporarily_unavailable` instead of an unstructured `500`.

## Deployment Contract

Both controls use process-shared state resolved below the HTTP dispatch layer.
The current Redis implementation derives the local `chat-proc` connection from
descriptor-owned `infra.redis`; the ECS task receives the corresponding
logical secret from Secrets Manager. A token minted on one worker can therefore
be consumed on another. Backend resolution or I/O failure returns a logged 503.

There is no deployment-global switch, environment variable, local file, or
process-local lock. The optional `csrf` property belongs to one API surface in
the bundle descriptor. Redis unavailability fails closed for operations that
resolve to `csrf: true` in both deployment shapes.

## Boundary Correction

The first implementation also marked SDK-owned `agent_selection_update` as
CSRF protected on `BaseEntrypoint`. Every chat-capable application inherited
that choice, and the existing settings picker began receiving `403` even
though its operation had never selected the new contract. The marker was
removed from the base class before release. Generic bundle POST behavior now
remains unchanged unless the operation decorator or effective descriptor
surface opts in. A regression test exercises this exact cookie-authenticated
settings-write path without a CSRF token.

The canonical default-install, local reference, custom-authority, staging, and
demo descriptors were inspected. None declares a Connection Hub API-surface
override, so they retain the built-in app's explicit defaults. No blanket
descriptor policy was added.

The Apps dashboard now receives the effective CSRF value, decorator default,
override path, and explicit-override state for each API endpoint. For eligible
operation POSTs it can write `csrf: true` or `csrf: false`, or clear the value
to restore the decorator default. The MCP editor intentionally has no CSRF
control because MCP uses explicit protocol credentials rather than ambient
browser cookies.

## MCP 2026-07-28 Relationship

This hardening protects the OAuth and delegated-authority state used by MCP;
it is separate from the MCP wire codec. KDCube's official-SDK v2 regression
records the modern HTTP exchange and verifies stateless per-request metadata,
routing headers, no session header, `resultType`, server identity, and cache
hints, while retaining a legacy initialize test. The focused wire and OAuth
suites establish support for KDCube's exposed core capabilities. The official
full conformance runner and live external DCR/CIMD journeys remain the evidence
gate for an unqualified MCP 2026-07-28 conformance claim.

## Documentation Ownership

The contracts are intentionally split by reader task:

- [Bundle Operation CSRF](https://github.com/kdcube/kdcube/blob/main/app/ai-app/docs/sdk/bundle/bundle-operation-csrf-README.md)
  owns the browser token exchange, opt-in rules, distributed consumption, and
  failure behavior.
- [Govern App API, MCP, and Widget Surfaces](https://github.com/kdcube/kdcube/blob/main/app/ai-app/docs/recipes/apps/govern-provider-surfaces-README.md)
  owns the builder workflow through decorators, descriptors, and the Apps
  dashboard.
- [Delegate A KDCube Service To An External Client](https://github.com/kdcube/kdcube/blob/main/app/ai-app/docs/recipes/connections/delegate-kdcube-service-to-external-client-README.md)
  owns the external-client journey and its post-consent live-authority checks.
- [Protect Bundle MCP With Managed Credentials](https://github.com/kdcube/kdcube/blob/main/app/ai-app/docs/recipes/connections/protect-bundle-mcp-with-managed-credentials-README.md)
  owns the app-builder journey for a managed MCP door.
- [OAuth Delegated Credential Protocol Adapter](https://github.com/kdcube/kdcube/blob/main/app/ai-app/docs/sdk/solutions/connections/delegated-credentials/oauth-delegated-credential-protocol-adapter-README.md)
  owns atomic OAuth transitions, live-card validation, structured availability
  responses, and the complete regression matrix.

The recipes state the guarantees a builder can rely on and link to the adapter
for protocol depth. They do not require an app to know that Redis or Lua is the
current implementation behind those guarantees.

## Verification

- The complete delegated OAuth adapter suite passes: `183 passed, 1 skipped`;
  the skipped case is the opt-in real-Redis test when
  `KDCUBE_TEST_REDIS_URL` is absent.
- The real-Redis run passes the overlap test for authorization code, consent
  CSRF, and refresh rotation against the local disposable Redis container.
- Focused proc operation-auth, surface override, decorator metadata, and
  Connection Hub manifest coverage passes: `32 passed`.
- Bundle decorator metadata coverage passes: `2 passed`.
- Modern/legacy MCP, CIMD, and mounted Connection Hub discovery coverage passes
  in the rebuilt processor image; the modern test inspects the bytes and
  headers on the actual `KDCubeMCPServer` exchange.
- Cookie-authenticated protected operations require a context-bound token and
  reject missing, reused, or mismatched tokens.
- Cookie-authenticated POSTs without an effective `csrf: true` declaration run
  under their existing operation contract; the CSRF discovery route reports
  `csrf_required: false`.
- Explicit bearer and internal peer calls remain compatible.
- Token-store failure returns `503` at OAuth and managed MCP/REST boundaries
  rather than running the operation.
- Two concurrent consumers of an authorization code or consent-CSRF record
  produce exactly one success.
- Two concurrent refresh rotations using the same authorized snapshot create
  exactly one replacement token.
- Pointer-backed managed MCP/REST calls reject malformed or unavailable live
  authority.
- Refresh does not rotate when current authority cannot be trusted.
- Live narrowing, including an intentionally empty grant set, reaches both the
  access token and replacement refresh record.
- Legacy records without a card pointer retain their snapshot behavior.

Standalone `tsc --noEmit` passes for the Connection Hub widget. Per the widget
maintenance contract, no local widget build was run.
