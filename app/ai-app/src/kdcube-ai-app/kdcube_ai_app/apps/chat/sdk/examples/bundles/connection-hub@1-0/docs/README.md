---
id: kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/connection-hub@1-0/docs/README.md
title: "Connection Hub Design"
summary: "Design overview of connection-hub@1-0: connection edges, connected provider accounts, delegated client credentials, shared OAuth callbacks, named-service boundaries, and the Connections widget."
status: active
tags: ["app", "connection-hub", "identity", "connections", "named-services", "mcp", "oauth", "delegated-credentials", "email", "design"]
keywords: ["connection hub app", "delegated access", "connected accounts", "grant mutation csrf", "live grant authority"]
updated_at: 2026-08-01
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/delegated-credentials/oauth-delegated-credential-protocol-adapter-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/bundle-platform-integration-README.md
---

# Connection Hub Design

`connection-hub@1-0` is the platform example app (bundle) for connecting
external identities and provider accounts to KDCube, and for delegating bounded
KDCube access back out to external clients.

It has three responsibilities that must stay separate:

```text
connection edges
  external proof -> platform user id
  examples: google:person@example.com, telegram:314062490, bundle:app:user-77
  used by: auth bridges, inbound hooks, cross-channel user routing

delegated account connections
  platform user id -> external account credential with approved provider claims
  examples: Gmail OAuth token, Slack workspace OAuth token, iCloud app password
  used by: automation that acts for the user

delegated client credentials
  platform user -> external automation allowed into selected KDCube boundaries
  examples: MCP OAuth connector, manual script/agent bearer
  used by: external clients acting for the approving KDCube user
```

Do not infer platform roles from delegated accounts. The long-term authority is:

```text
verified external identity
  -> Connection Hub connection edge
  -> platform principal/role resolver
  -> platform user id + roles/permissions
```

## What it is

A user-scoped hub that:

- stores external connection edges for the current platform user;
- creates short-lived connection-edge challenges for proof flows such as
  Telegram Mini App linking;
- resolves a verified external identity to a platform principal envelope;
- resolves a current actor/platform user to its linked identity family and
  canonical user ids for aggregation surfaces such as memories;
- serves the public `connections` named-service contract over HTTP via the
  `named_service` op, and registers that provider into discovery so other apps
  can resolve delegated tokens (`bundle_registry` transport);
- owns the single shared OAuth callback route used by all providers/apps;
- offers a Connections settings widget for users to link identities and
  connect/disconnect accounts;
- issues and revokes external-client credentials through OAuth consent or
  **Delegated by KDCube -> Create automation access**;
- renders exact named-service namespace operations for manual automation access,
  validates them against the descriptor, and persists the narrowed policy in
  the delegated grant record.

## Building blocks it wires

- **SDK `connections.hub.connection_edges`** — connection-edge storage and the temporary
  principal-resolution fixture, plus one-time connection-edge challenges. This
  module deliberately returns a
  `role_resolution` envelope that points to the future platform principal/role
  resolver instead of treating the app as the role authority.
- **SDK `connections.hub.resolver`** — identity-family resolver. Given the current
  actor/platform user id, it returns platform authority identity,
  provider/integration identities, and `memory_user_ids` for server-side
  aggregation.
- **`integrations/connections`** — `ConnectionStore` (user-scoped, shared-tokens),
  `ConnectionsProviderBase`, and the connection registry of `ConnectionProvider`s.
  Importing the providers package registers the built-ins (Slack, …).
- **SDK `connections.hub.provider_impl`** — `ConnectionHubProvider`, the concrete
  `connections` provider. It makes the STORAGE CHOICE explicit (user-scoped
  `ConnectionStore`) and resolves the per-request user from the named-service
  context. `get_token` auto-picks only when exactly one account is connected,
  otherwise raises `AmbiguousConnectionAccount`.
- **SDK `connections.hub.authenticators`** — authenticator modules for request
  authentication. Telegram proof verification is implemented. Slack, OIDC,
  Google, webhook HMAC, and API-key providers are explicit module slots, not
  role authority shortcuts.
- **SDK `connections.hub.authenticator_store`** — Postgres store for
  widget-managed request-authenticator metadata. It stores `secret_ref`, never
  secret values. See [storage/README.md](storage/README.md).
- **`integrations/email`** — iCloud app-password settings, exposed through
  dedicated `email_*` ops. Gmail is not handled here; it is a `connections`
  provider.
- **`connections_settings` widget** — built from `ui/widgets/connections`.

> Current correction: Gmail rides the `connections` framework. The email
> integration serves iCloud app-password settings only.

## The three-level model

```text
provider (mechanics, no creds)
  -> connector app (operator app config; MANY per provider; carries client_id/secret + claim ceiling)
     -> user account (one credential, records connector_app_id and approved claims)
```

Providers are dynamic (registry-driven). Connector apps are deploy config under
`connections.delegated_to_kdcube.providers.<provider>.connector_apps`.
Accounts are user state created by the OAuth or credential flow. The delegated
integration OAuth callback is shared by the OAuth providers and signed by
`connections.delegated_to_kdcube.oauth_state_secret`.

Application tool claim policy is not stored as a separate capability registry.
A tool declares its own provider/connector/claims requirements beside the tool
definition, for example under `tool_claims.<tool>.connections.delegated_to_kdcube.connected_accounts`.
The SDK resolves those declarations into `ToolClaimPolicy` objects.

The first runtime integration is agent preflight: when a ReAct agent starts a
turn, it checks all configured tool claim policies for that agent against the
current platform user's connected accounts. If something is missing, the
workflow emits a structured `needs_connected_account_consent` chat step with a
Connection Hub URL. The chat UI shows a banner, the user opens Connection Hub
to connect/upgrade the account, and then retries the request. Later tool-level
runtime checks can reuse the same payload shape.

Connection Hub may scan tool declarations for admin visibility, but the
application bundle/tool definition remains the source of truth for which
provider claims a tool needs.

## Delegated automation and provider-backed named services

The generic named-services MCP resource has two authorization layers and may
have a third provider prerequisite:

```text
managed MCP boundary
  resource + generic MCP tool + named_services:use
        |
        v
named-service boundary
  namespace + selected operation + namespace grants
        |
        v
connected provider boundary (only when declared)
  grantor's account + provider claims
```

The manual **Create automation access** screen projects the first two layers
from `connections.delegated_credentials.oauth.resources[]`. For a resource with
`named_services`, it submits:

```text
named_service_operations
  <resource>
    <namespace>
      [<existing descriptor operation>, ...]
```

`AutomationAccessService` validates the exact selection and stores a narrowed
copy of the descriptor's `named_services` policy in `GrantStore`. KDCube
Services uses that stored policy for its runtime `NamedServiceBoundaryCatalog`,
so unselected namespaces and operations are denied without a second registry.
Pointer-backed OAuth and bundle-session credentials resolve this record on
every managed call and refresh. Missing, expired, malformed, unavailable, or
mismatched live authority fails closed; a pointed credential does not recover
its older embedded grant snapshot.

A provider may separately publish `metadata.connected_accounts`. Connection
Hub renders those requirements and links to **Delegated to KDCube**, preserving
flat or operation-specific claim structure. This is presentation and consent
guidance only: provider discovery metadata and provider credentials are never
copied into the automation grant. The provider resolves the grantor's connected
account at call time.

## Surfaces

- `named_service` (operations) — the whole `connections` contract.
- `connection_edges_list`, `connection_edge_upsert`, `connection_edge_remove`,
  `connection_edge_challenge_create`, `connection_edge_challenge_status`,
  `identity_resolve`, `identity_family_resolve`,
  `delegated_identity_scope_resolve` (operations) — connection-edge management,
  proof challenges, principal resolution, linked-family user-id expansion, and
  delegated credential identity-scope resolution.
- `telegram_connection_edge_complete` (public) — validates Telegram Mini App
  `initData` and completes a pending Telegram link challenge.
- `telegram_connection_edge_status`, `telegram_connection_edge_start`,
  `telegram_connection_edge_remove` (public) — Telegram-first Mini App linking:
  read current link, create a provider-proof challenge, and unlink the current
  Telegram subject.
- `federated_data_bus_claim` (public) — validates the promoted request auth
  context and returns a short-lived Socket.IO token for the Connection Hub
  widget's own live channel.
- `request_authenticate` (public) — platform/app request-auth selector endpoint:
  request envelope in, verified provider identity plus linked platform authority
  out.
- `authenticators_list`, `authenticators_upsert`, `authenticators_remove`
  (operations) — admin/widget APIs for request-authenticator metadata. These
  APIs reject secret values; use `secret_ref` and bundle secrets. Controlled
  surfaces carry `X-KDCube-Auth-Authority-ID` and
  `X-KDCube-Auth-Authenticator-ID`.
- `delegated_to_kdcube_catalog`, `delegated_to_kdcube_start_oauth`,
  `delegated_to_kdcube_connect_credential`, `delegated_to_kdcube_disconnect`,
  `delegated_to_kdcube_resolve` (operations) — current widget and server-side
  broker helpers for delegated external accounts.
- `delegated_to_kdcube_oauth_callback` (public) — the shared OAuth browser
  redirect for delegated to KDCube providers such as Gmail and Slack.
- `delegated_access_list`, `delegated_access_create`,
  `delegated_agent_grant_create`, `delegated_access_revoke` (operations) —
  list, issue, extend, and revoke delegated access. State-changing Connection
  Hub operations opt into proc's one-time CSRF contract for
  cookie-authenticated browsers, including grant, connected-account,
  connection-edge, authenticator, DCR-allowlist, email-account, and generic
  named-service mutations. Every effective POST operation is classified in
  `CSRF_PROTECTED_OPERATION_ALIASES` or
  `CSRF_EXEMPT_POST_OPERATION_ALIASES`; public protocol POSTs have a separate
  explicit exemption inventory. Explicit token and authenticated internal
  calls keep their existing request proof. Manual named-service access accepts the exact
  `named_service_operations[resource][namespace][]` selector.
- `email_accounts_status`, `email_connect_app_password`, `email_disconnect_account`
  (operations) — older iCloud-only email integration surface.
- `connections_settings` (widget) — the React/Redux settings UI.

## Telegram Mini App embedding

Workspace hosts the Connection Hub widget in its Telegram Mini App Connect tab.
This uses the same widget handshake as scene-hosted widgets:

```text
Connection Hub iframe
  -> CONFIG_REQUEST(identity=CONNECTIONS_WIDGET)
Workspace Telegram host
  -> CONFIG_RESPONSE(config.authContext.headers)
Connection Hub iframe
  -> public Connection Hub APIs with promoted authContext.headers
```

The child iframe does not read `window.parent.Telegram`, does not know bot
tokens, and does not call Workspace APIs for Connection Hub work. It receives
an opaque header map and promotes it onto its own requests. Connection Hub then
validates the request through its authenticator modules.

For link completion, the child iframe creates a Connection Hub live channel:

```text
iframe -> federated_data_bus_claim
       -> short-lived Data Bus token backed by an actor UserSession
       -> Socket.IO session scoped to connection-hub@1-0
       -> telegram_connection_edge_start(live_event_session_id)
browser claim -> connection_edge_challenge_status
       -> explicit user confirmation
       -> connection_edge_challenge_claim(confirmed=true)
       -> connection_hub.edge.changed to the iframe session
       -> iframe reclaims/reconnects and refreshes link status
```

There is no polling in this flow. The browser-side claim returns normally for
the browser user, and Connection Hub separately signals the original Telegram
Mini App iframe that initiated the provider-proof challenge.

The Data Bus claim is owned by Connection Hub, not the host app. For unlinked
Telegram users it creates a low-authority actor session such as
`telegram_100200300`. After a link exists, the next claim keeps that actor id
and projects the linked platform authority into `session.identity_authority`.

The standalone browser claim page performs its own KDCube platform sign-in using
`/api/cp-frontend-config`; it does not depend on website auth code. See
[journal/2026-06-27-telegram-claim-platform-auth.md](journal/2026-06-27-telegram-claim-platform-auth.md).

See [../interface/README.md](../interface/README.md) for the full contract and
prerequisites, [storage/README.md](storage/README.md) for storage boundaries,
and [journal/README.md](journal/README.md) for build decisions.
