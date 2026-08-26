---
id: connection-hub@1-0/agents
title: "Connection Hub Builder-Agent Onboarding"
summary: "Builder-agent onboarding guide for the platform Connection Hub example app: connection edges, connected provider accounts, delegated client credentials, OAuth callbacks, named-service boundaries, and the Connections widget."
status: "active"
tags: ["agents", "builder", "onboarding", "connection-hub", "identity", "connections", "oauth", "mcp", "named-services", "delegated-credentials", "react", "redux"]
see_also:
  - "repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/connection-hub@1-0/README.md"
  - "repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/connection-hub@1-0/docs/README.md"
  - "repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/connection-hub@1-0/docs/storage/README.md"
  - "repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/connection-hub@1-0/docs/journal/README.md"
  - "repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/connection-hub@1-0/docs/journal/2026-06-25-connection-edges-and-delegated-connections.md"
  - "repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/connection-hub@1-0/docs/journal/2026-06-26-request-authenticators.md"
  - "repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/connection-hub@1-0/interface/README.md"
  - "repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/connection-hub@1-0/interface/connection-hub.openapi.yaml"
  - "repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/connection-hub@1-0/config/bundles.template.yaml"
  - "ks:docs/service/auth/app-hosted-platform-login-and-session-README.md"
  - "ks:docs/service/auth/app-simple-idp-bridge-README.md"
  - "ks:docs/sdk/solutions/ecosystem-component/components-ecosystem-README.md"
  - "ks:docs/sdk/namespace-services/README.md"
---

# Connection Hub Builder-Agent Onboarding

This is the builder-agent landing page for `connection-hub@1-0`.

The app is the playground for connecting external identities and delegated
accounts into KDCube. Keep these two concepts separate:

```text
connection edge
  purpose: prove that an external identity belongs to a platform user
  examples: google:elena@example.com, telegram:314062490, bundle:app:user-77
  used by: auth bridges, inbound webhooks, channel-specific entrypoints

connected account
  purpose: let automation use a user's external account with delegated access
  examples: Gmail OAuth token, Slack workspace token, iCloud app password
  used by: user automation and app workflows that act for the user

delegated client credential
  purpose: let an external script, agent, or MCP client enter selected KDCube
           resources and operations on behalf of an approving platform user
  examples: manual automation bearer, OAuth-issued MCP connector credential
  used by: managed REST/MCP guards and the named-service bridge

request authenticator
  purpose: verify that an incoming request proves a channel identity, then
           project linked platform authority into a UserSession
  examples: Telegram initData, Slack signature, webhook HMAC, API key
  used by: gateway auth selector and app/channel handlers
```

Do not infer platform roles from a delegated account token. A Gmail token can let
automation read/send mail, but it does not prove admin rights. The target flow is:

```text
verified external identity
  -> Connection Hub connection edge
  -> platform principal/role resolver
  -> platform user id + roles/permissions
```

`connection-hub@1-0` may include a configured role-binding fixture for local
demos, but that fixture is not the long-term security authority.

## Read First

Start with these app-local files:

- [README.md](README.md)
- [docs/README.md](docs/README.md)
- [docs/storage/README.md](docs/storage/README.md)
- [docs/journal/README.md](docs/journal/README.md)
- [interface/README.md](interface/README.md)
- [interface/connection-hub.openapi.yaml](interface/connection-hub.openapi.yaml)
- [config/bundles.template.yaml](config/bundles.template.yaml)
- [config/bundles.secrets.template.yaml](config/bundles.secrets.template.yaml)
- [entrypoint.py](entrypoint.py)
- SDK core:
  `kdcube_ai_app.apps.chat.sdk.solutions.connections.hub`
- [ui/widgets/connections/src/App.tsx](ui/widgets/connections/src/App.tsx)

When changing auth/session behavior, also read the platform docs:

- `ks:docs/service/auth/app-hosted-platform-login-and-session-README.md`
- `ks:docs/service/auth/app-simple-idp-bridge-README.md`

Read the journal before changing behavior. Add a dated journal entry for every
implementation round that changes API contracts, auth semantics, storage shape,
widget behavior, or deployment config.

## Product Shape

```text
Connection Hub app
  entrypoint.py
    operations API
      - connections_*: delegated account connection helpers
      - identity_*: connection edge and principal-resolution helpers
      - request_authenticate: provider-proof verification for request auth
      - email_*: iCloud app-password helper ops
    public OAuth callback
      - connection_oauth_callback

  named service provider
    namespace: connections
    purpose: cross-app token resolution for delegated accounts

  widget: connections_settings
    source: ui/widgets/connections
    stack: React + Redux Toolkit
    purpose: one user-facing surface for connection edges and connected accounts

  delegated access
    OAuth consent or manual Create automation access
    resource_grants + selected top-level operations
    optional exact named_service_operations selection
    provider-account prerequisites stay in Delegated to KDCube
```

## Implementation Rules

- Keep connection edges and delegated account connections separate in code,
  storage, docs, and UI labels.
- Keep request authenticators separate from connected accounts. A Telegram bot
  token or Slack signing secret proves requests; it is not a delegated user
  account token.
- Give every app/provider surface stable non-secret authority/authenticator ids.
  KDCube-controlled surfaces should carry them as
  `X-KDCube-Auth-Authority-ID` and `X-KDCube-Auth-Authenticator-ID`; raw
  provider-shape matching is fallback for uncontrolled hooks.
- Use `role_providing` only for authenticators that directly establish platform
  authority. Linked external providers such as Telegram normally keep it false.
- Do not decide real platform roles inside this app. Call or model a platform
  principal/role resolver after identity resolution.
- Do not grant roles because an external account exists. Roles belong to the
  platform principal.
- Do not put OAuth client secrets or user tokens in descriptor templates.
- Do not put request-authenticator secret values in Postgres or bundle-local
  state. Store only `secret_ref` metadata there; secret values stay in
  `bundles.secrets.yaml` or the configured bundle secrets provider.
- Keep the widget as a React/Redux app. Add slices/components instead of turning
  it into an ad hoc script.
- Keep `entrypoint.py` as shallow orchestration. Storage, resolver,
  authenticator, and provider domain logic belongs in the platform SDK package
  `kdcube_ai_app.apps.chat.sdk.solutions.connections.hub`.
- Keep `interface/README.md`, `docs/README.md`, config templates, and journal in
  sync when changing an API or behavior.

## Runtime Checks

After backend changes:

- run Python syntax checks for changed modules;
- refresh the local KDCube runtime before testing the app through the platform.

After widget changes:

- run `npm run build` inside `ui/widgets/connections` when dependencies are
  available;
- test the widget through KDCube rather than only through Vite when validating
  auth/config propagation.
