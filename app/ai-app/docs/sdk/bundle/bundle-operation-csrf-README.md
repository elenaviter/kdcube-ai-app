---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/bundle-operation-csrf-README.md
title: "Bundle Operation CSRF"
summary: "Opt-in, descriptor-overridable CSRF protection for cookie-authenticated bundle operation POSTs, using distributed single-use request proofs."
status: current
tags: ["sdk", "bundle", "api", "security", "csrf"]
keywords: ["bundle operation csrf", "csrf api surface", "single use csrf token", "cookie authenticated post", "distributed csrf", "shared request proof"]
updated_at: 2026-08-01
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/bundle-platform-integration-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/configuration/bundles-descriptor-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/arch/security-and-trust-model-README.md
---
# Bundle Operation CSRF

KDCube offers a distributed CSRF mechanism that an application can enable for
selected bundle operations. The default is `false`: an operation keeps its
existing request contract unless its resolved surface declares `csrf: true`.

This is an API-surface policy, not a rule inferred from the `POST` verb. Each
application decides which operations need the mechanism. OAuth callbacks,
webhooks, public protocol routes, bearer-authenticated APIs, and other
surfaces may use their own request-proof contract.

## Declare The Policy

The decorator supplies the application default:

```python
from kdcube_ai_app.infra.plugin.bundle_loader import api


@api(method="POST", alias="account_disconnect", route="operations", csrf=True)
async def account_disconnect(self, account_id: str, **kwargs):
    ...
```

The bundle descriptor can override that default at the exact provider surface:

```yaml
bundles:
  items:
    - id: my-app@1-0
      config:
        surfaces:
          as_provider:
            api:
              operations:
                account_disconnect:
                  POST:
                    csrf: true
```

`csrf: false` explicitly disables a decorator default for that deployed app.
The alias-level fallback is also accepted when the alias has one method:

```yaml
surfaces:
  as_provider:
    api:
      operations:
        account_disconnect:
          csrf: true
```

The current mechanism is valid only for `POST` APIs on the `operations`
route. Other operations remain unaffected.

## Browser Exchange

For a cookie-authenticated call to a protected operation:

```text
GET  .../operations/account_disconnect/csrf
  -> {csrf_required: true, csrf_token: "...", expires_in: 600}

POST .../operations/account_disconnect
X-KDCube-CSRF-Token: ...
  -> operation result
```

The token is bound to the authenticated subject, tenant, project, bundle,
operation, and HTTP method. It expires after ten minutes and is consumed once.
A reused, expired, absent, or context-mismatched token returns `403`.

The `/csrf` route resolves the effective operation surface first. It returns
`csrf_required: false` when that surface did not opt in or when an explicit
bearer/ID-token header provides non-ambient request proof.

## Scaled Runtime Contract

The request-level CSRF service resolves its shared state backend below HTTP
dispatch. The current backend uses async Redis, and consumption is an atomic
Lua `GET` plus `DELETE`, so the browser's GET and POST may land on different
workers and only one POST can consume the token. Backend unavailability
returns `503` and the protected operation does not run.

Authenticated internal peer calls retain their existing request proof and do
not perform the browser exchange.

## Client Responsibility

The UI that invokes a protected cookie-authenticated operation performs the
GET before its POST and sends the returned header. Enabling `csrf` for an
existing operation therefore includes updating that operation's browser
client. Unprotected operations do not need a token request.

Connection Hub uses this capability for its selected browser mutations. The
capability is part of the general bundle API contract and is available to any
application surface that opts in.
