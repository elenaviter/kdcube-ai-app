---
id: repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/resolve-connected-credential-README.md
title: "Resolve a Connected Credential in Tool Code"
summary: "The credential-resolution mechanism every integration shares: after both authorization gates pass, tool code asks the account broker for the user's provider credential, which resolves and refreshes only at the trusted boundary, for one call - so no provider token ever lives in your code, the prompt, or the model context."
status: active
tags: ["recipes", "connections", "connection-hub", "delegated-to-kdcube", "connected-accounts", "credential-resolution", "trusted-boundary"]
updated_at: 2026-07-27
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/custom-oauth-oidc-service-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/google-service-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/slack-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/integrations/provider-error-contract-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/authenticated-mcp/authenticated-mcp-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/delegated-accounts/delegated-accounts-README.md
  - repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/integrations/connected_accounts.py
  - repo:app-ecosystem/products/connection-hub/packages/connection-hub/src/connection_hub/agent_account_scope.py
---
# Resolve a Connected Credential in Tool Code

Every integration recipe - [Google (Gmail, Sheets)](google-service-README.md),
[Slack](slack-README.md), or a
[custom OAuth/OIDC service](custom-oauth-oidc-service-README.md) -
wires a specific provider. This recipe covers the one step they all share: how
your tool code, at call time, obtains the user's provider credential without any
token ever living in your code.

This is the **delegated to KDCube** direction, resolved at the trusted boundary:

```text
caller (hosted agent OR external MCP app)
  |  delegated bearer
managed door guard      gate 1 (caller grant) + gate 2 (account + binding) pass
  |  authorized request for a resolved user - NO token
your tool code          "this user's gmail:read on account A"
  |  broker call at the trusted boundary
account broker          resolves + refreshes the stored connected credential
  |  live provider credential, this call only
provider API (Gmail / Sheets / Slack / your S1)
```

The two gates decide *whether* the call may run (see the
[authenticated-MCP chain](../../../sdk/solutions/connections/authenticated-mcp/authenticated-mcp-README.md)).
This recipe is what happens *after* they pass: resolution.

## 1. Where resolution sits

Your tool never receives a token as an argument and never reads one from
config. When a call reaches your code, the platform has already authorized it -
the request carries a resolved user (tenant, project, user id), a namespace,
an operation, and the caller's identity. Your code turns that into exactly one
resolution call per provider claim it needs.

The credential is fetched **only here, per call**. Nothing on your side of the
fence stores or holds a provider secret.

## 2. Resolve one claim: `resolve_connected_account_claim`

Ask the broker for one provider claim, for the current invocation:

```python
from kdcube_ai_app.apps.chat.sdk.integrations.connected_accounts import (
    resolve_connected_account_claim,
    connected_account_auth_failure,
    run_with_connected_account_retry,
)
from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.connector_app_resolution import (
    resolve_connector_app_id,
)

PROVIDER_ID = "google"      # a Delegated-to-KDCube provider row (see the config trace)
READ_CLAIM  = "gmail:read"  # a claim declared under that provider

async def read_message(source, message_id: str, account_id: str = ""):
    async def _run():
        credential = await resolve_connected_account_claim(
            source,
            provider_id=PROVIDER_ID,
            connector_app_id=resolve_connector_app_id(PROVIDER_ID),  # RESOLVED, never hardcoded
            claim=READ_CLAIM,
            tool_name="mail.read_message",
            account_id=account_id,          # empty -> broker picks / asks
        )
        if not credential.ok:
            # A structured gate-2 consent envelope - relay it, do not retry blindly.
            return credential.error_envelope(where="mail.read_message")

        resp = await call_gmail_api(
            access_token=credential.access_token,
            message_id=message_id,
        )
        if resp.status_code in (401, 403):
            return connected_account_auth_failure(credential, "Gmail rejected the token")
        return resp.json()

    return await run_with_connected_account_retry(
        source, where="mail.read_message", run=_run,
    )
```

This is the exact shape the built-in `GmailTools._credential` uses
(`sdk/integrations/google/gmail_tools.py`). **Every argument, and where its
value comes from - nothing is hardcoded magic:**

| argument | what it is | where its value comes from |
| --- | --- | --- |
| `source` | the request scope | The platform threads it to your tool; the SDK reads tenant/project/user from it (or the ambient request context). Not something you invent. |
| `provider_id` | the Delegated-to-KDCube **provider row** | A key under `connections.delegated_to_kdcube.providers.<id>` in `bundles.yaml` (`google`, `slack`, or your own). It names a provider you (or a platform integration) registered - adding a new one is step 1 of *Adding a new service* below. |
| `connector_app_id` | which OAuth **connector app** of that provider | **Resolved, never hardcoded.** `resolve_connector_app_id(provider_id)` reads the guarded service's declaration (`surfaces.as_provider.mcp.<door>.connector_apps: {google: gmail}`), which names one of `providers.<provider>.connector_apps.<app_id>`. A provider can have several connector apps; the service picks one. Empty = provider-wide (any connector app's account qualifies). |
| `claim` | the exact provider **claim** the op needs | A key under `providers.<provider>.claims.<claim>` (e.g. `gmail:read`), each mapping to the real OAuth `provider_scopes`; also fenced by the connector app's `allowed_claims`. |
| `tool_name` | your label for logs + consent surfaces | Free-form; name it after the tool/operation (`mail.read_message`). |
| `account_id` | which of the user's connected accounts | Runtime. Empty lets the broker pick or ask - `account_required` returns labeled `candidates`, and you resend with the chosen id. |

`resolve_connected_account_claim` returns a `ConnectedAccountCredential`:

- `credential.ok` / `credential.access_token` - the live token for this call.
- `credential.error_envelope(where=...)` - when resolution cannot be satisfied,
  a structured **gate-2** consent envelope (`needs_connected_account_consent`
  with a `reason`, `retry_hint`, `candidates`, and a `connection_hub_url`),
  ready to return to the caller. Never a bare 403.

The broker restricts *which* connected account may satisfy the claim by the
calling agent's per-account binding (`agent_account_scope`): an agent bound to
`account A` read-only cannot resolve a send claim, and cannot resolve on an
account it is not bound to - default-closed. A non-agent caller has no such
restriction.

### The config trace: three declarations, one vocabulary

Every value above traces to `bundles.yaml`, in three places that must agree.
The built-in **productivity** MCP door is the live reference - it wires Slack
and Google in one surface
(`sdk/examples/bundles/kdcube-services@1-0/surfaces/mcp/productivity.py`).

**1. The provider** - `connections.delegated_to_kdcube.providers.<provider>`:
the provider row, its **connector apps** (each an OAuth client: `client_id`,
`client_secret_ref`, `allowed_claims`), and its **claims** (each mapping to the
real OAuth `provider_scopes`):

```yaml
connections:
  delegated_to_kdcube:
    providers:
      google:
        connector_apps:
          gmail:                         # <- connector_app_id
            client_id: <FILL_ME>
            allowed_claims: [gmail:read, gmail:send]
        claims:
          gmail:read:                    # <- claim
            provider_scopes: [ openid, email, profile,
                               "https://www.googleapis.com/auth/gmail.readonly" ]
```

**2. The service's connector-app pick** -
`surfaces.as_provider.mcp.<door>.connector_apps`: which connector app this door
uses per provider. `resolve_connector_app_id("google")` returns exactly this:

```yaml
surfaces:
  as_provider:
    mcp:
      productivity:
        connector_apps: { slack: slack-demo, google: gmail }
```

**3. The tool's declared need** - the tool's `connected_accounts` policy names
the provider and claims (never the connector app - that is resolved):

```python
"productivity_mail_search": {
    "connections": {"delegated_to_kdcube": {"connected_accounts": [
        {"provider_id": "google", "claims": ["gmail:read"]},
    ]}},
}
```

So the tool declares *provider + claim*, the surface picks *which connector
app*, and the provider config defines *the OAuth client and the real scopes*.
Each string is a key that must exist in the provider config - change the
connector app for the whole door in one line (step 2), and no tool code moves.

## 3. The provider-error contract

Credential resolution is only one boundary. The provider call must preserve
the actual failure category as well. Follow the
[Provider Error And Observability Contract](../../../sdk/integrations/provider-error-contract-README.md)
for every integration.

In particular:

- a missing account or claim returns the Connection Hub consent envelope;
- an expired or revoked token enters the refresh/reconnect path;
- an insufficient scope remains distinguishable from a disabled provider API;
- a generic provider `403` is not automatically treated as an expired token;
- transport and provider errors retain safe status/reason/stage details;
- an ambiguous mutation is not replayed blindly.

## 4. The refresh-and-retry contract

OAuth access tokens expire. `run_with_connected_account_retry` owns the refresh
cycle so your tool body stays a single happy path:

```text
first provider call rejected (401/403)
  -> force-refresh the stored credential
  -> retry the call once
  -> if still rejected, mark the account reconnect_required
  -> return a Connection Hub reconnect link (gate-2 reason: reconnect_required)
```

A credential that can no longer be refreshed is exactly what surfaces gate 2's
`reconnect_required` - a directed, actionable denial, never a silent failure.
Report the auth failure with `connected_account_auth_failure(credential, ...)`
so the wrapper can drive that cycle.

## 5. What the caller sees

Because resolution returns the gate-2 vocabulary, the caller (a hosted agent's
chat banner, or an external MCP client) always gets an actionable answer:

| reason | what the user does |
| --- | --- |
| `connect_required` | connect an account on the backing provider |
| `claim_upgrade_required` | approve the listed claim for an existing account |
| `reconnect_required` | reconnect an account whose stored credential stopped working |
| `account_required` | resend the same call with `account_id` from `candidates` |
| `agent_grant_required` | tick the claim for an account on this caller's grant card |

Only `account_required` is fixed by resending; every other reason needs a
human action at the `connection_hub_url` first.

## 6. Guarantees

- **No token in your code.** Your tool resolves a live credential for one call
  and lets it go; the secret lives in Connection Hub.
- **No token in the model.** The credential is never in the prompt, the context,
  the tool arguments, or generated code - prompt injection has nothing to leak.
  The agent sees the *result* of the call, never the key that made it.
- **Per call, at the trusted boundary.** Resolution and refresh happen only at
  the broker call, never at connect time and never cached on your side.
- **Claim- and binding-scoped.** The broker resolves the exact claim the gate
  approved, on the account the binding names - never a broader token than the
  call earned.

## 7. Adding a new service (Google Sheets, or your own)

"I want to support a new service and get its token from code" is these five
steps - each string you pass in section 2 becomes a config key here:

1. **Enable the external service APIs** in the provider project/account that
   owns the OAuth client. OAuth connection success does not prove that Gmail,
   Sheets, Drive, or another product API is enabled.
2. **Register the provider, connector app, and claims** in Connection Hub
   (`connections.delegated_to_kdcube.providers.<provider>`): the OAuth client
   (`client_id`, `client_secret_ref`), the connector app's `allowed_claims`,
   and each claim's real `provider_scopes`. For a Google service like Sheets you
   reuse the `google` provider and add claims such as `sheets:read`. The built-in
   productivity door deliberately reuses the stable `gmail` connector-app id,
   so an existing Google account can serve both Gmail and Sheets without
   changing stored account identity.
   Worked references:
   - [Google Services (Gmail, Sheets)](google-service-README.md) - the Google
     example end to end (operator setup, provider claims, per-service wiring,
     consent, and verification); Sheets reuses the same `google` provider.
   - [Custom OAuth/OIDC Service Integration](custom-oauth-oidc-service-README.md) -
     a brand-new provider (your own S1), including resolving its credential in
     tool code.
   - [Slack Integration](slack-README.md) - a single-provider realm with its own
     claim set and connector app.
3. **Point your door at the connector app** - add it to
   `surfaces.as_provider.mcp.<door>.connector_apps` (or the named-service
   `connector_apps`), e.g. `google: gmail` (or a dedicated `google: sheets`).
   This is the one line `resolve_connector_app_id` reads.
4. **Write the tool** - resolve the credential exactly as in section 2
   (`resolve_connected_account_claim` + `resolve_connector_app_id`) with the new
   claim, then call the provider API with `credential.access_token`.
5. **Declare the tool's need** - the `connected_accounts` policy
   `{provider_id, claims}` so the gates, consent surfaces, and demand ordering
   know what to ask for. The productivity door's Sheets tools are a complete
   reference implementation of this shape.

The user connects the account once ([Delegated to KDCube], per provider and
connector app); from then on your tool gets a live token per call, at the
trusted boundary, with no secret in your code.

## 8. Expose it over MCP (optional)

A tool that resolves this way is already governed. To let a generic external
agent reach it, expose the namespace as a named service or a plain MCP door -
the same resolution runs behind the door, and the same gate-2 envelopes reach
the MCP client. See
[Expose a Governed Service over MCP](../../apps/expose-mcp-service-README.md)
and the named-service integration recipes
([Mail](mail-named-service-README.md), [Slack](slack-README.md)).
