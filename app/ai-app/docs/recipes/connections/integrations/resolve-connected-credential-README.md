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
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/google-gmail-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/slack-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/authenticated-mcp/authenticated-mcp-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/delegated-accounts/delegated-accounts-README.md
  - repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/integrations/connected_accounts.py
  - repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/solutions/connections/agent_account_scope.py
---
# Resolve a Connected Credential in Tool Code

Every integration recipe - [Gmail](google-gmail-README.md),
[Slack](slack-README.md), a [custom OAuth/OIDC service](custom-oauth-oidc-service-README.md) -
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

async def read_message(source, message_id: str, account_id: str | None = None):
    async def _run():
        credential = await resolve_connected_account_claim(
            source,                       # the authorized request scope
            provider_id="google",         # the Delegated-to-KDCube provider row
            connector_app_id="gmail",     # the connector app under that provider
            claim="gmail:read",           # the exact provider claim this op needs
            tool_name="mail.read_message",
            account_id=account_id or "",  # empty -> broker picks / asks
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

## 3. The refresh-and-retry contract

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

## 4. What the caller sees

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

## 5. Guarantees

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

## 6. Expose it over MCP (optional)

A tool that resolves this way is already governed. To let a generic external
agent reach it, expose the namespace as a named service or a plain MCP door -
the same resolution runs behind the door, and the same gate-2 envelopes reach
the MCP client. See
[Expose a Governed Service over MCP](../../apps/expose-mcp-service-README.md)
and the named-service integration recipes
([Mail](mail-named-service-README.md), [Slack](slack-README.md)).
