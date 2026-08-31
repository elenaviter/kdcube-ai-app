---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/integrations/google/google-README.md
title: "Google SDK Integration"
summary: "SDK mechanics for Google connected accounts: the google.oauth adapter, connected-account credential resolution, scope machinery, and the Gmail, Sheets, and Docs provider execution paths."
tags: ["sdk", "integrations", "google", "gmail", "sheets", "docs", "oauth", "provider-scopes"]
keywords: ["google oauth adapter", "google.oauth", "provider scopes", "readonly collapse", "drive.file", "gspread create", "google docs proxy", "connected account credential"]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/integrations/google/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/google-service-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/resolve-connected-credential-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/integrations/provider-error-contract-README.md
  - https://github.com/elenaviter/app-ecosystem/blob/main/docs/connection-hub/frontend/application/integrations/google.md
  - repo:app-ecosystem/products/connection-hub/packages/connection-hub/src/connection_hub/delegated_to_kdcube/providers/google.py
  - repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/integrations/google/sheets_proxy.py
  - repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/integrations/google/docs_proxy.py
  - repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/integrations/google/gmail_tools.py
---

# Google SDK Integration

This doc covers the reusable SDK mechanics behind every Google connected
account: the `google.oauth` adapter, how tool code resolves a Google credential
at the trusted boundary, and the scope machinery that turns KDCube claims into
the exact Google OAuth scopes a token is issued with.

It is provider mechanics, not deployment steps. For the end-to-end recipe
(provider claims, per-service wiring, connect, grant, verify) read
[Google Services Through KDCube](../../../recipes/connections/integrations/google-service-README.md).
For the external Google Cloud setup (project, OAuth client, redirect URIs, and
per-service API enablement) read the bundle-local
[Connection Hub Google setup](https://github.com/elenaviter/app-ecosystem/blob/main/docs/connection-hub/frontend/application/integrations/google.md).

## The `google.oauth` adapter

A `google` provider row selects `adapter: google.oauth`. The adapter lives at:

```text
connection_hub.delegated_to_kdcube.providers.google
  (repo:app-ecosystem/products/connection-hub/packages/connection-hub/src/connection_hub/delegated_to_kdcube/providers/google.py)
```

It owns the Google-specific OAuth mechanics that the generic OAuth/OIDC adapter
does not cover:

- building the Google authorize URL with `access_type=offline`,
  `prompt=consent`, and `include_granted_scopes=true`, so Google can return a
  refresh token for offline automation;
- reconciling the requested scope union at connect time (see the scope machinery
  below) before Google issues the token;
- exchanging the OAuth code and refreshing the access token;
- mapping the Google profile/userinfo into KDCube connected-account metadata.

One Google OAuth client, registered once as the `gmail` connector app on the
`google` provider, backs every Google service. Adding Sheets, Drive, Calendar,
or Docs adds claims and scopes on that same connector app - no new adapter, OAuth
client, or code.

## Resolving a Google credential in tool code

Google tools never receive or store a Google token. They resolve one, per call,
at the trusted boundary. This is the shape `GmailTools._credential` and the
Sheets tools use:

```python
from kdcube_ai_app.apps.chat.sdk.integrations.connected_accounts import (
    resolve_connected_account_claim,
    connected_account_auth_failure,
    run_with_connected_account_retry,
)
from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.connector_app_resolution import (
    resolve_connector_app_id,
)

credential = await resolve_connected_account_claim(
    source,
    provider_id="google",
    connector_app_id=resolve_connector_app_id("google"),  # resolved, never hardcoded
    claim="gmail:read",                                    # or sheets:read / sheets:write
    tool_name="mail.read_message",
    account_id=account_id,
)
```

The broker resolves and refreshes the stored connected credential only here, for
this one call. The general mechanism - the config trace, the gate-2 consent
envelope, and the refresh/retry contract - is covered once in
[Resolve a Connected Credential in Tool Code](../../../recipes/connections/integrations/resolve-connected-credential-README.md).
The provider call itself must preserve the actual failure category per the
[Provider Error And Observability Contract](../provider-error-contract-README.md).

## Scope machinery

A KDCube claim is not a Google scope. Each claim declares its minimal
`provider_scopes` under `providers.google.claims.<claim>`, and a connect requests
the union of the scopes for every claim the user is connecting.

```text
claims connected (e.g. sheets:read + sheets:write)
  -> union of their provider_scopes
  -> read-write-supersedes-read-only collapse
  -> the exact scope set Google issues the token for
```

### Read-write scopes supersede read-only ones

Google treats a scope and its read-only sibling as DISTINCT scopes:
`.../spreadsheets` grants read AND write; `.../spreadsheets.readonly` grants read
only. The read-write scope already covers reads. If a single consent requests
BOTH - exactly what connecting `sheets:read` and `sheets:write` together does -
Google grants the read-only scope and drops the read-write one. The stored token
can read but not write, so later writes fail with Google's
`Request had insufficient authentication scopes`, and the account flips to
`reconnect_required` even though it shows `sheets:write` as granted.

The `google.oauth` adapter (`.../delegated_to_kdcube/providers/google.py`)
reconciles the requested union at connect time: it drops any `<X>.readonly` whose
read-write base `<X>` is also in the request. So connecting `sheets:read` +
`sheets:write` sends Google `spreadsheets` alone - which grants read and write -
and both claims work. A `sheets:read`-only connect still requests
`spreadsheets.readonly`, so least privilege is preserved.

The rule keys on the exact `<X>` / `<X>.readonly` pair, so scopes that are not a
clean read-only/read-write pair are left alone:

- `gmail.readonly` + `gmail.send` keeps both - `gmail.send` does not cover reads,
  so it is not the read-only sibling of `gmail.readonly`. Gmail is therefore
  unaffected by the collapse.
- a per-file `drive.file` alongside `drive` is untouched.

An account connected before this reconciliation reports `reconnect_required` on
the first write; reconnect it and approve the edit scope.

### Creating a spreadsheet is a Drive write

`sheets:write` maps to `spreadsheets` plus `drive.file`. The Drive scope is not
optional. Creating a spreadsheet creates a Drive FILE: gspread's `create()`
(the async proxy lives at
`kdcube_ai_app.apps.chat.sdk.integrations.google.sheets_proxy`) posts to the
Drive API, not the Sheets API. So a Drive WRITE scope is required.

`drive.file` is the least-privilege choice: the app may create and manage only
the files it makes. Without it, `create` fails with
`Request had insufficient authentication scopes` even though `spreadsheets`
(read/write) is granted. Spreadsheet search and file metadata likewise go
through Drive, which is why `sheets:read` maps to
`spreadsheets.readonly` + `drive.metadata.readonly`.

## The same pattern for other Google services

Every Google service connects the same way: enable its API in Google Cloud, and
add a claim under `providers.google.claims` mapping to the real Google scopes.
Scopes are managed as connector claims in `bundles.yaml` (Connection Hub), not
on the console consent screen — while the OAuth app is in *Testing*, the
descriptor drives the authorization request and a test user grants the scopes at
connect time. The `google.oauth` adapter, connector app, and
credential-resolution path are unchanged, and the read-write-supersedes-read-only
collapse is generic across services.

| Service | Read claim -> scope | Read-write claim -> scope |
| --- | --- | --- |
| Sheets | `sheets:read` -> `spreadsheets.readonly` | `sheets:write` -> `spreadsheets` (+ `drive.file` to create) |
| Drive | `drive:read` -> `drive.readonly` | `drive:write` -> `drive` (or `drive.file`, per file) |
| Calendar | `calendar:read` -> `calendar.readonly` | `calendar:write` -> `calendar` |
| Docs | `docs:read` -> `documents.readonly` | `docs:write` -> `documents` |

Connecting a read + write pair for ANY of these sends only the read-write scope
(Google grants read and write), because the adapter drops `<X>.readonly` when
`<X>` is present.

Runtime shape varies by service. Docs runs async in-proc over raw REST (the Docs
API plus the Drive API via `httpx`, in `.../integrations/google/docs_proxy.py`),
with no venv/`gspread` subprocess — unlike Sheets, whose `gspread` proxy runs in
an app-owned `@venv`. The Docs proxy owns exact and logical-title-first Drive
discovery, Shared Drive-compatible listing, provider-native copy for Google
Docs, DOCX/ODT/RTF copy-and-convert into native Docs, tab/table-aware text
extraction, typed edits, comments, import, and export. Import sources are
reported separately from native documents, and their original bytes remain
unchanged during conversion. A native read reports every tab and its stable id.
On a multi-tab document, typed edits require an explicit tab scope; the named
service can resolve that scope from a title, literal title fragment, position,
or hierarchy. It also resolves document-level comments from bounded lexical
predicates such as text, author, and resolved state. Ambiguity returns
candidates before any provider write is sent. Stable Drive comments remain
document-scoped, and tab-scoped comment requests receive a precise capability
error. The operator-facing
configuration and complete agent scenario catalog are documented in
[Google Services Through KDCube](../../../recipes/connections/integrations/google-service-README.md).
