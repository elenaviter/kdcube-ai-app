---
id: repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/google-sheets-README.md
title: "Google Sheets Through the Productivity MCP Door"
summary: "Configure, grant, call, and verify the built-in governed Google Sheets tools without exposing the user's Google credential to an agent."
status: active
tags: ["recipes", "connections", "connection-hub", "google", "sheets", "mcp", "connected-accounts", "delegated-access"]
updated_at: 2026-07-27
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/google-gmail-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/resolve-connected-credential-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/protect-bundle-mcp-with-managed-credentials-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/bundle-venv-README.md
  - repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/kdcube-services@1-0/interface/README.md
---
# Google Sheets Through the Productivity MCP Door

Use this recipe when an agent must find and operate on the signed-in user's
Google Sheets. The built-in implementation lives in the `kdcube-services@1-0`
app (bundle) and is exposed through:

```text
/api/integrations/bundles/{tenant}/{project}/kdcube-services@1-0/public/mcp/productivity
```

The agent gets typed tools and structured results. It never receives a Google
access or refresh token.

## What runs where

```text
agent or MCP client
  -> managed productivity MCP resource
     gate 1: may this caller use this selected tool?
  -> connected-account resolver
     gate 2: may KDCube use this user's Google account for this claim?
  -> kdcube-services async productivity service
  -> app-owned @venv subprocess (gspread)
  -> Google Sheets API / Drive metadata API
```

`Delegated by KDCube` owns gate 1. `Delegated to KDCube` owns gate 2. Revoking
either one blocks the next call.

## 1. Configure Google Cloud

In the Google Cloud project used by the existing Google connector:

1. Enable **Google Sheets API**.
2. Enable **Google Drive API**. Spreadsheet search uses Drive file metadata.
3. Keep the existing Web OAuth client and its exact KDCube callback URI. The
   [Gmail recipe](google-gmail-README.md#google-cloud-configuration) lists the
   callback shapes for local, staging, and demo runtimes.
4. Add the Sheets scopes to the OAuth consent configuration. If the app is in
   testing mode, include every test user.

No new OAuth client or secret is required. The built-in surface reuses the
existing `google` provider and stable `gmail` connector-app id.

## 2. Add provider claims in `bundles.yaml`

Edit the `connection-hub@1-0` item at:

```text
config.connections.delegated_to_kdcube.providers.google
```

Allow the claims on the existing connector app and define their provider
scopes:

```yaml
connector_apps:
  gmail:
    allowed_claims:
      - gmail:read
      - gmail:send
      - sheets:read
      - sheets:write
claims:
  sheets:read:
    label: Read Google Sheets
    description: Find spreadsheets and read their metadata and values.
    provider_scopes:
      - openid
      - email
      - profile
      - https://www.googleapis.com/auth/spreadsheets.readonly
      - https://www.googleapis.com/auth/drive.metadata.readonly
  sheets:write:
    label: Edit Google Sheets
    description: Create and edit spreadsheets, tabs, values, and formatting.
    provider_scopes:
      - openid
      - email
      - profile
      - https://www.googleapis.com/auth/spreadsheets
      - https://www.googleapis.com/auth/drive.metadata.readonly
```

The existing client secret stays in `bundles.secrets.yaml`. This feature adds
no environment variable and no new secret key.

## 3. Configure the productivity surface

In the `kdcube-services@1-0` item, declare the managed surface and select the
existing connector app:

```yaml
config:
  surfaces:
    as_provider:
      mcp:
        productivity:
          auth:
            mode: managed
            authority_id: delegated_client
            selected_tool_grants: true
          connector_apps:
            google: gmail
            slack: slack-demo
```

This declaration is what `resolve_connector_app_id("google")` reads. Tool code
does not hard-code the OAuth client.

## 4. Publish capabilities and tools in Connection Hub

Under:

```text
config.connections.delegated_credentials.oauth
```

add delegable `sheets:read` and `sheets:write` capabilities. Then add the
`productivity_sheets_*` tools to this resource:

```yaml
resources:
  - resource: "*/api/integrations/bundles/*/*/kdcube-services@1-0/public/mcp/productivity*"
    label: KDCube productivity MCP
    tools:
      productivity_sheets_search:
        grants: [sheets:read]
      productivity_sheets_describe:
        grants: [sheets:read]
      productivity_sheets_read:
        grants: [sheets:read]
      productivity_sheets_update_values:
        grants: [sheets:write]
      productivity_sheets_append_rows:
        grants: [sheets:write]
      productivity_sheets_clear_values:
        grants: [sheets:write]
      productivity_sheets_create_spreadsheet:
        grants: [sheets:write]
      productivity_sheets_add_tab:
        grants: [sheets:write]
      productivity_sheets_update_tab:
        grants: [sheets:write]
      productivity_sheets_delete_tab:
        grants: [sheets:write]
      productivity_sheets_format_range:
        grants: [sheets:write]
```

Give each capability the roles and permissions that your deployment allows to
delegate. The reference descriptors allow registered, paid, privileged, and
super-admin users to delegate their own Sheets access.

## 5. Refresh the app

The optional dependency belongs to `kdcube-services@1-0/requirements.txt`:

```text
gspread==6.2.1
```

Refresh or reload the app after code/config changes. The first Sheets call
creates the app's cached venv; later calls reuse it until `requirements.txt`
changes. The async decorated helper keeps the shared proc event loop free while
the child process runs the synchronous provider library.

## 6. Connect and grant

1. Open **Connection Hub -> Delegated to KDCube**.
2. Connect the Google account. An account connected earlier for Gmail receives
   `claim_upgrade_required` when Sheets access is first requested; approve it.
3. Open **Connection Hub -> Delegated by KDCube** for the agent or external MCP
   client.
4. Select only the required Sheets tools and account claims, then save.

The caller grant never contains the Google token. It binds the caller, selected
resource/tools, KDCube grants, and the user's approved connected account.

## 7. Let the agent work

The normal chain is:

```text
productivity_sheets_search
  -> productivity_sheets_describe
  -> productivity_sheets_read
  -> one explicit mutation tool when needed
```

Search accepts a title fragment; a blank query lists recently modified
spreadsheets. Every later tool accepts either the returned `spreadsheet_id` or
a full Google Sheets URL. `describe` returns stable `sheet_id` values for tab
and formatting operations.

Calls are deliberately bounded: at most 50 search results, 20 ranges, 20,000
read cells, 10,000 written cells, 1,000 appended rows, and 1,000,000 cells in a
new/resized tab. Oversized requests fail explicitly instead of truncating.

`append_rows` and `create_spreadsheet` are not exactly-once operations. If a
transport failure reports `outcome_unknown`, inspect/search before retrying.
The optional `idempotency_key` is returned as correlation data; it does not
turn Google into an exactly-once provider.

## 8. Verify the complete boundary

Use a disposable spreadsheet and verify:

1. `search`, `describe`, and `read` work with a read-only grant.
2. A write tool is denied until both the selected-tool grant and
   `sheets:write` connected-account claim exist.
3. Update values, append one row, format one range, add/update/delete a test
   tab, and create a test spreadsheet.
4. With two Google accounts, an ambiguous call returns `account_required`; the
   retry succeeds with one returned `account_id`.
5. Revoke the caller's tool grant and confirm the next call stops at gate 1.
6. Restore it, revoke the Google claim, and confirm the next call stops at gate
   2.
7. Inspect tool output, timeline, logs, and model input: no Google bearer or
   refresh token may appear.
8. Re-test Gmail and Slack through the same productivity MCP endpoint.

Provider sharing/permission administration, deleting the spreadsheet file,
Apps Script, comments, and raw `batchUpdate` payloads are intentionally outside
this surface. They require separate claims and review.
