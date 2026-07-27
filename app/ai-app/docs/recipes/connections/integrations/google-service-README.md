---
id: repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/google-service-README.md
title: "Google Services Through KDCube (Gmail, Sheets)"
summary: "One recipe for connecting Google services to KDCube: one Google OAuth client, one google provider, one gmail connector app serving Gmail and Sheets (extensible to Drive/Calendar/Docs). Configure provider claims, wire each service's tools and named services, connect, grant, and verify."
status: active
tags: ["recipes", "connections", "connection-hub", "google", "gmail", "sheets", "oauth", "connected-accounts", "delegated-to-kdcube", "mcp"]
updated_at: 2026-07-28
see_also:
  - repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/connection-hub@1-0/docs/integrations/google.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/integrations/google/google-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/mail-named-service-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/resolve-connected-credential-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/integrations/provider-error-contract-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/slack-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/apps/named-services-mcp-README.md
---
# Google Services Through KDCube (Gmail, Sheets)

Use this recipe to let a signed-in KDCube user connect their own Google account,
then let KDCube tools and named services act on that user's behalf across Google
services. This is the **delegated to KDCube** direction:

```text
Google user
  -> user consents in Google OAuth
  -> Connection Hub stores the connected account credential
  -> KDCube tool/named service resolves that credential for the current user
  -> tool calls the Google API with the user's delegated Google token
```

**One client serves every Google service.** One Google OAuth client, one
`google` provider (`adapter: google.oauth`), and one `gmail` connector app back
Gmail, Sheets, and any Drive/Calendar/Docs service added the same way. Each
service only adds claims, provider scopes, and tools; it does not add an OAuth
client, adapter, or code.

## Operator setup (external)

The Google Cloud work happens **outside** KDCube and is documented once in the
bundle-local operator doc. Do it there, not here:
[Connection Hub - Google (Gmail and Sheets) setup](../../../../src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/connection-hub@1-0/docs/integrations/google.md).

That doc covers, in one place:

- the Google Cloud project and OAuth Web-application client;
- the delegated-to-KDCube **Authorized redirect URIs** (the callback path ends
  with `.../connection-hub@1-0/public/delegated_to_kdcube_oauth_callback`) for
  the local, custom-authority, demo, and dev runtimes;
- enabling the per-service product APIs in the same project (Gmail API for mail;
  Sheets API plus Drive API for spreadsheets);
- the client id/secret keys and the hub-level `oauth_state_secret`.

A completed Google OAuth connection proves identity and consent worked; it does
not prove a product API is enabled. Enable each service's API in the same Google
Cloud project that owns the OAuth client.

## Configure provider claims

Under the `connection-hub@1-0` item at
`config.connections.delegated_to_kdcube.providers.google`, allow every claim on
the one `gmail` connector app and define each claim's provider scopes. Add only
the services you use; the block below shows Gmail and Sheets together:

```yaml
connector_apps:
  gmail:
    label: Gmail
    enabled: true
    client_id: "<GOOGLE_OAUTH_CLIENT_ID>"
    client_secret_ref: connections.delegated_to_kdcube.providers.google.connector_apps.gmail.client_secret
    allowed_claims:
      - gmail:read
      - gmail:send
      - sheets:read
      - sheets:write
claims:
  gmail:read:
    label: Read Gmail
    description: Search and read Gmail messages for the approving user.
    provider_scopes:
      - openid
      - email
      - profile
      - https://www.googleapis.com/auth/gmail.readonly
  gmail:send:
    label: Send Gmail
    description: Send email through the approving user's Gmail account.
    provider_scopes:
      - openid
      - email
      - profile
      - https://www.googleapis.com/auth/gmail.send
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
      # Creating a spreadsheet creates a Drive FILE (gspread's create() posts to
      # the Drive API), so a Drive WRITE scope is required. drive.file is the
      # least-privilege one: the app may create and manage only the files it
      # makes. Without it, create fails with "Request had insufficient
      # authentication scopes" even though spreadsheets (read/write) is granted.
      - https://www.googleapis.com/auth/drive.file
```

The client secret stays in `bundles.secrets.yaml` at the `client_secret_ref`
above (see the operator doc). Adding Sheets to a deployment that already had
Gmail adds no new secret and no environment variable.

**Read-write scopes supersede read-only ones.** When one connect requests both a
Google scope and its read-only sibling (`spreadsheets` and
`spreadsheets.readonly`), Google grants the read-only one and drops the
read-write one, so later writes fail with
`Request had insufficient authentication scopes`. The `google.oauth` adapter
reconciles this at connect time - it drops any `<X>.readonly` whose read-write
base `<X>` is also requested. You still declare each claim's minimal scope as
above. Gmail is unaffected (`gmail.send` is not the read-only sibling of
`gmail.readonly`). The full scope machinery is in the SDK doc,
[Google SDK Integration](../../../sdk/integrations/google/google-README.md).

## Per-service wiring

Each service adds its own tools (and optionally a named-service namespace) on top
of the shared provider claims above.

### Gmail

Give the main agent the Gmail tool module and declare each tool's connected-account
claims. A tool names the provider and claims it needs, never the connector app -
the broker resolves the account at call time:

```yaml
- name: gmail
  kind: python
  module: kdcube_ai_app.apps.chat.sdk.integrations.google.gmail_tools
  alias: gmail
  allowed:
    - search_gmail
    - read_gmail_message
    - download_gmail_attachments
    - send_gmail
    - forward_gmail_message
  tool_claims:
    search_gmail:
      connections:
        delegated_to_kdcube:
          connected_accounts:
            - provider_id: google
              claims: [gmail:read]
    read_gmail_message:
      connections:
        delegated_to_kdcube:
          connected_accounts:
            - provider_id: google
              claims: [gmail:read]
    download_gmail_attachments:
      connections:
        delegated_to_kdcube:
          connected_accounts:
            - provider_id: google
              claims: [gmail:read]
    send_gmail:
      connections:
        delegated_to_kdcube:
          connected_accounts:
            - provider_id: google
              claims: [gmail:send]
    forward_gmail_message:
      connections:
        delegated_to_kdcube:
          connected_accounts:
            - provider_id: google
              claims: [gmail:read, gmail:send]
```

The same connected Gmail account can also be exposed to external agents through
the provider-neutral `mail` named-service namespace on
`kdcube-services@1-0/public/mcp/named_services`. That adds a second delegated
consent layer (KDCube grants `mail:read`/`mail:send`; the connected account holds
`gmail:read`/`gmail:send`). See
[Mail Named Service Over MCP](mail-named-service-README.md) for the namespace
refs, MCP operations, and Connection Hub boundary config.

### Google Sheets

Sheets has two MCP doors on the built-in `kdcube-services@1-0` app: the typed
`productivity_sheets_*` tools (`public/mcp/productivity`) and the generic
`sheets` named-service namespace (`public/mcp/named_services`). Both call the
same bounded async app service through an app-owned `@venv` subprocess
(`gspread`); the agent never receives a Google token.

Declare the managed surface and select the existing connector app:

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
        named_services:
          auth:
            mode: managed
            authority_id: delegated_client
            selected_tool_grants: true
```

`resolve_connector_app_id("google")` reads this `connector_apps.google: gmail`
declaration; tool code does not hard-code the OAuth client. Add the optional
dependency to `kdcube-services@1-0/requirements.txt` (`gspread==6.2.1`) and
refresh the app so it builds the cached venv.

Publish the delegable capabilities and tools in Connection Hub under
`config.connections.delegated_credentials.oauth`. For the typed productivity
door, grant each `productivity_sheets_*` tool:

```yaml
resources:
  - resource: "*/api/integrations/bundles/*/*/kdcube-services@1-0/public/mcp/productivity*"
    label: KDCube productivity MCP
    tools:
      productivity_sheets_search:            {grants: [sheets:read]}
      productivity_sheets_describe:          {grants: [sheets:read]}
      productivity_sheets_read:              {grants: [sheets:read]}
      productivity_sheets_update_values:     {grants: [sheets:write]}
      productivity_sheets_append_rows:       {grants: [sheets:write]}
      productivity_sheets_clear_values:      {grants: [sheets:write]}
      productivity_sheets_create_spreadsheet:{grants: [sheets:write]}
      productivity_sheets_add_tab:           {grants: [sheets:write]}
      productivity_sheets_update_tab:        {grants: [sheets:write]}
      productivity_sheets_delete_tab:        {grants: [sheets:write]}
      productivity_sheets_format_range:      {grants: [sheets:write]}
```

For the generic named-services door, add the `sheets` namespace to its resource.
The bridge dispatches an action to provider `object.action` but authorizes the
exact `object.action.<action>`, so granting one Sheets mutation does not grant
every mutation:

```yaml
- resource: "*/api/integrations/bundles/*/*/kdcube-services@1-0/public/mcp/named_services*"
  label: KDCube named services MCP
  named_services:
    connector_apps:
      google: gmail
    namespaces:
      sheets:
        label: Spreadsheets
        authority_id: delegated_client
        tools:
          about:        {operation: provider.about,        grants: [named_services:use]}
          capabilities: {operation: provider.capabilities, grants: [named_services:use]}
          list:         {operation: object.list,           grants: [named_services:use, sheets:read]}
          schema:       {operation: object.schema,         grants: [named_services:use]}
          search:       {operation: object.search,         grants: [named_services:use, sheets:read]}
          get:          {operation: object.get,            grants: [named_services:use, sheets:read]}
          upsert:       {operation: object.upsert,         grants: [named_services:use, sheets:write]}
          action:
            operation: object.action
            operations:
              object.action.update_values: {grants: [named_services:use, sheets:write]}
              object.action.append_rows:   {grants: [named_services:use, sheets:write]}
              object.action.clear_values:  {grants: [named_services:use, sheets:write]}
              object.action.add_tab:       {grants: [named_services:use, sheets:write]}
              object.action.update_tab:    {grants: [named_services:use, sheets:write]}
              object.action.delete_tab:    {grants: [named_services:use, sheets:write]}
              object.action.format_range:  {grants: [named_services:use, sheets:write]}
          delete:       {operation: object.delete,         grants: [named_services:use, sheets:write]}
```

These are caller grants under **Delegated by KDCube**. The separate connected
Google account uses `sheets:read` for reads and `sheets:write` for mutations.
Give each capability the roles/permissions your deployment allows to delegate.

**Connect and grant.**

1. Open **Connection Hub -> Delegated to KDCube** and connect the Google account.
   An account connected earlier for Gmail receives `claim_upgrade_required` when
   Sheets access is first requested; approve it.
2. Open **Connection Hub -> Delegated by KDCube** for the agent or external MCP
   client, select only the required Sheets tools and account claims, and save.

The caller grant never contains the Google token. It binds the caller, the
selected resource/tools, the KDCube grants, and the user's approved connected
account.

Spreadsheet and tab refs are provider-neutral at the agent boundary:

```text
sheets:<provider>:<account_id>:spreadsheet:<spreadsheet_id>
sheets:<provider>:<account_id>:spreadsheet:<spreadsheet_id>:tab:<sheet_id>
```

Search returns at most 50 results. Mutations are bounded: at most 20 ranges and
10,000 cells per write, 1,000 appended rows, and 1,000,000 cells in a new or
resized tab. Reads preserve every value the provider returns. `append_rows` and
`create_spreadsheet` are not exactly-once; on an `outcome_unknown` transport
failure, inspect/search before retrying. Provider failures preserve Google's safe
message plus `provider_status`, `provider_code`, `provider_reason`, `stage`, and
`retryable`, per the
[Provider Error And Observability Contract](../../../sdk/integrations/provider-error-contract-README.md).

The SDK mechanics behind these tools (the async gspread proxy, the credential
resolver, and the snapshot artifacts) are in
[Google SDK Integration](../../../sdk/integrations/google/google-README.md).

## Verify

Refresh the runtime after descriptor changes, then walk both services on
disposable data:

**Gmail.** Connect Gmail first, then from an agent that has the Gmail tools:

- with `gmail:read`: `search_gmail` finds messages, `read_gmail_message` reads a
  body and lists attachment ids, `download_gmail_attachments` materializes
  attachments as KDCube files;
- with `gmail:send`: `send_gmail` sends, including KDCube-file attachments;
- with both: `forward_gmail_message` forwards with original attachments;
- if the account lacks a claim, the tool returns a managed connected-account
  consent error the chat UI can surface as a connect/upgrade action.

**Sheets.** On a disposable spreadsheet:

1. `search`, `describe`, and `read` work with a read-only grant.
2. A write tool is denied until both the selected-tool grant and the
   `sheets:write` connected-account claim exist.
3. Update values, append a row, format a range, add/update/delete a test tab,
   and create a test spreadsheet.
4. With two Google accounts, an ambiguous call returns `account_required`; the
   retry succeeds with one returned `account_id`.
5. Revoke the caller's tool grant; the next call stops at gate 1. Restore it,
   revoke the Google claim; the next call stops at gate 2.
6. Inspect tool output, timeline, logs, and model input: no Google bearer or
   refresh token appears.
7. Repeat search/read and one disposable update through `namespace=sheets` on the
   named-services endpoint, and fetch `ret.object.snapshot.download.url` to verify
   the complete JSON snapshot.

## Add another Google service, the same way

Any other Google API connects the same way: enable the API in Google Cloud, add
its scopes to the OAuth consent screen, and add a claim under
`providers.google.claims` mapping to the real Google scopes, then wire its tools
or named service as Sheets does above. The connect, grant, and two-gate machinery
are unchanged.

| Service | Read claim -> scope | Read-write claim -> scope |
| --- | --- | --- |
| Sheets | `sheets:read` -> `spreadsheets.readonly` | `sheets:write` -> `spreadsheets` |
| Drive | `drive:read` -> `drive.readonly` | `drive:write` -> `drive` (or `drive.file`, per file) |
| Calendar | `calendar:read` -> `calendar.readonly` | `calendar:write` -> `calendar` |
| Docs | `docs:read` -> `documents.readonly` | `docs:write` -> `documents` |

The read-write-supersedes-read-only reconciliation is generic: connecting a
read + write pair for ANY of these sends only the read-write scope (Google grants
read and write), because the adapter drops `<X>.readonly` when `<X>` is present.
It keys on the exact `<X>` / `<X>.readonly` pair, so scopes that are not a clean
read-only/read-write pair are left alone.
