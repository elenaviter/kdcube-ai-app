---
id: repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/google-service-README.md
title: "Google Services Through KDCube (Gmail, Sheets, Docs)"
summary: "One recipe for connecting Google services to KDCube: one Google OAuth client, one google provider, one gmail connector app serving Gmail, Sheets, and Docs (extensible to Drive/Calendar). Configure provider claims, wire each service's tools and named services, connect, grant, and verify."
status: active
tags: ["recipes", "connections", "connection-hub", "google", "gmail", "sheets", "docs", "oauth", "connected-accounts", "delegated-to-kdcube", "mcp"]
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
# Google Services Through KDCube (Gmail, Sheets, Docs)

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
Gmail, Sheets, Docs, and any Drive/Calendar service added the same way. Each
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
  Sheets API plus Drive API for spreadsheets; Docs API plus Drive API for
  documents);
- the client id/secret keys and the hub-level `oauth_state_secret`.

A completed Google OAuth connection proves identity and consent worked; it does
not prove a product API is enabled. Enable each service's API in the same Google
Cloud project that owns the OAuth client.

## Configure provider claims

Under the `connection-hub@1-0` item at
`config.connections.delegated_to_kdcube.providers.google`, allow every claim on
the one `gmail` connector app and define each claim's provider scopes. Add only
the services you use; the block below shows Gmail, Sheets, and Docs together:

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
      - docs:read
      - docs:write
      - docs:comment
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
  docs:read:
    label: Read Google Docs
    description: Find documents, read their text and structure, export to a
      format, and read comments.
    provider_scopes:
      - openid
      - email
      - profile
      - https://www.googleapis.com/auth/documents.readonly
      # Docs read uses drive.readonly, not sheets' drive.metadata.readonly:
      # get/export stream the document's Drive content (full text, exported
      # bytes), not only file metadata, and search lists Docs files by content.
      - https://www.googleapis.com/auth/drive.readonly
  docs:write:
    label: Edit Google Docs
    description: Create documents and apply typed edits - insert/append/replace
      text, text styling, page breaks, embedded images, and import.
    provider_scopes:
      - openid
      - email
      - profile
      - https://www.googleapis.com/auth/documents
      # create posts to the Drive API and import uploads a Drive FILE, so a
      # Drive WRITE scope is required. drive.file is least-privilege: the app
      # may create and manage only the files it makes - same rule as Sheets.
      - https://www.googleapis.com/auth/drive.file
  docs:comment:
    label: Comment on Google Docs
    description: List, create, reply to, resolve, and delete comments on a
      document through the Drive comments API.
    provider_scopes:
      - openid
      - email
      - profile
      # Comments (and export) are Drive operations that act on ANY document the
      # user names, including ones this app did not create, so drive.file (which
      # covers only app-created files) is insufficient here - the full drive
      # scope is required.
      - https://www.googleapis.com/auth/drive
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

### Google Docs

Docs has the same two MCP doors on `kdcube-services@1-0` as Sheets: the typed
`productivity_docs_*` tools (`public/mcp/productivity`) and the generic `docs`
named-service namespace (`public/mcp/named_services`). Both call the same bounded
async app service; the agent never receives a Google token. Docs differs from
Sheets in one mechanical way: the proxy speaks raw REST to the Docs API and the
Drive API over async `httpx`, so it runs in-proc with no `@venv`/`gspread`
subprocess. No new connector app, adapter, or requirement is added.

The same managed `productivity` and `named_services` surfaces already declared
for Sheets serve Docs; no additional surface wiring is needed. Publish the
delegable Docs capabilities and tools in Connection Hub under
`config.connections.delegated_credentials.oauth`.

Docs splits into three connected-account claims (Sheets had two). `docs:read`
covers find, read, export, and reading comments; `docs:write` covers document
creation and typed edits; `docs:comment` covers comment mutations. For the typed
productivity door, grant each `productivity_docs_*` tool:

```yaml
resources:
  - resource: "*/api/integrations/bundles/*/*/kdcube-services@1-0/public/mcp/productivity*"
    label: KDCube productivity MCP
    tools:
      productivity_docs_search:           {grants: [docs:read]}
      productivity_docs_get:              {grants: [docs:read]}
      productivity_docs_export:           {grants: [docs:read]}
      productivity_docs_list_comments:    {grants: [docs:read]}
      productivity_docs_get_comment:      {grants: [docs:read]}
      productivity_docs_create:           {grants: [docs:write]}
      productivity_docs_insert_text:      {grants: [docs:write]}
      productivity_docs_append_text:      {grants: [docs:write]}
      productivity_docs_replace_text:     {grants: [docs:write]}
      productivity_docs_apply_text_style: {grants: [docs:write]}
      productivity_docs_insert_page_break:{grants: [docs:write]}
      productivity_docs_embed_image:      {grants: [docs:write]}
      productivity_docs_import:           {grants: [docs:write]}
      productivity_docs_create_comment:   {grants: [docs:comment]}
      productivity_docs_reply_comment:    {grants: [docs:comment]}
      productivity_docs_resolve_comment:  {grants: [docs:comment]}
      productivity_docs_delete_comment:   {grants: [docs:comment]}
```

For the generic named-services door, add the `docs` namespace alongside `sheets`
on its resource. As with Sheets, the bridge authorizes the exact
`object.action.<action>`, so granting one edit does not grant every edit, and the
comment mutations key on `docs:comment` rather than `docs:write`:

```yaml
- resource: "*/api/integrations/bundles/*/*/kdcube-services@1-0/public/mcp/named_services*"
  label: KDCube named services MCP
  named_services:
    connector_apps:
      google: gmail
    namespaces:
      docs:
        label: Documents
        authority_id: delegated_client
        tools:
          about:        {operation: provider.about,        grants: [named_services:use]}
          capabilities: {operation: provider.capabilities, grants: [named_services:use]}
          schema:       {operation: object.schema,         grants: [named_services:use]}
          search:       {operation: object.search,         grants: [named_services:use, docs:read]}
          get:          {operation: object.get,            grants: [named_services:use, docs:read]}
          upsert:       {operation: object.upsert,         grants: [named_services:use, docs:write]}
          action:
            operation: object.action
            operations:
              object.action.insert_text:       {grants: [named_services:use, docs:write]}
              object.action.append_text:       {grants: [named_services:use, docs:write]}
              object.action.replace_text:      {grants: [named_services:use, docs:write]}
              object.action.apply_text_style:  {grants: [named_services:use, docs:write]}
              object.action.insert_page_break: {grants: [named_services:use, docs:write]}
              object.action.embed_image:       {grants: [named_services:use, docs:write]}
              object.action.import:            {grants: [named_services:use, docs:write]}
              object.action.create_comment:    {grants: [named_services:use, docs:comment]}
              object.action.reply_comment:     {grants: [named_services:use, docs:comment]}
              object.action.resolve_comment:   {grants: [named_services:use, docs:comment]}
              object.action.delete_comment:    {grants: [named_services:use, docs:comment]}
```

These are caller grants under **Delegated by KDCube**; the separate connected
Google account holds `docs:read`/`docs:write`/`docs:comment`. The connect/grant
two-gate machinery is identical to Sheets above.

**Connect and grant.** Connect the Google account under **Delegated to
KDCube** (an account connected earlier for Gmail or Sheets returns
`claim_upgrade_required` when Docs access is first requested; approve it), then
select the required Docs tools and account claims under **Delegated by KDCube**.

Document refs are provider-neutral at the agent boundary:

```text
docs:<provider>:<account_id>:document:<document_id>
```

Search returns at most 50 results. Operations are bounded: text reads and
replacements are capped at 200,000 characters, at most 50 replacements per
`replace_text`, comment bodies at 20,000 characters, at most 100 comments listed,
titles at 300 characters, and export/import at 10 MiB. `create` and `import` are
not exactly-once; on an `outcome_unknown` transport failure, search/get before
retrying. Provider failures preserve Google's safe message plus `provider_status`,
`provider_code`, `provider_reason`, `stage`, and `retryable`, per the
[Provider Error And Observability Contract](../../../sdk/integrations/provider-error-contract-README.md).

The SDK mechanics (the async REST proxy over the Docs and Drive APIs, and the
shared credential resolver) are in
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

**Docs.** On a disposable document:

1. `search`, `get`, and `export` work with a read-only grant.
2. A write tool (`insert_text`, `create`, ...) is denied until both the
   selected-tool grant and the `docs:write` connected-account claim exist.
3. Create a document, append/insert/replace text, apply a style, insert a page
   break, embed an image, and import a source document.
4. A comment tool (`create_comment`, `resolve_comment`, ...) is denied until the
   `docs:comment` claim exists - `docs:write` alone does not authorize it.
5. Revoke the caller's tool grant; the next call stops at gate 1. Restore it,
   revoke the Google claim; the next call stops at gate 2.
6. Inspect tool output, timeline, logs, and model input: no Google bearer or
   refresh token appears.
7. Repeat search/get and one disposable edit through `namespace=docs` on the
   named-services endpoint, and fetch the signed snapshot URL to verify the
   complete JSON snapshot.

## Add another Google service, the same way

Any other Google API connects the same way: enable the API in Google Cloud, and
add a claim under `providers.google.claims` mapping to the real Google scopes,
then wire its tools or named service as Sheets does above. Scopes are managed as
connector claims in `bundles.yaml` (Connection Hub) — **not** on the console
consent screen: while the OAuth app is in *Testing*, the descriptor drives the
authorization request and a test user grants the scopes at connect time. The
console is only for enabling the API, the OAuth client, redirect URIs, and test
users. The connect, grant, and two-gate machinery are unchanged.

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
