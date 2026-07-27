---
id: repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/google-sheets-README.md
title: "Google Sheets Through KDCube MCP"
summary: "Configure, grant, call, and verify Google Sheets through either typed productivity tools or the provider-neutral sheets named-service namespace."
status: active
tags: ["recipes", "connections", "connection-hub", "google", "sheets", "mcp", "connected-accounts", "delegated-access"]
updated_at: 2026-07-27
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/google-gmail-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/resolve-connected-credential-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/protect-bundle-mcp-with-managed-credentials-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/apps/named-services-mcp-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/bundle-venv-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/react-object-materialization-README.md
  - repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/kdcube-services@1-0/interface/README.md
---
# Google Sheets Through KDCube MCP

Use this recipe when an agent must find and operate on the signed-in user's
Google Sheets. The built-in implementation lives in the `kdcube-services@1-0`
app (bundle) and has two deliberate MCP doors:

```text
public/mcp/productivity
  explicit productivity_sheets_* tools

public/mcp/named_services
  generic named_services_* tools with namespace=sheets
```

Use the productivity door when the client benefits from explicit Google Sheets
tool names and typed parameters. Use the named-services door when an agent
already works through KDCube's common object ontology. Both paths call the same
bounded async app service. The agent never receives a Google access or refresh
token.

## What runs where

```text
agent or MCP client
  -> managed productivity or named-services MCP resource
     gate 1: may this caller use this selected tool?
  -> sheets named-service adapter, when using namespace=sheets
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
4. Add the scopes to the OAuth consent screen: `spreadsheets`,
   `spreadsheets.readonly`, `drive.metadata.readonly`, and — required so the app
   can CREATE spreadsheets (a Drive file operation) — `drive.file`. If the app
   is in testing mode, include every test user.

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
      # Creating a spreadsheet creates a Drive FILE (gspread's create() posts to
      # the Drive API), so a Drive WRITE scope is required. drive.file is the
      # least-privilege one: the app may create and manage only the files it
      # makes. Without it, create fails with "Request had insufficient
      # authentication scopes" even though spreadsheets (read/write) is granted.
      - https://www.googleapis.com/auth/drive.file
```

The existing client secret stays in `bundles.secrets.yaml`. This feature adds
no environment variable and no new secret key.

### Read-write scopes supersede read-only ones (why writes can fail)

Google treats a scope and its read-only sibling as DISTINCT scopes:
`.../spreadsheets` grants read AND write; `.../spreadsheets.readonly` grants read
only. The read-write scope already covers reads. If a single consent requests
BOTH — exactly what connecting `sheets:read` and `sheets:write` together does —
Google grants the read-only scope and drops the read-write one. The stored token
can read but not write, so later writes fail with Google's
`Request had insufficient authentication scopes`, and the account flips to
`reconnect_required` even though it shows `sheets:write` as granted.

You still declare each claim's minimal scope as above. The Google adapter
(`delegated_to_kdcube/providers/google.py`) reconciles the requested union at
connect time: it drops any `<X>.readonly` whose read-write base `<X>` is also in
the request. So connecting `sheets:read` + `sheets:write` sends Google
`spreadsheets` alone — which grants read and write — and both claims work. A
`sheets:read`-only connect still requests `spreadsheets.readonly`, so least
privilege is preserved. An account connected before this reconciliation reports
`reconnect_required` on the first write; reconnect it and approve the edit scope.

## 3. Configure the MCP surfaces

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
        named_services:
          auth:
            mode: managed
            authority_id: delegated_client
            selected_tool_grants: true
```

This declaration is what `resolve_connector_app_id("google")` reads. Tool code
does not hard-code the OAuth client. The named-services resource catalog below
also declares `connector_apps.google: gmail`, so relayed calls resolve the same
provider application.

## 4. Publish capabilities and tools in Connection Hub

Under:

```text
config.connections.delegated_credentials.oauth
```

add delegable `sheets:read` and `sheets:write` capabilities.

For the typed productivity door, add the `productivity_sheets_*` tools to this
resource:

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

For the generic named-services door, add the `sheets` namespace to its resource:

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

The reference descriptor also configures the generic `call` wrapper with the
same exact operation variants. The bridge dispatches an action to provider
`object.action`, but authorizes `object.action.<action>` so granting one Sheets
mutation does not grant every mutation.

These are caller grants under **Delegated by KDCube**. The separate connected
Google account uses `sheets:read` for reads and both `sheets:read` and
`sheets:write` for mutations, matching the typed productivity surface.

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

The typed productivity chain is:

```text
productivity_sheets_search
  -> productivity_sheets_describe
  -> productivity_sheets_read
  -> one explicit mutation tool when needed
```

The equivalent named-services chain is:

```text
named_services_search namespace=sheets
  -> sheets:google:<account_id>:spreadsheet:<spreadsheet_id>
named_services_get namespace=sheets object_ref=<spreadsheet ref>
  -> metadata, stable tab refs, and a signed full-snapshot URL on external MCP
named_services_get namespace=sheets object_ref=<spreadsheet ref>
  filters_json='{"ranges":["Plan!A1:D20"]}'
  -> selected values inline
named_services_upsert / named_services_action
  -> one explicit bounded change
```

Apps (bundles) can expose the same namespace to a hosted agent through an
`as_consumer` named-service tool connection. Add `sheets` to that agent's
`namespaces` map and allow only the operations it needs. The native
`named_services.get_object` tool accepts provider `filters`, so a hosted agent
reads ranges with `filters='{"ranges":["Plan!A1:D20"]}'`; MCP clients use the
equivalent `filters_json` argument shown above.

Spreadsheet refs and tab refs are provider-neutral at the agent boundary:

```text
sheets:<provider>:<account_id>:spreadsheet:<spreadsheet_id>
sheets:<provider>:<account_id>:spreadsheet:<spreadsheet_id>:tab:<sheet_id>
```

Google is the first backing provider. A provider name in a ref is routing data,
not a provider token.

On the turnless named-services MCP surface, `object.get` also carries
`ret.object.snapshot.download.url` when signed file delivery is configured,
including when selected A1 values are returned inline. Fetch that short-lived
URL to receive the complete authorized
`kdcube.sheets.snapshot.v1` JSON artifact outside the model response. The URL
is bound to the exact ref, user, tenant, and project. Google credentials stay
server-side and are resolved from the user's current connected account when
the URL is used. KDCube verifies current consent, fetches the current
spreadsheet, and streams it; the URL is a live delivery proxy rather than a
pre-hosted immutable artifact. The caller's delegated Sheets grant is checked
when the URL is minted; the URL itself is a short-lived bearer capability.
Revoking the connected Google claim blocks an existing URL, while revoking
only the caller grant blocks minting a fresh URL. Explicit A1 ranges are an
alternative for clients that want selected values inline.

### Make `sheets:` refs usable in the ReAct workspace

Exposing named-service tools lets the agent find and operate on Sheets. Add a
separate event-source declaration when the same refs should also work with
`react.pull` and owner-shaped `react.read` rendering:

This section applies only to an app using the KDCube ReAct harness. An external
MCP client such as Claude Code uses the named-service search/get/schema and
mutation tools, then uses its own local processing tools. MCP does not expose
`react.pull`, `react.read`, or `react.rg`.

```yaml
surfaces:
  as_consumer:
    agents:
      main:
        event_sources:
          - kind: named_service
            namespace: sheets
            enabled: true
            discovery:
              mode: service_discovery
            policies:
              block_production:
                mode: provider
                operation: block.produce
              pull:
                mode: provider
                operation: object.get
```

The flow is then:

```text
named_services.search_objects(namespace="sheets", ...)
  -> sheets:google:<account_id>:spreadsheet:<spreadsheet_id>

react.pull(paths=[<returned sheets ref>])
  -> conv:fi:.../<spreadsheet_id>.sheets.json

react.read(paths=[<returned conv:fi ref>])
  -> compact [SHEETS SNAPSHOT] inventory

react.rg(root=<returned conv:fi ref>, pattern=...)
  -> exact read_items for matching JSON regions

react.read(items=[<returned read_item>])
  -> exact line-numbered JSON chunk
```

The snapshot schema is `kdcube.sheets.snapshot.v1`. It includes canonical
identity, metadata, and used values for the selected grid tabs. KDCube does not
apply a range-count or returned-cell ceiling to reads. If Google cannot serve a
large selection in one provider request, the caller can retrieve explicit A1
ranges in separate calls; KDCube reports that provider failure rather than
silently truncating values.

A whole-file `react.read` does not inject all cell values into the model
context. The Sheets provider's `block.produce` operation renders workbook
identity, tabs, dimensions, materialized ranges and counts, completeness, and
the logical/physical snapshot paths. Code can process the complete local JSON
through `physical_path`. When exact JSON text is needed, use `react.rg` and pass
its `read_items` to `react.read`, or request manual line/symbol ranges. Ranged
reads intentionally bypass the compact projection and remain exact chunks.

On the productivity door, search accepts a title fragment; a blank query lists
recently modified spreadsheets. Every later typed tool accepts either the
returned `spreadsheet_id` or a full Google Sheets URL. `describe` returns stable
`sheet_id` values for tab and formatting operations. On the named-services
door, use the complete spreadsheet/tab ref returned by the preceding call.

Search returns at most 50 results. Mutations are deliberately bounded: at most
20 ranges and 10,000 cells per write, 1,000 appended rows, and 1,000,000 cells
in a new or resized tab. Read responses preserve every value returned by the
provider; they are not silently truncated.

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
8. Repeat search/read and one disposable update through `namespace=sheets` on
   the named-services endpoint.
9. From an external MCP client, fetch `ret.object.snapshot.download.url` and verify
   the complete JSON snapshot contains the expected tabs and values.
10. In Workspace, pull the returned `sheets:` ref and read its `conv:fi:` path.
   Verify the model sees a compact `[SHEETS SNAPSHOT]` inventory without cell
   values.
11. Run `react.rg` against the same path and read one returned range. Verify the
    exact JSON chunk, including values in that range, becomes visible.
12. Use code with the returned `physical_path` to inspect workbook values.
13. Re-test Gmail and Slack through the same productivity MCP endpoint.

Provider sharing/permission administration, deleting the spreadsheet file,
Apps Script, comments, and raw `batchUpdate` payloads are intentionally outside
this surface. They require separate claims and review.

## Other Google services, the same way

Sheets reuses the existing `google` provider and `gmail` connector app — it adds
only new claims and scopes, no new OAuth client, adapter, or code. Any other
Google API connects the same way: enable the API in Google Cloud, add its scopes
to the OAuth consent screen, and add a claim under `providers.google.claims`
mapping to the real Google scopes (section 2), then add its tools/named-service
as sections 3-4 do for Sheets. The connect, grant, and two-gate machinery are
unchanged.

| Service | Read claim → scope | Read-write claim → scope |
| --- | --- | --- |
| Sheets | `sheets:read` → `spreadsheets.readonly` | `sheets:write` → `spreadsheets` |
| Drive | `drive:read` → `drive.readonly` | `drive:write` → `drive` (or `drive.file`, per file) |
| Calendar | `calendar:read` → `calendar.readonly` | `calendar:write` → `calendar` |
| Docs | `docs:read` → `documents.readonly` | `docs:write` → `documents` |

The read-write-supersedes-read-only reconciliation above is generic: connecting a
read + write pair for ANY of these sends only the read-write scope (Google grants
read and write), because the adapter drops `<X>.readonly` when `<X>` is present.
It keys on the exact `<X>` / `<X>.readonly` pair, so scopes that are not a clean
read-only/read-write pair are left alone — `gmail.readonly` + `gmail.send` keeps
both (send does not cover read; see the [Gmail recipe](google-gmail-README.md)),
and a per-file `drive.file` alongside `drive` is untouched.
