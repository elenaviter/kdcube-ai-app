---
id: kdcube-services@1-0
title: "KDCube Services App"
summary: "Built-in KDCube service surfaces for delegated clients: conversations, named services, and connected-account productivity tools."
status: active
tags: ["app", "bundle", "mcp", "storage", "connection-hub", "delegated-credentials", "conversations"]
module: entrypoint
singleton: false
primary_surfaces:
  - "Widget `bundle_storage` — privileged operational storage browser"
  - "Privileged platform administration widgets — economics, conversations, gateway, Redis, and apps"
  - "MCP endpoint `conversations` — delegated access to conversations_export"
  - "MCP endpoint `named_services` — delegated access to configured namespaces, including provider-neutral spreadsheets"
  - "MCP endpoint `productivity` — governed Slack, mail, and Google Sheets tools over connected accounts"
  - "Signed public file transfer — conversation, mail, Slack, and staged uploads"
links:
  config: config/bundles.template.yaml
  interface: interface/README.md
  openapi: interface/kdcube-services.openapi.yaml
  design: docs/README.md
  storage: docs/storage/README.md
  journal: docs/journal/README.md
---

# KDCube Services App

`kdcube-services@1-0` is the built-in app for KDCube-owned service surfaces that
external clients may access through Connection Hub delegated credentials.

It is intentionally neutral: it is not an "admin bundle" as a whole. Some tools
are admin-only because their descriptor grants are delegable only by admins.
Other future tools can be regular user services.

## Current Services

### Storage Browser

Widget:

```text
/api/integrations/bundles/{tenant}/{project}/kdcube-services@1-0/widgets/bundle_storage
```

`bundle_storage` is the privileged operational storage browser. It is built from
the shared SDK source:

```text
sdk://solutions/storage/ui.widget.storage
```

The widget consumes the platform admin storage APIs:

| API | Purpose |
| --- | --- |
| `/api/admin/control-plane/storage/roots` | Discover bundle storage, managed app folders, and shared storage roots. |
| `/api/admin/control-plane/storage/list` | Browse a selected local filesystem root. |
| `/api/admin/control-plane/storage/export` | Export selected files/directories as a zip. |
| `/api/admin/control-plane/storage/delete` | Delete selected files/directories after confirmation. |
| `/admin/integrations/bundles/storage-registry` | Compare managed app folders with the active registry. |

Cloud deployments must mount the browsed filesystem roots into `chat-ingress`,
because the storage APIs are served by ingress.

### Platform Administration Widgets

The app is the stable home for these privileged platform widgets:

| Alias | Purpose |
| --- | --- |
| `control_plane` | Economics control-plane dashboard. |
| `conversation_browser` | Conversation inspection and operations. |
| `svc_gateway` | Gateway monitoring. |
| `redis_browser` | Redis inspection. |
| `ai_bundles` | App registry and administration. |

All are authenticated, privileged surfaces. They are not public MCP tools.

### Conversations

MCP endpoint:

```text
/api/integrations/bundles/{tenant}/{project}/kdcube-services@1-0/public/mcp/conversations
```

Tool:

| Tool | Grant | Default delegability |
| --- | --- | --- |
| `conversations_export` | `conversations:read` | `kdcube:role:super-admin` |

This is the platform-native replacement for older root `/mcp`
conversation-export shortcuts. The OAuth protocol and consent screen remain
Connection Hub responsibilities; this bundle only owns the protected product
surface.

### Named Services

MCP endpoint:

```text
/api/integrations/bundles/{tenant}/{project}/kdcube-services@1-0/public/mcp/named_services
```

Tools:

| Tool | Outer Grant | Purpose |
| --- | --- | --- |
| `named_services_list` | `named_services:use` | List namespaces exposed by this MCP surface. |
| `named_services_about` | `named_services:use` | Read provider about metadata. |
| `named_services_capabilities` | `named_services:use` | Read provider capabilities for a configured namespace. |
| `named_services_schema` | `named_services:use` | Read object schema metadata. |
| `named_services_search` | `named_services:use` | Search objects in a configured namespace. |
| `named_services_get` | `named_services:use` | Read one object by ref. |
| `named_services_upsert` | `named_services:use` | Create or update one object if the namespace permits `object.upsert`. |
| `named_services_host_file` | `named_services:use` | Host/register a file ref if the namespace permits `object.host_file`. |
| `named_services_action` | `named_services:use` | Run a bounded object action if the namespace permits `object.action`. |
| `named_services_delete` | `named_services:use` | Delete/archive one object if the namespace permits `object.delete`. |
| `named_services_call` | `named_services:use` | Generic named-service operation wrapper. |

Each namespace can require additional grants per operation. Those namespace
boundaries are not configured in this hosting bundle. Connection Hub owns the
resource consent catalog and persists the approved catalog into the delegated
credential grant record:

```yaml
connections:
  delegated_credentials:
    oauth:
      resources:
        - resource: "*/api/integrations/bundles/*/*/kdcube-services@1-0/public/mcp/named_services*"
          tools:
            named_services_schema:
              grants: [named_services:use]
          named_services:
            namespaces:
              mem:
                authority_id: delegated_client
                tools:
                  schema:
                    operation: object.schema
                    grants: [memories:read]
                  search:
                    operation: object.search
                    grants: [memories:read]
                  upsert:
                    operation: object.upsert
                    grants: [memories:write]
                  action:
                    operation: object.action
                    grants: [memories:read]
                  delete:
                    operation: object.delete
                    grants: [memories:write]
                  get:
                    operation: object.get
                    grants: [memories:read]
              task:
                authority_id: delegated_client
                tools:
                  search:
                    operation: object.search
                    grants: [tasks:read]
                  upsert:
                    operation: object.upsert
                    grants: [tasks:write]
                  host_file:
                    operation: object.host_file
                    grants: [tasks:write]
                  delete:
                    operation: object.delete
                    grants: [tasks:write]
              cnv:
                authority_id: delegated_client
                tools:
                  search:
                    operation: object.search
                    grants: [canvas:read]
                  upsert:
                    operation: object.upsert
                    grants: [canvas:write]
```

The outer MCP guard checks the selected MCP tool and `named_services:use`.
The named-services bridge then checks the namespace/operation authority and
grant from the delegated credential grant record before it calls the provider.
If the delegated credential lacks a namespace grant, the tool returns a
structured `delegated_consent_required` payload. Current MCP clients do not
reliably convert that tool result into a new OAuth consent flow, so production
resources should advertise likely namespace grants during initial Connection
Hub consent.

The built-in `sheets` namespace is an ontology adapter over the same Google
Sheets service used by `public/mcp/productivity`:

```text
named_services_search namespace=sheets
  -> sheets:google:<account_id>:spreadsheet:<spreadsheet_id>
named_services_get <spreadsheet ref>
  -> metadata, tab refs, or explicitly requested A1 ranges
named_services_upsert / named_services_action / named_services_delete
  -> bounded spreadsheet, value, formatting, and tab mutations
```

The namespace remains provider-neutral. Google is its first backing provider.
A read requires a caller grant and connected-account claim for `sheets:read`.
A mutation requires the caller's `sheets:write` grant, while the connected
Google account must hold both `sheets:read` and `sheets:write`. These two
boundaries are checked independently.

### Productivity

MCP endpoint:

```text
/api/integrations/bundles/{tenant}/{project}/kdcube-services@1-0/public/mcp/productivity
```

This typed MCP surface runs over accounts the approving user connected through
Connection Hub. It currently exposes Slack search, mail search/read, and Google
Sheets tools.

| Sheets tools | Caller grant | Connected Google claim |
| --- | --- | --- |
| `productivity_sheets_search`, `productivity_sheets_describe`, `productivity_sheets_read` | `sheets:read` | `sheets:read` |
| values, spreadsheet, tab, and formatting mutations | `sheets:write` | `sheets:read` + `sheets:write` |

The caller grant is selected under **Delegated by KDCube**. The Google claim is
approved under **Delegated to KDCube**. The app resolves the credential only
after both controls pass and never returns it to the MCP client or model.

Sheets calls use the app-owned `requirements.txt` and async `@venv` helper in
`services/productivity/google_sheets.py`; provider normalization lives in the
SDK proxy. The [Google Sheets recipe](../../../../../../../../../docs/recipes/connections/integrations/google-sheets-README.md)
contains the exact setup and regression walk.

### Signed File Transfer

The named-service MCP surface keeps binary bytes out of model context through
three session-less, signed routes:

| Alias | Method | Purpose |
| --- | --- | --- |
| `integration_file_upload` | POST | Upload one short-lived `staged:` file for a later mail/Slack action. |
| `integration_file_download` | GET | Stream a mail or Slack file under the signed delegated-user scope. |
| `conv_file_download` | GET | Stream a `conv:fi:` conversation artifact under the signed user/conversation scope. |

These routes are public only in transport terms. A managed MCP call mints the
short-lived token, and the route trusts the verified token rather than a
browser session or query-supplied identity.

## Shape

```text
kdcube-services@1-0/
  entrypoint.py                  # thin surface adapter
  surfaces/
    mcp/
      conversations.py           # FastMCP tool registration
      named_services.py          # FastMCP named-service bridge registration
      productivity.py            # productivity surface composition
      productivity_sheets.py     # typed Google Sheets MCP registrations
  services/
    conversations/
      export.py                  # conversation export product logic
    named_services/
      bridge.py                  # grant-record namespace policy + dispatch
    productivity/
      google_sheets.py           # credential orchestration + app-owned @venv
  requirements.txt               # optional app dependencies (gspread)
  config/
    bundles.template.yaml
    bundles.secrets.template.yaml
  interface/
    README.md
    kdcube-services.openapi.yaml
  docs/
    README.md
    storage/
      README.md
    journal/
  tests/
    test_interface_contract.py
```

The reusable ontology adapter lives outside the app at
`sdk/integrations/sheets/named_service.py`; the app injects its
`GoogleSheetsService.execute` function when registering the provider.

## Auth Model

The `@mcp(..., auth_config="surfaces.as_provider.mcp.conversations.auth")`
decorator points the platform to the descriptor path for the auth policy.
The actual policy is descriptor-owned:

```yaml
surfaces:
  as_provider:
    mcp:
      conversations:
        auth:
          mode: managed
          authority_id: delegated_client
          tools:
            conversations_export:
              grants: [conversations:read]
          selected_tool_grants: true
```

When a client calls the MCP endpoint:

```text
Bearer token
  -> proc managed MCP guard
  -> Connection Hub delegated credential grant record
  -> authority_id + resource + selected tool + required grants
  -> FastMCP tool dispatch
```

The tool does not check roles itself. By the time it runs, the surface guard has
validated the delegated credential and selected tool grant.

The FastMCP surface uses stateless streamable HTTP because the proc bridge
dispatches each bundle MCP request independently.

For the complete ownership map, including conversation read-through storage,
provider-owned bytes, temporary upload staging, Redis coordination, generated
UI output, and the signing secret, see
[docs/storage/README.md](docs/storage/README.md).

## Extension Rule

Add future KDCube service families as separate modules and separate MCP aliases.
For example:

```text
services/
  conversations/
  usage/
  users/
```

Each family gets its own `surfaces.as_provider.mcp.<alias>.auth` policy and
Connection Hub resource entry, so consent remains concrete and tool-centric.

For named-service republishing, add namespace boundaries under the Connection
Hub delegated credential resource metadata:
`connections.delegated_credentials.oauth.resources[].named_services.namespaces`.
This keeps the MCP service name aligned with the named-service system while
avoiding a separate MCP server per namespace.
