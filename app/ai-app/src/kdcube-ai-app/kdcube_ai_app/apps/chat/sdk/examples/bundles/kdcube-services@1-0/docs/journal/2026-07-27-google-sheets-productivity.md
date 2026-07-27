---
id: kdcube-services@1-0/docs/journal/2026-07-27-google-sheets-productivity.md
title: "Google Sheets Productivity Surface"
summary: "Adds governed Google Sheets operations through typed productivity tools and the provider-neutral sheets named-service namespace."
status: active
tags: ["kdcube-services", "google-sheets", "productivity", "named-services", "mcp", "connected-accounts", "venv"]
---

# Google Sheets Productivity Surface

`kdcube-services@1-0` now exposes typed Google Sheets discovery, read, values,
tab, and formatting tools through `public/mcp/productivity`.

It also publishes the same capability through `public/mcp/named_services` as
namespace `sheets`. The generic surface returns stable spreadsheet and tab
refs and maps `object.search`, `object.get`, `object.upsert`, `object.action`,
and tab-only `object.delete` onto the existing Sheets service.

The public productivity composition remains in `surfaces/mcp/productivity.py`;
the Sheets claim catalog and typed registrations live in the focused
`surfaces/mcp/productivity_sheets.py` module.

The app resolves the current user's Google credential through Connection Hub,
then invokes an async app-owned `@venv` helper. The optional `gspread`
dependency and its synchronous provider calls stay in that child process; the
SDK owns only the serializable Google Sheets proxy. Provider tokens do not enter
MCP arguments or results.

The `sdk/integrations/sheets/named_service.py` adapter owns only ontology:
provider-neutral refs, schema, action names, and response normalization. It
reuses `GoogleSheetsService.execute`, so the typed and generic doors share
account selection, claim checks, bounded provider operations, retry behavior,
and the app-owned venv.

Each resource uses two independent controls: selected tool/namespace grants
under Delegated by KDCube, and `sheets:read`/`sheets:write` connected-account
claims under Delegated to KDCube. Existing Google accounts use the stable
`gmail` connector-app id and receive a claim-upgrade flow when required.

See the [interface](../../interface/README.md) and the platform
[Google Sheets recipe](../../../../../../../../../../../docs/recipes/connections/integrations/google-sheets-README.md).
