---
id: kdcube-services@1-0/docs/journal/2026-07-27-google-sheets-productivity.md
title: "Google Sheets Productivity Surface"
summary: "Adds governed, bounded Google Sheets tools through the existing productivity MCP surface and connected-account boundary."
status: active
tags: ["kdcube-services", "google-sheets", "productivity", "mcp", "connected-accounts", "venv"]
---

# Google Sheets Productivity Surface

`kdcube-services@1-0` now exposes typed Google Sheets discovery, read, values,
tab, and formatting tools through `public/mcp/productivity`.

The public productivity composition remains in `surfaces/mcp/productivity.py`;
the Sheets claim catalog and typed registrations live in the focused
`surfaces/mcp/productivity_sheets.py` module.

The app resolves the current user's Google credential through Connection Hub,
then invokes an async app-owned `@venv` helper. The optional `gspread`
dependency and its synchronous provider calls stay in that child process; the
SDK owns only the serializable Google Sheets proxy. Provider tokens do not enter
MCP arguments or results.

The resource uses two independent controls: selected MCP tool grants under
Delegated by KDCube, and `sheets:read`/`sheets:write` connected-account claims
under Delegated to KDCube. Existing Google accounts use the stable `gmail`
connector-app id and receive a claim-upgrade flow when required.

See the [interface](../../interface/README.md) and the platform
[Google Sheets recipe](../../../../../../../../../../../docs/recipes/connections/integrations/google-sheets-README.md).
