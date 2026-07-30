---
id: kdcube-services@1-0/docs/journal/2026-07-29-google-docs-productivity.md
title: "Google Docs Productivity Surface"
summary: "Google Docs joins the productivity door and the named-service registry, mirroring Sheets, with comments, image embed, export, and import."
status: active
tags: ["kdcube-services", "productivity", "google", "docs", "named-services", "mcp"]
---

# Google Docs Productivity Surface

Google Docs is now a first-class productivity integration alongside Sheets,
exposed on **both** surfaces — the `docs` named-service namespace and the
`productivity_docs_*` MCP tools — through the same two-gate delegated-credential
model (a caller grant *and* a connected-account claim; the Google token never
reaches the agent).

## What the app gained

- **Service** `services/productivity/google_docs.py` — `GoogleDocsService`,
  resolving the per-account Google credential per claim, with one refresh-retry
  and the standard `{ok,error,ret}` envelope. It calls the SDK Docs proxy
  **async in-proc** (no `@venv`): Docs speaks raw REST to the Docs API + Drive
  API over `httpx`, needing no heavy blocking dependency, so unlike Sheets it
  adds nothing to `requirements.txt`.
- **MCP** `surfaces/mcp/productivity_docs.py` — 17 typed tools
  (`productivity_docs_*`): search, get, export, list_comments, get_comment
  (`docs:read`); create, insert_text, append_text, replace_text,
  apply_text_style, insert_page_break, embed_image, import
  (`docs:read`+`docs:write`); create_comment, reply_comment, resolve_comment,
  delete_comment (`docs:read`+`docs:comment`). `delete_comment` is the only
  destructive annotation; `resolve_comment` is a write (it posts a resolving
  reply). Registered in `productivity.py` next to Sheets; `bind_docs_service`
  is rebound per request in the MCP-door method for correct identity.
- **Named service** — the entrypoint's `_named_service_providers()` registers
  the `docs` namespace (SDK `integrations/docs/named_service.py`); the
  signed-download branch serves `kdcube.docs.snapshot.v1` snapshots via
  `fetch_google_docs_snapshot`. Single `docs.document` object kind (flatter
  than Sheets' spreadsheet+tab); comment mutations route through `object.action`
  and `object.delete`=delete_comment.

## New vs Sheets

Sheets deliberately excluded comments; Docs adds them (read + write), plus
inline **image embed**, **export** (PDF/DOCX/TXT/HTML/MD), and **import**
(upload+convert). This needs a third claim, `docs:comment`, and **broader Drive
scopes** than Sheets: `docs:read` uses `drive.readonly` (read/export any doc,
read comments), `docs:comment` uses full `drive` (write comments on any doc),
while `docs:write` stays least-privilege at `drive.file`. The third claim
isolates the full-`drive` escalation to only users who enable commenting.

Image embed: the Docs API `insertInlineImage` needs a public HTTPS URI Google
fetches once; the proxy takes `image_uri`, and staging arbitrary bytes to a
short-lived signed URL is left to the service/hosting layer.

## Configuration

`docs:read` / `docs:write` / `docs:comment` claims were added to the existing
`google`/`gmail` connector (one Google OAuth app, more claims) across all five
descriptor sets: provider claims+scopes, connector `allowed_claims`, door
grants, the `docs` tool catalog, the `productivity_docs_*` MCP grants, and the
agent named-service namespace allow-list. Operator: enable the Google Docs API
and register the new scopes on the OAuth consent screen, then reconnect.

Verification: 96 automated tests (proxy 14, named-service 24, bundle roster +
import contract). See the platform feature journal
`journal/26/07/productivity/journal/2026.07.29-google-docs-plan.md`.
