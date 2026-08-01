---
id: kdcube-services@1-0/docs/journal/2026-07-29-google-docs-productivity.md
title: "Google Docs Productivity Surface"
summary: "Google Docs joins the productivity door and named-service registry with exact-title discovery, native copy, editing, comments, and portable export delivery."
status: active
tags: ["kdcube-services", "productivity", "google", "docs", "named-services", "mcp"]
keywords: ["google docs productivity", "document named service", "google docs mcp", "connected account", "document export"]
see_also:
  - repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/bundles/kdcube-services@1-0/docs/journal/2026-08-01-docs-explicit-tab-selection.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/connections/integrations/google-service-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/integrations/google/google-README.md
---

# Google Docs Productivity Surface

Google Docs is now a first-class productivity integration alongside Sheets,
exposed on **both** surfaces — the `docs` named-service namespace and the
`productivity_docs_*` MCP tools — through the same two-gate delegated-credential
model (a caller grant *and* a connected-account claim; the Google token never
reaches the agent).

The later [import-source discovery update](2026-08-01-docs-import-source-discovery.md)
extends title lookup and copy to compatible DOCX, ODT, and RTF files stored in
Drive. Native Google Docs continue to use the provider-native copy path below.
The [explicit tab-selection update](2026-08-01-docs-explicit-tab-selection.md)
adds complete multi-tab metadata, exact mutation scope, and natural lexical tab
selectors on the provider-neutral named-service path.

## What the app gained

- **Service** `services/productivity/google_docs.py` — `GoogleDocsService`,
  resolving the per-account Google credential per claim, with one refresh-retry
  and the standard `{ok,error,ret}` envelope. It calls the SDK Docs proxy
  **async in-proc** (no `@venv`): Docs speaks raw REST to the Docs API + Drive
  API over `httpx`, needing no heavy blocking dependency, so unlike Sheets it
  adds nothing to `requirements.txt`.
- **MCP** `surfaces/mcp/productivity_docs.py` — 18 typed
  `productivity_docs_*` tools: search, get, export, list_comments, get_comment
  (`docs:read`); create, insert_text, append_text, replace_text,
  copy, apply_text_style, insert_page_break, embed_image, import
  (`docs:read`+`docs:write`); create_comment, reply_comment, resolve_comment,
  delete_comment (`docs:read`+`docs:comment`). `delete_comment` is the only
  destructive annotation; `resolve_comment` is a write (it posts a resolving
  reply). Three additional tools provide separately governed structure, tab,
  and native batch-edit access. Registered in `productivity.py` next to Sheets;
  `bind_docs_service` is rebound per request in the MCP-door method for correct
  identity.
- **Named service** — the entrypoint's `_named_service_providers()` registers
  the `docs` namespace (SDK `integrations/docs/named_service.py`); the
  signed-download branch serves `kdcube.docs.snapshot.v1` snapshots and
  portable document exports. Exact-title search runs before Google's
  title-prefix query, and `object.action.copy` uses Drive's native copy so the
  document structure and formatting survive. `docs.document` represents the
  provider document; `docs.export` represents one resolvable file export.
  Comment mutations route through `object.action`, while `object.delete`
  deletes one comment rather than the document file.

Named-service export returns a portable ref and filename. In KDCube chat, the
signed URL becomes a file card and is removed from model-visible output. A
materializing client can stream the export ref, while a turnless MCP client can
use the short-lived signed URL. Google credentials remain on the trusted side
for every route.

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

The original Docs pack passed 96 automated tests. The exact-search/copy/export
follow-up passed 55 focused proxy, named-service, ReAct projection, claim-roster,
and signed-download tests. The full bundle suite reached 280 passes and 35
skips; three surface-construction tests stop at a stale local MCP SDK import
before bundle code runs.
