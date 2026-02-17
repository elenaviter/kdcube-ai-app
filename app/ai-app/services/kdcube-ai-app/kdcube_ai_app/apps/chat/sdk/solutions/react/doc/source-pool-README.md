# Sources Pool (source-pool-README.md)

This document describes how the **sources pool** is maintained, persisted, and rendered.

## What it is
The sources pool is a per‑conversation registry of sources collected by tools:
- `web_tools.web_search`, `web_tools.web_fetch` results (URL/title/text/etc.)
- user attachments **if** they are cite‑eligible (`text/*`, `image/*`, `application/pdf`)
- files produced by tools **if** they are cite‑eligible (`text/*`, `image/*`, `application/pdf`)

Each item in the pool has a stable `sid` and metadata.  
The pool is append‑only within a conversation (SIDs never change).

## Eligibility (what is included)
- Web results are always eligible.
- Attachments and produced files are included **only** if their MIME is one of:
  - `text/*`
  - `image/*`
  - `application/pdf`
- Non‑eligible files (e.g., `.xlsx`, `.pptx`, `.docx`, archives) are **not** added to the pool.

## Source item fields (canonical)

All source rows are dictionaries. Fields are **additive**; only a subset may be present.

Common fields:
- `sid` (int, required): Stable ID.
- `source_type` (str, required): `web` | `file` | `attachment` | `manual`.
- `title` (str): Human‑readable label.
- `text` (str): Extract/snippet for web sources or textual artifacts.
- `url` (str): Canonical URL (web sources).
- `domain` (str): Normalized domain (web sources).
- `mime` (str): MIME type for file/attachment sources.
- `size_bytes` (int): File size for file/attachment sources.
- `artifact_path` (str): Logical path, e.g. `fi:<turn>.files/<name>`.
- `physical_path` (str): OUT_DIR‑relative physical path, e.g. `turn_<id>/files/...`.
- `hosted_uri`, `rn`, `key` (str): Hosting references (not rendered to the model).
- `base64` (str): Optional; only for inline binary sources (rare; not produced for large files).

Optional metadata (web sources):
- `published_time_iso`, `modified_time_iso`, `fetched_time_iso`
- `author`, `authority`, `provider_rank`, `weighted_rank`, `date_confidence`

## Source item shapes (examples)

### Web search / fetch
```json
{
  "sid": 12,
  "source_type": "web",
  "url": "https://example.com/article",
  "title": "Article title",
  "text": "Short extract or summary...",
  "domain": "example.com",
  "published_time_iso": "2026-02-10T00:00:00Z",
  "fetched_time_iso": "2026-02-10T12:01:22Z"
}
```

### User attachment (pdf/png)
```json
{
  "sid": 5,
  "source_type": "attachment",
  "title": "menu.pdf",
  "physical_path": "turn_123/attachments/menu.pdf",
  "artifact_path": "fi:turn_123.user.attachments/menu.pdf",
  "mime": "application/pdf",
  "size_bytes": 183942,
  "rn": "rn:.../menu.pdf",
  "hosted_uri": "s3://.../menu.pdf"
}
```

### Files produced by bot (pdf/png)
```json
{
  "sid": 8,
  "source_type": "file",
  "title": "report.pdf",
  "physical_path": "turn_123/files/report.pdf",
  "artifact_path": "fi:turn_123.files/report.pdf",
  "mime": "application/pdf",
  "size_bytes": 45921,
  "rn": "rn:.../report.pdf",
  "hosted_uri": "s3://.../report.pdf"
}
```

## Where it lives (source of truth)
The pool is stored inside `timeline.json`:
```
{
  "version": 1,
  "blocks": [ ... ],
  "sources_pool": [ {sid: 1, ...}, {sid: 2, ...} ],
  ...
}
```

This means:
- It is persisted with the timeline artifact (conv.timeline.v1).
- It is available for `react.read` and `ctx_tools.fetch_ctx` via `so:sources_pool[...]`.

## How it is updated
The pool grows when tools return sources (search/fetch) or when attachments/files are registered as sources.
When new sources are merged, they are assigned new SIDs, and the timeline is flushed to disk.

## How agents see it
When `timeline.render(include_sources=True)` is used, the sources pool is rendered as a **single uncached tail block**:

```
SOURCES POOL (3 sources)
[S:1] example.com  |  "Title..."
[S:2] another.com  |  "Snippet..."
[S:3] fi:turn_123.user.attachments/menu.pdf  |  "<binary>"
```

Notes on rendering:
- The list is **not truncated**; all pool entries are shown.
- For file/attachment sources, the “domain” column shows the **logical artifact path**
  (e.g., `fi:turn_123.files/report.pdf`), not the hosting bucket.
- For `image/*` and `application/pdf`, the snippet is rendered as `<base64>`.
- Token counts for binary sources are estimated from `size_bytes`.
- Hosting fields (`rn`, `hosted_uri`, `key`) are kept in the pool but **not** rendered.

This block is **always the last non‑cached block**, and appears **after announce** when both are included.

## How agents cite sources
- Agents **must** cite sources using tokens like `[[S:1]]`, `[[S:1,2]]`, or `[[S:1-3]]`.
- We do **not** embed raw links into generated text blocks (saves tokens, reduces errors).
- When the system can trace citations for an artifact, it records `sources_used` in the artifact’s timeline metadata.

## Streaming behavior
- Streaming output replaces `[[S:...]]` tokens in‑flight using the sources_pool resolution.
- This keeps user-facing streams concise while preserving source attribution internally.

## How to access sources (agent guidance)
- Use `react.read(paths=["so:sources_pool[1,2]"])` to load sources into visible context.
- Use `ctx_tools.fetch_ctx("so:sources_pool[1,2]")` inside exec code for programmatic access.
- Prefer ranges: `so:sources_pool[1-5]`.
- Comma lists also work: `so:sources_pool[1,3,7]`.

## fetch_ctx behavior for sources_pool
- `fetch_ctx("so:sources_pool[...])` returns the **raw list of source rows**.
- If a row includes `base64`, it is returned as‑is (rare).
- For file/attachment sources:
- **Code must open files via OUT_DIR + physical_path** (physical path).  
  Example: `Path(OUT_DIR) / row["physical_path"]`.
  - `artifact_path` is **logical** and should be used only with `react.read`, not for direct file I/O.

## Example
Rendered order when include_sources + include_announce:

```
... timeline blocks ...
[ANNOUNCE]  (uncached)
SOURCES POOL (uncached)
```

## Notes for clients
Clients may rehydrate citations by matching used SIDs in turn logs with the pool
from `timeline.json`.
