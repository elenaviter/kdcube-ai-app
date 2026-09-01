# Web search and fetch on the productivity surface

**2026-09-01**

The productivity MCP door gains two no-account tools:
`productivity_web_search` and `productivity_web_fetch`
(`surfaces/mcp/productivity_web.py`). Unlike every other tool on the
surface they touch no connected account and declare no claims: the
service side runs on the platform's own search and fetch backends
(`tools/backends/web`).

Two operator-owned knobs, read from the surface config
(`surfaces.as_provider.mcp.productivity.web` in the descriptor), and a
call cannot widen either:

- **Domain allowlist** (`allowlist` inline or `allowlist_file`, one
  entry per line, re-read on change). Search results from hosts outside
  it are dropped inside the search backend before any content fetch
  (the backend's new `allowed_domains` parameter), and fetching an
  outside host answers with `denied_by_allowlist`, naming the host and
  the allowlist, while other URLs in the same call still fetch. With an
  allowlist configured, the fetcher's archive-mirror fallback stays off:
  an archive host is a different host. Matching semantics live in
  `tools/backends/web/allowlist.py` (`example.org` = domain plus
  subdomains, `*.example.org` = subdomains only). Unset = every host
  allowed.
- **`use_llm_default`** (False when absent). The search backend's new
  `use_llm` parameter runs the pipeline without LLM steps: no snippet
  reconciliation, no LLM content filtering, `_SERVICE` may be None, so
  no model keys are needed. Search, allowlist filtering, and content
  fetch still work. Callers can pass `use_llm` per call; the default is
  the operator's.

Both tools answer in the `{ok, error, ret}` envelope with the same row
shape (`title`, `url`, `text` preview, `content` full text, `mime` /
`base64` for binaries, dates). Tool documentation follows the register
of the resident `web_tools.py` pair — discovery vs dereference selection
rules, refinement modes with coverage percentages, result shape — with
the resident-runtime concepts (sources pool, sids, read tool) left out,
since an external MCP caller has none of them.

The same allowlist and `use_llm` reached the standalone MCP server
(`tools/mcp/web_search/web_search_server.py`), which also gained
`web_fetch` and `allowlist_status`; its README carries the Claude Code
and Claude Desktop setup for operators outside KDCube.

Tests: `tests/test_productivity_web.py` (allowlist to backend, LLM-off
builds no model service, denial rows, archive fallback off, envelope
shapes), `tools/backends/web/test_allowlist.py`,
`tools/mcp/web_search/test_web_search_server.py`. Roster and claim-free
declarations added to `tests/test_productivity_surface.py`.
