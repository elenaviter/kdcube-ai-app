# Agent guide: web_search MCP server

Orientation for an agent working in this folder — extending the server,
debugging a deployment of it, or auditing what it does.

## What this folder is

An MCP server exposing `web_search`, `web_fetch`, and `allowlist_status`
over KDCube's web backends, with a server-side domain allowlist and
optional LLM steps. Contracts and config: [TOOLS.md](TOOLS.md). Setup:
[README.md](README.md).

## Layout and the code it stands on

```
web_search_server.py      the MCP wrapper: tool registration, allowlist
                          wiring, LLM on/off, per-URL fetch denials
test_web_search_server.py contract tests (run with pytest)
../mcp_app_transport.py   stdio/sse/http runners (dependency-free)
../../backends/web/
  allowlist.py            entry parsing, hostname matching, Allowlist
                          (env/file source, file re-read on mtime change)
  search_backends.py      the search orchestrator; `use_llm` and
                          `allowed_domains` parameters land here
  fetch_backends.py       fetch_url_contents (text extraction, dates)
  test_allowlist.py       matcher + source semantics tests
```

The server builds on `KDCubeMCPServer`
(`kdcube_ai_app/apps/chat/sdk/runtime/mcp/server.py`), which needs the
MCP SDK v2 (`from mcp.server import MCPServer`). Environments without it
can still import and test everything except `_build_mcp_app`; the tests
stub that seam.

## Running and testing

`requirements.txt` in this folder is the complete dependency set: a
clean venv with it, plus `app/ai-app/src/kdcube-ai-app` on `PYTHONPATH`,
runs everything below. Set up that way first — it is also the proof that
the folder stays self-contained.

```bash
# from app/ai-app/src/kdcube-ai-app
python3 -m venv .venv-websearch
.venv-websearch/bin/pip install -r \
  kdcube_ai_app/apps/chat/sdk/tools/mcp/web_search/requirements.txt
.venv-websearch/bin/pip install pytest

PYTHONPATH=$PWD .venv-websearch/bin/python -m pytest \
  kdcube_ai_app/apps/chat/sdk/tools/mcp/web_search/ \
  kdcube_ai_app/apps/chat/sdk/tools/backends/web/test_allowlist.py

PYTHONPATH=$PWD .venv-websearch/bin/python -m \
  kdcube_ai_app.apps.chat.sdk.tools.mcp.web_search.web_search_server \
  --transport stdio --allowlist /path/to/allowlist.txt
```

The tool functions are directly callable for a live check without the
MCP layer: import `web_search_server` and await `web_search(...)` /
`web_fetch(...)` / `allowlist_status()` with the env from TOOLS.md set.
When you add an import to the server or the backends it reaches, re-run
the clean-venv setup and add what broke to `requirements.txt` — that
file is part of the tool's contract.

## Invariants to keep

- **Enforcement lives in code, never in the model's instructions.** The
  allowlist filter sits inside `search_backends.web_search` after
  deduplication and before any content fetch, and in `web_fetch` before
  any request. A tool call must not be able to widen the operator's
  allowlist.
- **Denials are explained in-band**: host, reason, allowlist source in
  the result, so the calling agent can relay what the operator would
  need to change. Do not turn a denial into a silent drop or a bare
  exception.
- **`use_llm=false` must stay key-free**: no model service is built and
  no LLM step runs. If you touch the orchestrator, keep the
  `_SERVICE=None` path working.
- **Archive fallback stays off while an allowlist is configured** — an
  archive mirror is a different host.
- **Unset allowlist = allow all; configured-empty = deny all.** Existing
  deployments without the config must keep working unchanged.
- **Tool descriptions follow the register of `sdk/tools/web_tools.py`**
  (selection rules, refinement modes with coverage, result shapes) but
  never mention resident-runtime concepts (sources pool, sids, react
  tools): an external MCP caller has none of them.
- **Accounting events flow through unchanged.** The backends emit usage
  events (search provider/tier, LLM usage); inside KDCube they land in
  deployment accounting, standalone they drop harmlessly with no storage
  bound. Don't strip the decorators and don't make an unbound context an
  error.

## Known limitations

- The page fetcher follows HTTP redirects, so a listed site redirecting
  off-domain can lead outside the allowlist (hop-level re-checking would
  need changes in `fetch_backends`).
- Local dev machines may lack the MCP SDK v2 and CA certificates for
  aiohttp; both are environment gaps, not tool logic.

## Related surfaces

The same backends serve the resident agent tools
(`sdk/tools/web_tools.py`) and the productivity MCP door of the
kdcube-services app
(`sdk/examples/bundles/kdcube-services@1-0/surfaces/mcp/productivity_web.py`),
where the allowlist and the `use_llm` default come from the app
descriptor instead of env. A change to backend semantics reaches all
three — check their tests together.
