# Agent guide: web_search MCP server

Orientation for an agent working in this folder — extending the server,
debugging a deployment of it, or auditing what it does.

## What this folder is

An MCP server exposing `web_search`, `web_fetch`, and `allowlist_status`
over KDCube's web backends, with a server-side egress filter (allowlist
plus blocklist, deny wins), per-call `sites` scoping, and optional LLM
steps. Contracts and config: [TOOLS.md](TOOLS.md). Setup:
[README.md](README.md).

## Layout and the code it stands on

```
web_search_server.py      the MCP wrapper: tool registration, egress
                          filter wiring, sites clamping, LLM on/off,
                          per-URL fetch denials, and the YAML config
                          loader (apply_yaml_config: scoped sections ->
                          env, env wins on conflict)
list_edit.py              textual editor behind the operator-enabled
                          allowlist_edit tool: entry lines only,
                          comments and everything else byte-identical
config.example.yaml       the YAML config template (the recommended
                          mode). The operator's config.yaml lives in
                          the INSTALL DIRECTORY, beside the clone, and
                          is passed with --config (discovery also finds
                          it in the working directory; one beside the
                          server file is the in-repo dev case only).
                          Its inline filter.allowlist / filter.blocklist
                          are the live list sources
.env.example              the same settings as raw environment variables
                          (the mode MCP client configs and CI speak)
test_web_search_server.py contract tests (run with pytest)
../mcp_app_transport.py   stdio/sse/http runners (dependency-free)
../../backends/web/
  allowlist.py            entry parsing, hostname matching, Allowlist
                          (env/file/yaml sources, re-read on mtime
                          change) and EgressFilter (blocklist deny wins
                          over allowlist; deny_reason for in-band
                          denials)
  search_backends.py      the search orchestrator; `use_llm`,
                          `allowed_domains`/`blocked_domains`, and the
                          `sites` query rewrite land here
  fetch_backends.py       fetch_url_contents (text extraction, dates)
                          + the SSRF guard's per-URL pre-check
  rows.py                 the outward row shape: MCP surfaces keep-list
                          result fields so pipeline internals (spans,
                          ranks, sids) never reach the calling model
  ssrf_guard.py           address-level layer (ported from PR #1):
                          blocked IP ranges, metadata hostnames, DNS
                          validation, and the guarded aiohttp resolver
                          installed by web_extractors' session
  test_allowlist.py       matcher + source semantics tests
  test_ssrf_guard.py      blocked ranges, resolver rejection, pre-check
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
MCP layer — but the YAML config is applied only by `main()`, so a direct
caller MUST start with `load_config()` (or set the env vars from
TOOLS.md itself), and then confirm `allowlist_status()` reports
`enforced: true` with the expected entries BEFORE any live call. An
unconfigured allowlist allows every host, so skipping this turns a
"small live check" into unrestricted egress:

```python
import kdcube_ai_app.apps.chat.sdk.tools.mcp.web_search.web_search_server as srv
srv.load_config()                              # discovers config.yaml
status = await srv.allowlist_status()
assert status["enforced"] and status["allowlist_entries"], status
rows = await srv.web_search(queries="...", n=5, fetch_content=False, use_llm=False)
```
When you add an import to the server or the backends it reaches, re-run
the clean-venv setup and add what broke to `requirements.txt` — that
file is part of the tool's contract.

## Invariants to keep

- **Enforcement lives in code, never in the model's instructions.** The
  egress filter sits inside `search_backends.web_search` after
  deduplication and before any content fetch, and in `web_fetch` before
  any request. A tool call must not be able to widen the operator's
  filter — the `sites` parameter only narrows inside it, and a call
  whose every site is excluded fails with the per-site reasons named.
- **Deny wins.** A blocklisted host is refused even when the allowlist
  admits it, everywhere: search post-filter, fetch, sites clamping.
- **The SSRF guard defaults ON and sits under the lists.** Private,
  loopback, link-local, and metadata addresses stay unreachable
  whatever the operator's lists say, unless the operator explicitly
  disables the guard. Don't weaken the resolver path: it is what checks
  redirect targets and rebinding answers with the exact IPs used to
  connect.
- **Denials are explained in-band**: host, reason, the list's source in
  the result, so the calling agent can relay what the operator would
  need to change. Do not turn a denial into a silent drop or a bare
  exception.
- **`use_llm=false` must stay key-free**: no model service is built and
  no LLM step runs. If you touch the orchestrator, keep the
  `_SERVICE=None` path working.
- **Archive fallback stays off while the egress filter is configured** —
  an archive mirror is a different host.
- **Unset allowlist = allow all; configured-empty = deny all; unset
  blocklist = block none.** Existing deployments without the config must
  keep working unchanged.
- **Tool descriptions follow the register of `sdk/tools/web_tools.py`**
  (selection rules, refinement modes with coverage, result shapes) but
  never mention resident-runtime concepts (sources pool, sids, react
  tools): an external MCP caller has none of them.
- **Accounting events flow through unchanged.** The backends emit usage
  events (search provider/tier, LLM usage); inside KDCube they land in
  deployment accounting, standalone they drop harmlessly with no storage
  bound. Don't strip the decorators and don't make an unbound context an
  error.

## Running safely on a dev machine

The README's Prerequisites section is the checklist; the parts that
matter to an agent:

- **Always work from a venv built from this folder's
  `requirements.txt`** — a preexisting platform venv may carry an old
  `mcp` (no v2 `MCPServer`, the server won't import) or a drifted
  `anthropic` (the segmenter silently returns nothing). The pins are the
  contract.
- **The pytest suite is the safe check**: it fakes every network and
  model call, needs no keys, and can run on any machine without egress
  or spend. Run it first.
- **A live check egresses and spends**: search calls the provider on
  the operator's key, `use_llm=true` calls the model. Do it only with a
  config the operator gave you, keep the allowlist configured so egress
  stays inside it, and prefer one small call (`n<=5`,
  `fetch_content=false`) before anything bigger.
- **`config.yaml`, `.env`, and `web-allowlist.txt` are gitignored
  operator files** — never commit them, never print the key values.
- **HTTPS failures that look like tool bugs**
  (`CERTIFICATE_VERIFY_FAILED`, or pages classified `paywall`/`error`
  that open fine in a browser) are usually the machine's missing CA
  store: set `tls.cert_file` to certifi's bundle
  (`python -m certifi`) and retry before touching fetch code.

## Known limitations

- The page fetcher follows HTTP redirects, so a listed site redirecting
  off-domain can lead outside the allowlist (hop-level re-checking would
  need changes in `fetch_backends`).

## Related surfaces

The same backends serve the resident agent tools
(`sdk/tools/web_tools.py`) and the productivity MCP door of the
kdcube-services app
(`sdk/examples/bundles/kdcube-services@1-0/surfaces/mcp/productivity_web.py`),
where the allowlist and the `use_llm` default come from the app
descriptor instead of env. A change to backend semantics reaches all
three — check their tests together.
