# Web search MCP server

Web search and fetch as an MCP server, with two properties an operator
controls server-side:

- **Egress filter: allowlist and blocklist.** When the allowlist is
  configured, search results from hosts outside it are dropped before
  any content fetch, and a fetch of an outside host is denied with a
  result naming the host and the list's source. The optional blocklist
  holds always-refused hosts, and deny wins over allow. Both are plain
  config the operator owns and can read; the model reaches the internet
  only through these tools. Allowlist unset = every host allowed;
  blocklist unset = no host blocked.
- **LLM on/off.** Every call takes `use_llm`. With `use_llm=false` the
  pipeline runs without LLM steps (no snippet reconciliation, no LLM
  content filtering) and needs no model API keys: search, egress
  filtering, and content fetch still work.

The model can also scope one search WITHIN chosen domains through
`web_search`'s `sites` parameter — the provider query is rewritten with
`site:` operators, and the scoping narrows inside the operator's
filter, never widens it.

Tools: `web_search`, `web_fetch`, `allowlist_status` (the egress filter
exactly as the server enforces it — both lists — so the model and the
operator read the same truth), and optionally `site_filter_edit` when the
operator sets `filter.expose_edit_tool: true` — then the user changes
the lists conversationally from any client, and without the opt-in no
tool can touch them. Result rows leave the server keep-listed: what the
model acts on (title, url, snippet, content, statuses, scores, dates),
none of the pipeline's internal bookkeeping.

Full reference — every tool's contract, how the pipeline works inside,
and the complete config table — is in [TOOLS.md](TOOLS.md). An agent
working on this folder starts at [AGENTS.md](AGENTS.md).

The short way in is the repo-root launcher
[`mcp/web-search/`](/mcp/web-search/): same server, no PYTHONPATH
needed, one shallow path to point people at — its README also carries
the usage cheatsheet (what to say to Claude, operator quick moves).

## Prerequisites

- **Python 3.11+** and network access to PyPI for the install.
- **A dedicated venv built from this folder's `requirements.txt`.** Do
  not run the server from a preexisting platform venv: those may carry
  an older `mcp` package without the v2 `MCPServer`, and an older or
  newer `anthropic` than the pinned one (either breaks the neural
  pipeline, the newer one silently). Check `which python3` before
  creating the venv — on a machine with KDCube installed, `python3`
  itself may resolve into a platform venv; use an explicit interpreter
  (`python3.11`, `python3.12`, or a full path) if it does.
- **A Brave Search API key is optional.** Without one, `web_search`
  runs on the DuckDuckGo backend (no key, no signup); a Brave key
  gives better results and limits. A free-tier Brave key rate-limits
  quickly on consecutive searches, and the server falls back to
  DuckDuckGo transparently either way. `web_fetch` and
  `allowlist_status` need no key at all.
- **An Anthropic key** (or OpenAI/Google) only for `use_llm=true` — the
  neural pipeline. Everything runs without it at `use_llm=false`.
- **A working CA store.** Some Python builds (pyenv, the macOS
  installer) ship without CA certificates wired for aiohttp, and HTTPS
  fetches then fail with `CERTIFICATE_VERIFY_FAILED`. Check and fix:

  ```bash
  .venv/bin/python -c "import ssl; print(ssl.create_default_context().cert_store_stats())"
  # {'x509': 0, ...} or a verify error on fetches means no CA store:
  # point tls.cert_file (or SSL_CERT_FILE) at certifi's bundle:
  .venv/bin/python -m certifi
  ```

Sanity check after setup — offline, no keys, safe anywhere (run from
your install dir, see Quick start for `$REPO_SRC`):

```bash
.venv/bin/pip install pytest
PYTHONPATH=$REPO_SRC .venv/bin/python -m pytest \
  $REPO_SRC/kdcube_ai_app/apps/chat/sdk/tools/mcp/web_search/ \
  $REPO_SRC/kdcube_ai_app/apps/chat/sdk/tools/backends/web/test_allowlist.py
```

The tests fake every network and model call, so this proves the install
without egress and without spending anything.

## Quick start

The layout: **your install directory owns the config; the clone is just
code beside it.** The `config.yaml` with your keys and allowlist sits at
the top of the directory you set the tool up in — never buried inside
the clone, and it survives the clone being updated or replaced.

```bash
mkdir web-search-mcp && cd web-search-mcp    # your install dir
# a sparse clone: the repo is large and this tool needs two dirs of it
git clone --depth 1 --filter=blob:none --sparse https://github.com/kdcube/kdcube.git
git -C kdcube sparse-checkout set mcp app/ai-app/src/kdcube-ai-app
export REPO_SRC=$PWD/kdcube/app/ai-app/src/kdcube-ai-app

python3.11 -m venv .venv    # explicit interpreter, see Prerequisites
.venv/bin/pip install -r \
  $REPO_SRC/kdcube_ai_app/apps/chat/sdk/tools/mcp/web_search/requirements.txt

# all settings in one YAML: allowlist, provider keys, pipeline models
cp $REPO_SRC/kdcube_ai_app/apps/chat/sdk/tools/mcp/web_search/config.example.yaml \
   ./config.yaml    # edit it; keep it out of any git repo

PYTHONPATH=$REPO_SRC .venv/bin/python -m \
  kdcube_ai_app.apps.chat.sdk.tools.mcp.web_search.web_search_server \
  --transport stdio --config $PWD/config.yaml
```

Config discovery order: `--config PATH`, then `WEB_SEARCH_CONFIG`, then
`config.yaml` in the working directory, then one beside the server file
(the in-repo development case). For launch configs prefer the explicit
`--config` with an absolute path — an MCP client's working directory is
not always yours. With the allowlist written inline under
`filter.allowlist`, the config file is its live source: edit the list
and the next call already follows it. Everything can also be configured
through environment variables instead — the two modes carry the same
settings, and an environment variable wins over the file (TOOLS.md has
both forms in full).

Claude Code (from the install dir):

```bash
claude mcp add web-search \
  --env PYTHONPATH=$REPO_SRC \
  -- $PWD/.venv/bin/python -m kdcube_ai_app.apps.chat.sdk.tools.mcp.web_search.web_search_server \
  --config $PWD/config.yaml
```

Claude Desktop (`claude_desktop_config.json`) — same idea; any variable
added under `env` overrides a YAML value:

```json
{
  "mcpServers": {
    "web-search": {
      "command": "/path/to/web-search-mcp/.venv/bin/python",
      "args": [
        "-m", "kdcube_ai_app.apps.chat.sdk.tools.mcp.web_search.web_search_server",
        "--config", "/path/to/web-search-mcp/config.yaml"
      ],
      "env": {
        "PYTHONPATH": "/path/to/web-search-mcp/kdcube/app/ai-app/src/kdcube-ai-app"
      }
    }
  }
}
```

The list sources are re-read whenever they change, so edits to
`filter.allowlist` and `filter.blocklist` apply to the next call without
a restart. Entry format, shared by both lists:

```
example.org        # example.org and every subdomain
*.example.org      # subdomains only
```

**Per-user filters.** The filter belongs to a server process, and with
stdio every user's client launches its own process — so per-user lists
are the natural deployment: give each user (or group) their own config,
owned by the admin and readable by the user, and point that user's MCP
registration at it. Different users on the same machine, or different
teams, simply get different configs. A shared `http`/`sse` instance
applies one filter to everyone it serves; run one instance per profile
when profiles differ.

Search needs a search-provider key (Brave by default) — in
`services.secrets` of config.yaml or the environment. `use_llm=true`
additionally needs a model key and turns on the neural pipeline —
snippet relevance scoring, content filtering, and objective-guided span
segmentation of fetched pages, each stage on its own configured model
role (Haiku-class by default; see TOOLS.md, "The neural pipeline").
`web_fetch` needs neither.

## Enforcement notes

- The egress filter is enforced in this server and in the search
  backend, never by instructions to the model, and deny wins: a
  blocklisted host is refused even when the allowlist admits it.
- The `sites` scoping is clamped against the filter before the provider
  is called; a call whose every site is excluded fails with the reasons
  named.
- Under the name-level filter sits the address-level **SSRF guard**
  (default on): private, loopback, link-local (cloud metadata included),
  CGNAT, multicast, and reserved addresses, and metadata-style
  hostnames, are refused regardless of the lists — as a per-URL
  pre-check and as a guarded DNS resolver validating every answer at
  connect time. `filter.ssrf_guard: false` disables it for deployments
  that must fetch internal hosts.
- With a filter configured, `web_fetch`'s archive-mirror fallback stays
  off: an archive host is a different host.
- The page fetcher follows HTTP redirects, so a listed site that
  redirects off-domain can lead outside the *allowlist*; the SSRF guard
  still checks each redirect hop's DNS answers, so an off-domain
  redirect cannot reach private or metadata addresses. A redirect
  straight to an internal IP literal is the one residual gap. List only
  sites you trust.
