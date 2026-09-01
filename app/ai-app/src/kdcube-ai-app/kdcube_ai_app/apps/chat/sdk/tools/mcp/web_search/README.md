# Web search MCP server

Web search and fetch as an MCP server, with two properties an operator
controls server-side:

- **Domain allowlist.** When configured, search results from hosts
  outside the allowlist are dropped before any content fetch, and a
  fetch of a host outside it is denied with a result naming the host and
  the allowlist source. The allowlist is a plain config the operator
  owns and can read; the model reaches the internet only through these
  tools. When no allowlist is configured, every host is allowed.
- **LLM on/off.** Every call takes `use_llm`. With `use_llm=false` the
  pipeline runs without LLM steps (no snippet reconciliation, no LLM
  content filtering) and needs no model API keys: search, allowlist
  filtering, and content fetch still work.

Tools: `web_search`, `web_fetch`, `allowlist_status` (the allowlist
exactly as the server enforces it, so the model and the operator read
the same truth).

Full reference — every tool's contract, how the pipeline works inside,
and the complete config table — is in [TOOLS.md](TOOLS.md). An agent
working on this folder starts at [AGENTS.md](AGENTS.md).

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
- **A Brave Search API key** for `web_search`; `web_fetch` and
  `allowlist_status` need no key at all. A free-tier Brave key
  rate-limits quickly on consecutive searches; the server then falls
  back to the DuckDuckGo backend transparently, so a first live check
  may report a different provider on its second call.
- **An Anthropic key** (or OpenAI/Google) only for `use_llm=true` — the
  neural pipeline. Everything runs without it at `use_llm=false`.
- **A working CA store.** Some Python builds (pyenv, the macOS
  installer) ship without CA certificates wired for aiohttp, and HTTPS
  fetches then fail with `CERTIFICATE_VERIFY_FAILED`. Check and fix:

  ```bash
  .venv-websearch/bin/python -c "import ssl; print(ssl.create_default_context().cert_store_stats())"
  # {'x509': 0, ...} or a verify error on fetches means no CA store:
  # point tls.cert_file (or SSL_CERT_FILE) at certifi's bundle:
  .venv-websearch/bin/python -m certifi
  ```

Sanity check after setup — offline, no keys, safe anywhere:

```bash
.venv-websearch/bin/pip install pytest
PYTHONPATH=$PWD .venv-websearch/bin/python -m pytest \
  kdcube_ai_app/apps/chat/sdk/tools/mcp/web_search/ \
  kdcube_ai_app/apps/chat/sdk/tools/backends/web/test_allowlist.py
```

The tests fake every network and model call, so this proves the install
without egress and without spending anything.

## Quick start

The server lives inside the KDCube repo but runs standalone: clone the
repo, install this folder's `requirements.txt` into a fresh venv, and
put the source root on `PYTHONPATH`. Nothing else of the platform needs
to be set up.

```bash
git clone https://github.com/kdcube/kdcube.git
cd kdcube/app/ai-app/src/kdcube-ai-app

python3 -m venv .venv-websearch
.venv-websearch/bin/pip install -r \
  kdcube_ai_app/apps/chat/sdk/tools/mcp/web_search/requirements.txt

# all settings in one YAML: allowlist, provider keys, pipeline models
cd kdcube_ai_app/apps/chat/sdk/tools/mcp/web_search
cp config.example.yaml config.yaml   # gitignored; edit it
cd -

PYTHONPATH=$PWD .venv-websearch/bin/python -m \
  kdcube_ai_app.apps.chat.sdk.tools.mcp.web_search.web_search_server \
  --transport stdio
```

The server finds `config.yaml` beside itself (or via `--config PATH` /
`WEB_SEARCH_CONFIG`). With the allowlist written inline under
`filter.allowlist`, the config file is its live source: edit the list
and the next call already follows it. Everything can also be configured
through environment variables instead — the two modes carry the same
settings, and an environment variable wins over the file (TOOLS.md has
both forms in full).

Claude Code (from the same directory; the config.yaml in the server's
folder supplies everything else):

```bash
claude mcp add web-search \
  --env PYTHONPATH=$PWD \
  -- $PWD/.venv-websearch/bin/python -m kdcube_ai_app.apps.chat.sdk.tools.mcp.web_search.web_search_server
```

Claude Desktop (`claude_desktop_config.json`) — same idea; any variable
added under `env` overrides the YAML value:

```json
{
  "mcpServers": {
    "web-search": {
      "command": "/path/to/kdcube/app/ai-app/src/kdcube-ai-app/.venv-websearch/bin/python",
      "args": ["-m", "kdcube_ai_app.apps.chat.sdk.tools.mcp.web_search.web_search_server"],
      "env": {
        "PYTHONPATH": "/path/to/kdcube/app/ai-app/src/kdcube-ai-app"
      }
    }
  }
}
```

The allowlist file is re-read whenever it changes, so edits apply to the
next call without a restart. `WEB_ALLOWLIST` (comma-separated) is the
inline alternative. Entry format:

**Per-user allowlists.** The allowlist belongs to a server process, and
with stdio every user's client launches its own process — so per-user
lists are the natural deployment: give each user (or group) their own
file, owned by the admin and readable by the user, and point that user's
MCP config at it. Different users on the same machine, or different
teams, simply get different files. A shared `http`/`sse` instance
applies one allowlist to everyone it serves; run one instance per
profile when profiles differ.

```
example.org        # example.org and every subdomain
*.example.org      # subdomains only
```

Search needs a search-provider key (Brave by default) in the
environment; see `.env.example`. `use_llm=true` additionally needs a
model key and turns on the neural pipeline — snippet relevance scoring,
content filtering, and objective-guided span segmentation of fetched
pages, each stage on its own configured model role (Haiku-class by
default; see TOOLS.md, "The neural pipeline"). `web_fetch` needs
neither.

## Enforcement notes

- The allowlist is enforced in this server and in the search backend,
  never by instructions to the model.
- With an allowlist configured, `web_fetch`'s archive-mirror fallback
  stays off: an archive host is a different host.
- The page fetcher follows HTTP redirects, so a listed site that
  redirects off-domain can lead outside the allowlist; list only sites
  you trust not to.
