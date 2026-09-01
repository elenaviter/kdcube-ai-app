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

# allowlist: one domain per line, '#' comments allowed
cat > /etc/claude/web-allowlist.txt <<EOF
usgs.gov
noaa.gov
data.census.gov
EOF

PYTHONPATH=$PWD BRAVE_API_KEY=... \
.venv-websearch/bin/python -m \
  kdcube_ai_app.apps.chat.sdk.tools.mcp.web_search.web_search_server \
  --transport stdio --allowlist /etc/claude/web-allowlist.txt
```

Claude Code (from the same directory):

```bash
claude mcp add web-search \
  --env PYTHONPATH=$PWD \
  --env BRAVE_API_KEY=... \
  --env WEB_ALLOWLIST_FILE=/etc/claude/web-allowlist.txt \
  -- $PWD/.venv-websearch/bin/python -m kdcube_ai_app.apps.chat.sdk.tools.mcp.web_search.web_search_server
```

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "web-search": {
      "command": "/path/to/kdcube/app/ai-app/src/kdcube-ai-app/.venv-websearch/bin/python",
      "args": ["-m", "kdcube_ai_app.apps.chat.sdk.tools.mcp.web_search.web_search_server"],
      "env": {
        "PYTHONPATH": "/path/to/kdcube/app/ai-app/src/kdcube-ai-app",
        "BRAVE_API_KEY": "...",
        "WEB_ALLOWLIST_FILE": "/etc/claude/web-allowlist.txt"
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
model key. `web_fetch` needs neither.

## Enforcement notes

- The allowlist is enforced in this server and in the search backend,
  never by instructions to the model.
- With an allowlist configured, `web_fetch`'s archive-mirror fallback
  stays off: an archive host is a different host.
- The page fetcher follows HTTP redirects, so a listed site that
  redirects off-domain can lead outside the allowlist; list only sites
  you trust not to.
