# Web search MCP server — launcher

Web search and fetch for MCP clients, behind an egress filter the
operator owns (allowlist, optional blocklist — deny wins), with
per-call site scoping and an optional neural pipeline (relevance
scoring, content filtering, objective-guided span segmentation). This folder is
the short way in; the implementation and the full documentation live in
[`app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/tools/mcp/web_search`](../../app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/tools/mcp/web_search/)
— read its README (prerequisites included), TOOLS.md (tool contracts and
config reference), and AGENTS.md (for an agent doing the setup).

## Quick start

Your install directory owns the config; the clone is just code beside
it:

```bash
mkdir web-search-mcp && cd web-search-mcp    # your install dir
git clone https://github.com/kdcube/kdcube.git   # creates ./kdcube

python3.11 -m venv .venv    # explicit interpreter, see the README's prerequisites
.venv/bin/pip install -r kdcube/mcp/web-search/requirements.txt

cp kdcube/mcp/web-search/config.example.yaml ./config.yaml   # edit: keys, allowlist

.venv/bin/python kdcube/mcp/web-search/server.py \
  --transport stdio --config $PWD/config.yaml
```

No PYTHONPATH needed: the launcher locates the source tree itself.

Claude Code (from the install dir):

```bash
claude mcp add web-search \
  -- $PWD/.venv/bin/python $PWD/kdcube/mcp/web-search/server.py --config $PWD/config.yaml
```

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "web-search": {
      "command": "/path/to/web-search-mcp/.venv/bin/python",
      "args": [
        "/path/to/web-search-mcp/kdcube/mcp/web-search/server.py",
        "--config", "/path/to/web-search-mcp/config.yaml"
      ]
    }
  }
}
```

`config.example.yaml` here is a symlink to the implementation's template
(on Windows checkouts without symlinks, open it to read the real path
and copy from there).

## Cheatsheet

Things to say to Claude once the server is connected:

| Say | What happens |
| --- | --- |
| "Call allowlist_status and show me my whitelist." | Both lists, their sources, and the SSRF guard state — the quickest proof you're talking to your server. |
| "Search the web for `<topic>`." | `web_search` over your allowlist; results carry title, url, snippet, and content when fetched. |
| "Search for `<topic>` only on en.wikipedia.org." | The `sites` scoping: the provider searches within that domain, so the whole result page is on-site. |
| "Fetch `<url>` and summarize it." | `web_fetch` dereferences the exact URL — text, dates, status. |
| "Search without the LLM steps." | `use_llm=false`: cheaper, no model spend, provider ranking only. |
| "Why was that site refused?" | The denial itself says: `denied_by_allowlist` (host not on your list), `denied_by_blocklist` (host you banned), `denied_by_ssrf_guard` (private/loopback/metadata address — refused whatever the lists say). |

Operator quick moves (config.yaml sits at the top of your install dir):

| Do | How |
| --- | --- |
| Allow or ban a site | Edit `filter.allowlist` / `filter.blocklist` in config.yaml — **live on the next call**, no restart. Ask your coding agent, or use any text editor. If you opted into `filter.expose_edit_tool` at setup, Claude itself can do it from any client, Desktop included ("allow noaa.gov too"); without the opt-in no tool edits the lists by design. |
| Rotate a key | Edit `services.secrets`, then respawn the server: Claude Code `/mcp` reconnect or new session, Desktop quit and reopen. |
| Upgrade | `git pull` in `kdcube/`, re-run pip on the same requirements.txt, respawn. Your config is untouched. |
| Internal-network deployment | `filter.ssrf_guard: false` — only then are private hosts fetchable, and you own that trade. |
| See what it does | The server narrates to stderr, and your MCP client keeps it: Claude Desktop in `~/Library/Logs/Claude/mcp-server-web-search.log` (`tail -f` it during a search: filter drops, pipeline models, denials with reasons), Claude Code via `/mcp` → the server's logs. Every tool result also explains itself in-band — "why was that refused?" works. |
