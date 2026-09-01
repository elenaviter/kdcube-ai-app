# Web search MCP server — launcher

Web search and fetch for MCP clients, behind a domain allowlist the
operator owns, with an optional neural pipeline (relevance scoring,
content filtering, objective-guided span segmentation). This folder is
the short way in; the implementation and the full documentation live in
[`app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/tools/mcp/web_search`](../../app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/tools/mcp/web_search/)
— read its README (prerequisites included), TOOLS.md (tool contracts and
config reference), and AGENTS.md (for an agent doing the setup).

## Quick start

Your install directory owns the config; the clone is just code beside
it:

```bash
mkdir web-search-mcp && cd web-search-mcp    # your install dir
git clone https://github.com/kdcube/kdcube.git repo

python3.11 -m venv .venv    # explicit interpreter, see the README's prerequisites
.venv/bin/pip install -r repo/mcp/web-search/requirements.txt

cp repo/mcp/web-search/config.example.yaml ./config.yaml   # edit: keys, allowlist

.venv/bin/python repo/mcp/web-search/server.py \
  --transport stdio --config $PWD/config.yaml
```

No PYTHONPATH needed: the launcher locates the source tree itself.

Claude Code (from the install dir):

```bash
claude mcp add web-search \
  -- $PWD/.venv/bin/python $PWD/repo/mcp/web-search/server.py --config $PWD/config.yaml
```

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "web-search": {
      "command": "/path/to/web-search-mcp/.venv/bin/python",
      "args": [
        "/path/to/web-search-mcp/repo/mcp/web-search/server.py",
        "--config", "/path/to/web-search-mcp/config.yaml"
      ]
    }
  }
}
```

`config.example.yaml` here is a symlink to the implementation's template
(on Windows checkouts without symlinks, open it to read the real path
and copy from there).
