# MCP servers

Standalone MCP servers shipped in this repo, launchable from here
without PYTHONPATH. Each launcher defers to its implementation inside
the source tree, where the full documentation lives.

**Setting one up with an agent?** Point it at [`AGENTS.md`](AGENTS.md)
in this folder — the complete self-contained instruction (layout, venv,
config with keys and the allowlist/blocklist, offline sanity, safe live
check, registration, troubleshooting), no link-following needed.

| Server | Launcher | Documentation |
| --- | --- | --- |
| Web search + fetch behind an operator-owned egress filter (allowlist, optional blocklist — deny wins) with an SSRF guard underneath, per-call site scoping, opt-in conversational list editing, and an optional neural pipeline (relevance scoring, content filtering, span segmentation) | [`web-search/server.py`](web-search/server.py) | [README](../app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/tools/mcp/web_search/README.md) · [TOOLS](../app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/tools/mcp/web_search/TOOLS.md) · [AGENTS](../app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/tools/mcp/web_search/AGENTS.md) |
