#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Repo-root launcher for the web search MCP server.

Runs the server that lives in
``app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/tools/mcp/web_search``
without needing PYTHONPATH: the source root is located relative to this
file and put on sys.path, then the real module's ``main()`` runs. Same
CLI (``--transport``, ``--config``, ``--allowlist``, ``--host``,
``--port``); the documentation lives beside the implementation
(README.md, TOOLS.md, AGENTS.md in that folder).
"""

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _REPO_ROOT / "app" / "ai-app" / "src" / "kdcube-ai-app"
sys.path.insert(0, str(_SOURCE_ROOT))

from kdcube_ai_app.apps.chat.sdk.tools.mcp.web_search.web_search_server import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
