# Agent instruction: setting up the MCP servers in this folder

This file is self-contained on purpose: everything an agent needs to set
up a server for its user is here, no link-following required. The deep
reference (tool contracts, every config knob, the pipeline internals)
lives beside the implementation in
`app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/tools/mcp/web_search/`
(README.md, TOOLS.md, AGENTS.md) — consult it when you need more than
setup.

One server ships today: **web-search** — web search and fetch for MCP
clients behind a domain allowlist the user owns, enforced server-side.
Three tools: `web_search` (discovery), `web_fetch` (dereference known
URLs), `allowlist_status` (the enforced list, as the server sees it).
An optional neural pipeline (relevance scoring, content filtering,
objective-guided span extraction) runs on the user's model key when
`use_llm=true`; everything else works without any model key.

## The layout you are building

The user's install directory owns the config; the clone is just code
beside it. Never put the config (it holds API keys) inside the clone.

```
<install-dir>/
  config.yaml   # the user's keys + allowlist; keep out of any git repo
  kdcube/       # the clone of this repository, replaceable
  .venv/        # built from kdcube/mcp/web-search/requirements.txt
```

## Setup, step by step

1. **Venv.** Check `which python3` first — on a machine with KDCube
   installed it may resolve into a platform venv, whose package
   versions break this server (an old `mcp` fails to import, a drifted
   `anthropic` silently breaks the pipeline). Use an explicit
   interpreter:

   ```bash
   mkdir <install-dir> && cd <install-dir>
   git clone https://github.com/kdcube/kdcube.git   # creates ./kdcube
   python3.11 -m venv .venv
   .venv/bin/pip install -r kdcube/mcp/web-search/requirements.txt
   ```

2. **Config.** Write `./config.yaml` (mode 600). Fill the allowlist
   with the sites the user named, and the keys the user gave you —
   never print or log key values, refer to them only by name:

   ```yaml
   filter:
     allowlist:            # this file is the live source: editing the
       - example.org       #   list applies on the next call, no restart.
       - "*.example.net"   # example.org = domain + subdomains;
                           # *.example.net = subdomains only.
   services:
     secrets:
       brave:
         api_key: "<the user's Brave key - serves web_search>"
       anthropic:
         api_key: "<the user's Anthropic key - only for use_llm=true>"
     role_models:          # the pipeline's models; Haiku is the
       default:            #   intended class for these roles
         provider: anthropic
         model: claude-haiku-4-5-20251001
       tool.source.reconciler:
         provider: anthropic
         model: claude-haiku-4-5-20251001
       tool.sources.filter.by.content:
         provider: anthropic
         model: claude-haiku-4-5-20251001
       tool.sources.filter.by.content.and.segment:
         provider: anthropic
         model: claude-haiku-4-5-20251001
   cache:
     redis_url: ""         # empty = no cache, fine
   # tls:
   #   cert_file: <certifi cacert.pem>   # only if HTTPS verification
   #                                     # fails, see Troubleshooting
   ```

   The same settings also exist as environment variables (TOOLS.md has
   the table); an env var set at launch overrides the file. Allowlist
   semantics: **unset = every host allowed; configured = only listed
   hosts, and a configured-but-empty list denies everything.**

3. **Offline sanity check** — fakes every network and model call, needs
   no keys, spends nothing. Run it before anything live:

   ```bash
   .venv/bin/pip install pytest
   SRC=$PWD/kdcube/app/ai-app/src/kdcube-ai-app
   PYTHONPATH=$SRC .venv/bin/python -m pytest \
     $SRC/kdcube_ai_app/apps/chat/sdk/tools/mcp/web_search/ \
     $SRC/kdcube_ai_app/apps/chat/sdk/tools/backends/web/test_allowlist.py
   ```

4. **Live check** — this egresses and spends the user's keys, so it
   comes AFTER step 3, sized small. Non-negotiable order: load the
   config, then PROVE the allowlist is enforced, then call. Skipping
   the assert on an unconfigured allowlist means unrestricted egress.

   ```python
   # run with: PYTHONPATH=$SRC .venv/bin/python check.py  (from <install-dir>)
   import asyncio, json
   import kdcube_ai_app.apps.chat.sdk.tools.mcp.web_search.web_search_server as srv

   async def main():
       assert srv.load_config() is not None, "no config.yaml found"
       status = await srv.allowlist_status()
       assert status["enforced"] and status["allowlist_entries"], status

       # one small search, no LLM, snippets only
       rows = await srv.web_search(queries="<something on-topic for the user>",
                                   n=5, fetch_content=False, use_llm=False)
       print(len(rows), "rows, hosts:", {r["url"].split("/")[2] for r in rows})

       # prove the block: one allowed URL, one outside the list
       out = await srv.web_fetch(urls=json.dumps([
           "<an allowed URL>", "https://www.iana.org/"]))
       for url, row in out.items():
           print(url, "->", row.get("status"))

   asyncio.run(main())
   ```

   Expected: every returned host is on the list; the outside URL comes
   back `status: "denied_by_allowlist"` with the host and the config
   path named (relay that shape to the user — it is how the tool
   explains itself). If the user paid for the pipeline, one more call
   with `use_llm=True, n=3` verifies it; expect real model spend.

5. **Register** (this writes to the Claude config — confirm with the
   user if your instructions restrict writes):

   ```bash
   claude mcp add web-search \
     -- $PWD/.venv/bin/python $PWD/kdcube/mcp/web-search/server.py --config $PWD/config.yaml
   ```

   Claude Desktop instead: `claude_desktop_config.json` gets `command`
   = `<install-dir>/.venv/bin/python`, `args` =
   `["<install-dir>/kdcube/mcp/web-search/server.py", "--config",
   "<install-dir>/config.yaml"]`. The launcher needs no PYTHONPATH.
   Always pass `--config` with the absolute path: the MCP client's
   working directory is not the install dir.

## Rules that hold throughout

- Key values never appear in output, logs, transcripts, or committed
  files. `config.yaml` stays out of every git repo.
- The venv comes from `kdcube/mcp/web-search/requirements.txt` and
  nothing else; the pins are the contract.
- A call can never widen the allowlist — only the user's config can.
  One allowlist per server process: per-user setups are one install
  dir (config + registration) per user.
- Known limit worth telling the user: the fetcher follows HTTP
  redirects, so a listed site that redirects off-domain can lead
  outside the list. List trusted sites.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `CERTIFICATE_VERIFY_FAILED`, or pages that open fine in a browser come back `paywall`/`error` | The machine's Python has no CA store wired. Set `tls.cert_file` in config.yaml to the path printed by `.venv/bin/python -m certifi`, retry. Do not touch fetch code. |
| `Can not decode content-encoding: brotli` | The venv was not built from this requirements.txt (it includes `brotli`). Rebuild it. |
| Second consecutive search reports a different provider | Free-tier Brave rate limit; the server fell back to DuckDuckGo transparently. Normal. |
| Empty results with `use_llm=true` and a refinement mode | Failed fetches starve the segmenter (it drops unfetched rows). Check per-URL statuses with `refinement="none"` or `fetch_content=False` first. |
| `from mcp.server import MCPServer` fails | Wrong venv (old `mcp` package). Rebuild from this requirements.txt. |
| Server starts but the allowlist shows `enforced: false` | No config found: pass `--config` with the absolute path, or check the working directory. Unset allowlist = every host allowed — do not leave it this way. |
