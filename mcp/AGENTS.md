# Agent instruction: setting up the MCP servers in this folder

This file is self-contained on purpose: everything an agent needs to set
up a server for its user is here, no link-following required. The deep
reference (tool contracts, every config knob, the pipeline internals)
lives beside the implementation in
`app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/tools/mcp/web_search/`
(README.md, TOOLS.md, AGENTS.md) — consult it when you need more than
setup.

One server ships today: **web-search** — web search and fetch for MCP
clients behind an egress filter the user owns (allowlist, optional
blocklist — deny wins), enforced server-side. Three tools: `web_search`
(discovery; the model can scope a search WITHIN chosen domains via its
`sites` parameter), `web_fetch` (dereference known URLs),
`allowlist_status` (both lists, exactly as the server enforces them).
An optional neural pipeline (relevance scoring, content filtering,
objective-guided span extraction) runs on the user's model key when
`use_llm=true`; everything else works without any model key.

**How this reaches you:** the user pastes this file's link into a
normal message with their keys and their site list — no special mode.
Any coding agent that can run shell commands can execute it (Claude
Code is one, not the only one). Without such an agent, a person follows
`kdcube/mcp/web-search/README.md` in a terminal — the same procedure
written for a human. Claude Desktop is a consumer of the finished
install, not the installer: it has no shell, but step 5 registers the
server in its config and it then uses the tools like any MCP server.

The install is a one-time action per user (one install directory per
user; that is also how per-user allowlists work). Part 1 below is that
first install and later upgrades; Part 2 is day-2 changes: the
allowlist, the keys.

# Part 1 — First install

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

0. **Install directory.** The user's choice — ask if they did not name
   one, and suggest a sensible default (for example
   `~/mcp-servers/web-search`). Everything below happens inside it.

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

2. **Config.** Start from the shipped template and lock it down:

   ```bash
   cp kdcube/mcp/web-search/config.example.yaml ./config.yaml
   chmod 600 config.yaml
   ```

   Then edit it: fill the allowlist with the sites the user named, and
   the keys the user gave you — never print or log key values, refer to
   them only by name. What the filled file looks like:

   ```yaml
   filter:
     allowlist:            # this file is the live source: editing the
       - example.org       #   lists applies on the next call, no restart.
       - "*.example.net"   # example.org = domain + subdomains;
                           # *.example.net = subdomains only.
     # blocklist:          # optional: always refused, deny wins over
     #   - tracker.example #   the allowlist. Same entry format.
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
   the table); an env var set at launch overrides the file. Filter
   semantics: **allowlist unset = every host allowed; configured = only
   listed hosts, and a configured-but-empty list denies everything.
   Blocklist unset = no host blocked, and deny wins over allow.** The
   model can also scope a single search WITHIN chosen domains via the
   `sites` tool parameter — it narrows inside this filter, never widens
   it.

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
   path named (a blocklisted host would say `denied_by_blocklist`).
   Relay that shape to the user — it is how the tool explains itself. If the user paid for the pipeline, one more call
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

## Upgrading later

The clone is replaceable code; the config is not touched by an upgrade:

```bash
cd <install-dir>/kdcube && git pull
cd .. && .venv/bin/pip install -r kdcube/mcp/web-search/requirements.txt
```

Re-run the offline sanity check (Part 1, step 3), then restart the
server. Nobody manages that process by hand in the stdio setup: the MCP
client spawns it on connect and kills it on exit, so restarting means
reconnecting the client — in Claude Code, `/mcp` and reconnect the
server (or start a new session); in Claude Desktop, quit and reopen the
app. The next spawn runs the pulled code and re-reads the whole config,
keys included. Only an `http`/`sse` deployment has a long-lived process
the operator restarts themselves.

# Part 2 — Day-2 changes

First, find the install: the MCP registration carries the config path.
`claude mcp list` / `claude mcp get web-search` (or the entry in
`claude_desktop_config.json`) shows the `--config <path>` argument —
that file is the single thing to edit, and its directory is the install
dir.

## Changing the allowlist or blocklist

Edit `filter.allowlist` (or `filter.blocklist` — always-refused hosts,
deny wins over allow) in that config.yaml. **The change is live**: the
server re-reads the file on the next call, no restart, no
re-registration. Verify by calling the server's `allowlist_status` tool
(you have it if the server is registered in your session) or by asking
Claude to call it — the reply must show the edited list and
`enforced: true`. When the user asks in plain words ("allow noaa.gov
too", "remove example.org"), this one edit is the whole task.

## Rotating or moving the keys

Keys are applied to the server's environment **at start**, so unlike
the allowlist a key change needs the server respawned — reconnect the
MCP client as described under "Upgrading later" (Claude Code: `/mcp`
reconnect or a new session; Desktop: quit and reopen). Three places a key can live — the file wins
only where the environment is silent, an env var set at launch always
overrides:

1. `services.secrets` in config.yaml (the default; file is mode 600).
2. The MCP registration's env block — `claude mcp add --env
   BRAVE_API_KEY=...` or the `"env"` object in
   `claude_desktop_config.json` — for users who want no keys in
   config.yaml at all.
3. The shell environment, for terminal-launched clients only: an
   `export BRAVE_API_KEY=...` in the user's shell profile reaches a
   server spawned by Claude Code from that terminal. Desktop apps
   launched from the dock do not read shell profiles — use 1 or 2
   there.

## Deployment shapes

Three ways this server serves people, in increasing strength of the
secrets and identity story. Pick deliberately — especially before
answering "run it for the whole team":

1. **stdio, one install per user** (everything above). Simplest, and
   the per-user allowlist model. The keys live on the user's account —
   in config.yaml, the MCP registration's env block, or the shell env —
   and anything the server can read at start, an agent running as the
   same OS user can read too. That is inherent to same-account
   deployment, not a config mistake to fix.
2. **One `http`/`sse` instance run by an admin** (another OS user,
   container, or host). Keys leave the users' machines; user configs
   hold only a URL. **This transport has no caller authorization**:
   whoever reaches the port uses the tool and spends the keys, so it is
   acceptable only behind a real network boundary (localhost or a
   controlled segment), and one instance carries one allowlist — one
   instance per profile when profiles differ. Do not stand this up for
   a team without saying this trade to the user.
3. **Behind KDCube.** The same search and fetch ship as
   `productivity_web_search` / `productivity_web_fetch` on the
   kdcube-services app's productivity MCP surface: keys live in the
   deployment's secrets, callers are authorized as signed-in users with
   per-user consent and accounting, and the allowlist and LLM default
   are platform config (the app descriptor). This is the shape where no
   secret exists on any user machine — its cost is running the KDCube
   app server.

## Rules that hold throughout

- Key values never appear in output, logs, transcripts, or committed
  files. `config.yaml` stays out of every git repo.
- The venv comes from `kdcube/mcp/web-search/requirements.txt` and
  nothing else; the pins are the contract.
- A call can never widen the egress filter — the `sites` parameter
  narrows inside it, and only the user's config changes it. One filter
  per server process: per-user setups are one install dir (config +
  registration) per user.
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
