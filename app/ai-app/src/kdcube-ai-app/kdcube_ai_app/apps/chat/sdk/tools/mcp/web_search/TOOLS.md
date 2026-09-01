# Tool documentation: web_search, web_fetch, allowlist_status

The contracts of the three tools this MCP server exposes, how each works
inside, and every config knob. The [README](README.md) covers setup for
Claude Code and Claude Desktop; this page is the reference.

## How the server works

`web_search_server.py` is a thin MCP wrapper over the platform's web
backends:

- `web_search` delegates to `search_backends.web_search` (the same
  orchestrator KDCube's resident agents use): provider search across the
  query variants, deduplication, optional LLM snippet reconciliation,
  optional content fetch, optional LLM content refinement.
- `web_fetch` delegates to `fetch_backends.fetch_url_contents`: direct
  dereference of known URLs with article-style text extraction and date
  metadata.
- The **egress filter** (`../../backends/web/allowlist.py`: allowlist
  plus blocklist, deny wins) is applied server-side in both paths:
  search results from refused hosts are dropped inside the orchestrator
  after deduplication and **before any content fetch** (no egress to
  refused hosts from the fetch stage), and `web_fetch` denies a refused
  host per URL before any request. A call cannot widen the filter — the
  `sites` parameter narrows inside it — only the operator's config can.
- The **SSRF guard** (`../../backends/web/ssrf_guard.py`, ported from
  the community's PR #1) is the address-level layer under the name-level
  filter: private, loopback, link-local (cloud metadata at
  169.254.169.254 included), CGNAT, multicast, and reserved addresses,
  plus metadata-style hostnames (`localhost`, `*.internal`, `*.local`,
  `metadata.google.internal`), are refused regardless of the lists.
  Enforced twice: a per-URL pre-check in the fetch core (IP literals,
  clean `denied_by_ssrf_guard` results), and a guarded DNS resolver on
  the fetcher's connector that validates every DNS answer at connect
  time — so hostname redirect targets and DNS-rebinding answers are
  checked with the exact IPs a connection would use. Default ON;
  `filter.ssrf_guard: false` (or `WEB_SSRF_GUARD=off`) disables it for
  deployments that must fetch internal hosts.
- The **LLM steps are optional**. `use_llm=false` (search) runs the
  pipeline with no model calls and no model keys; `web_fetch` defaults to
  `use_llm=false`. What the LLM adds when on is the neural pipeline
  below: snippet relevance scoring against the objective, dropping
  clearly irrelevant results, and objective-guided content refinement.

## The neural pipeline

With `use_llm=true` the search pipeline runs up to three model stages,
each on its own configured role:

| Stage | Role | What it does |
| --- | --- | --- |
| Snippet reconciler | `tool.source.reconciler` | Reads the search snippets against the objective and queries, scores each source (`objective_relevance` and `query_relevance`, 0..1), drops clearly irrelevant ones, ranks the rest. Runs before any content fetch. |
| Content filter | `tool.sources.filter.by.content` | Filter-only pass over fetched page content: which sources actually answer the objective. Used when segmentation is off. |
| Filter + segmenter | `tool.sources.filter.by.content.and.segment` | Two phases in one call: filters the fetched sources by content, then extracts the spans of each kept page that serve the objective. The code trims the page to those spans with context margins around each (the refinement mode sets the target coverage: `balanced` 50-70%, `recall` 80-95%, `precision` 20-50%). Spans that cover the whole page leave it intact; spans that fail to match keep the original text, so refinement degrades toward keeping content. |

A verified run on this setup: two encyclopedia pages of 21,648 and
19,463 extracted characters came back as 2,203 and 1,925 characters of
objective-targeted content, with relevance scores on every row — that
reduction is what the pipeline is for.

**Role configuration.** Every role resolves to an explicit
provider+model pair. Pin all three in `ROLE_MODELS_JSON` (see
`.env.example`, which pins them to Haiku — `claude-haiku-4-5-20251001` —
the intended model class for these roles: fast, cheap, strong enough for
scoring, filtering, and span extraction). A role left unpinned falls
back to `DEFAULT_LLM_MODEL_ID`. Model ids, their aliases, and their
prices live in the deployment's price table — in this repo,
`app/ai-app/deployment/economics.yaml` (the `claude-haiku-4-5-20251001`
entry is there with its per-token and cache pricing); a KDCube
deployment carries its own copy of that table.

**Failure semantics.** Reconciler failure returns the raw ranked rows
(nothing lost). Content-filter failure keeps every source. The
filter+segment stage runs only when MORE THAN ONE fetched row survives
the egress filter: a single-row result passes through unrefined with no
model spend (logged, not marked in the result). And a caveat on the
scores: `objective_relevance`/`query_relevance` carry signal only when
the reconciler actually ran — on backends whose snippet reconciler is
off by default (Brave), rows get 1.0 as a neutral placeholder, so a 1.0
proves nothing about the pipeline. The
filter+segment stage is stricter: a failed or empty span response drops
the unsegmented sources from the result, and in refinement mode rows
whose content fetch failed are dropped as well — with an allowlist this
means a starved fetch stage can empty the result, so if results vanish
with `use_llm=true` and a refinement mode, check fetch statuses first
(`fetch_content=false` or `refinement="none"` shows the undropped rows).

Two platform knobs reach this pipeline through assembly config when run
inside KDCube: `web_search_segmenter` (`fast` is the current default)
and `web_search_agentic_thinking_budget`.

## web_search

Discovery tool: use it to FIND pages. Prefer at most 2 query variants per
call.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `queries` | required | Array of rephrases/synonyms, or a single query string. |
| `objective` | null | The goal/question; drives relevance scoring and refinement when `use_llm=true`. |
| `refinement` | `balanced` | Post-fetch content refinement, needs `use_llm=true`: `none` full pages; `balanced` target + context (50-70%); `recall` bodies, minimal chrome (80-95%); `precision` direct answers only (20-50%, needs objective). |
| `n` | 8 | Max unique results, 1-20. Prefer max 5. |
| `fetch_content` | true | Fetch page content for the results. False = ranked snippets/URLs only (cheaper; fetch selected URLs yourself with `web_fetch`). |
| `include_binary_base64` | true | Attach base64 for binary/image/PDF results within size limits. |
| `freshness` | null | `day` \| `week` \| `month` \| `year`. |
| `country` | null | ISO2, e.g. `DE`, `US`. |
| `safesearch` | `moderate` | `off` \| `moderate` \| `strict`. |
| `sites` | null | Per-call site scoping: the provider query is rewritten with `site:` operators so the search runs WITHIN these domains (up to 8). Use it when you know where the answer lives. Clamped by the operator's egress filter — it narrows, never widens; if every requested site is excluded, the call fails with the reasons named. |
| `use_llm` | true | LLM reconciliation + refinement on/off. With false, no model keys are needed. |

Result: array of rows `{title, url, text, objective_relevance?,
query_relevance?, content?, mime?, base64?, size_bytes?, fetched_time_iso,
published_time_iso?, ...}`. `text` is the search snippet, `content` the
fetched page text when fetch ran. Non-HTML supported files carry
`mime`/`base64` instead of `content`. With the egress filter configured,
rows from refused hosts are already gone.

## web_fetch

Dereference-only: use it when you already hold concrete HTTP/HTTPS URLs.
It never searches. Skip URLs whose `web_search` row already carries usable
`content`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `urls` | required | Array of absolute URLs or a single URL string. |
| `objective` | null | Enables refinement when `use_llm=true`; without it content stays full. |
| `refinement` | `none` | Same modes as web_search; needs `use_llm=true` and an objective. URLs are never dropped by refinement; pages without reliable spans keep full content. |
| `max_content_length` | -1 | Max characters of cleaned content per URL, truncated at a sentence boundary. -1 = no limit. |
| `include_binary_base64` | true | Attach base64 for binary/PDF fetches within size limits. |
| `use_archive_fallback` | false | Try an archive mirror for blocked/paywalled pages. Forced off while the egress filter is configured: an archive host is a different host. |
| `use_llm` | false | Enables the refinement path (builds the model service). |

Result: JSON object mapping each input URL to
`{status, content?, content_length?, published_time_iso?,
modified_time_iso?, date_method?, date_confidence?, error?}`. Statuses
include `success`, `timeout`, `paywall`, `error`, `non_html`,
`blocked_403`, `http_XXX`, `pdf_redirect` — and `denied_by_allowlist` /
`denied_by_blocklist` / `denied_by_ssrf_guard`, whose entries name the
denied host and the reason while the other URLs in the same call still
fetch.

## allowlist_status

No parameters. Returns `{allowlist_source, allowlist_entries,
entry_count, blocklist_source, blocklist_entries, blocklist_count,
ssrf_guard, enforced}` — the egress filter exactly as the server
enforces it, so the model and the operator read the same truth. Deny
wins: a blocklisted host is refused even when the allowlist admits it.

## Dependencies

`requirements.txt` in this folder is the complete list beyond the
standard library and this repo on `PYTHONPATH` — a clean venv with
exactly that set runs the server and both tools end to end. The heavier
entries exist because the server reuses the platform's real backends:
`langchain-core`/`langchain-openai` and the model-key plumbing come with
the model service (only exercised on `use_llm=true`), `connection-hub`
comes with the config module, `redis` is imported by the optional cache
(the tool's cache stays disabled while `REDIS_URL` is empty; a shared
client object may still be constructed lazily by platform imports, but
nothing is read or written through it), and `ddgs` is the DuckDuckGo
fallback backend.

## Configuration

Two modes, same settings:

**YAML (recommended).** One structured file holds everything —
`config.example.yaml` is the copyable template. The config's home is
the operator's install directory, beside the clone rather than inside
it. The server loads it from `--config PATH`, or `WEB_SEARCH_CONFIG`,
or a `config.yaml` in the working directory, or (the in-repo
development case) one beside the server file; launch configs should
pass the explicit `--config` with an absolute path. Sections: `filter` (`allowlist` and `blocklist` inline — the
config file itself is then the live source, edits to the lists apply on
the next call — or `allowlist_file`/`blocklist_file`), `services.secrets` (per-provider
`api_key` blocks, the exact shape a KDCube deployment's secrets.yaml
nests under `services:`, so the block carries over verbatim),
`services.role_models` (the pipeline roles pinned as provider+model
pairs, with `default` covering every role not pinned), `cache`
(`redis_url`, `ttl_seconds`), `server` (`host`/`port` for http/sse),
`kdcube` (a deployment's `assembly_yaml`/`global_secrets_yaml` as the
key source instead of inline keys), `tls` (`cert_file`).

**Environment variables.** The same settings as raw variables —
`.env.example` is the copyable template; the table below is the
reference. This is the natural mode for MCP client configs (Claude
Desktop's `env` block) and CI. The two modes compose: YAML values are
applied onto the process environment, and a variable already set in the
environment wins over the file.

CLI flags: `--config`, `--allowlist`, `--transport`/`--host`/`--port`.

| Variable | Purpose |
| --- | --- |
| `WEB_ALLOWLIST_YAML` | Path to a YAML file whose `filter.allowlist` holds the entries (set automatically when the config.yaml carries an inline allowlist). Re-read whenever its mtime changes — edits apply to the next call without a restart. |
| `WEB_ALLOWLIST_FILE` | Path to a plain allowlist file: one domain per line, blank lines and `#` comments ignored. Also re-read on change. |
| `WEB_ALLOWLIST` | Inline comma-separated entries; fixed for the process. The file sources take precedence. |
| `WEB_BLOCKLIST_YAML`, `WEB_BLOCKLIST_FILE`, `WEB_BLOCKLIST` | The blocklist's three sources, same mechanics and same entry format. A blocklisted host is refused even when the allowlist admits it; unset = no host blocked. |
| `WEB_SSRF_GUARD` | The address-level SSRF guard (`filter.ssrf_guard` in YAML). Default on; `off` disables it for deployments that must fetch internal hosts. |
| `WEB_FILTER_EDIT_TOOL` | Exposes the `site_filter_edit` tool (`filter.expose_edit_tool` in YAML). Default off; set only on the user's explicit, trade-stated yes. |
| `BRAVE_API_KEY` | Search provider key (Brave is the default backend). |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY`, `DEFAULT_LLM_MODEL_ID`, `ROLE_MODELS_JSON` | Model service, needed only for `use_llm=true` calls. |
| `REDIS_URL`, `WEB_SEARCH_CACHE_TTL_SECONDS` | Optional result cache. Leave `REDIS_URL` empty to run without one. |
| `MCP_SERVER_HOST`, `MCP_SERVER_PORT` | Binding for `http`/`sse` transports (stdio needs neither). |
| `WEB_SEARCH_LOG_LEVEL` | Server narration level (`server.log_level` in YAML), default INFO. Logs go to stderr — filter drops, pipeline models, denials — and the MCP client keeps them (Claude Desktop: `~/Library/Logs/Claude/mcp-server-<name>.log`). stdout carries only JSON-RPC. |
| `ASSEMBLY_YAML_DESCRIPTOR_PATH`, `GLOBAL_SECRETS_YAML` | Optional KDCube-deployment lane: point them at a deployment's `assembly.yaml` / `secrets.yaml` and keys resolve from there (`services.brave.api_key`, model keys) instead of individual env vars. |
| `SSL_CERT_FILE` | CA bundle for verifying HTTPS certificates when fetching pages (`tls.cert_file` in YAML). Needed only on machines whose Python has no working CA store — point it at certifi's `cacert.pem`; otherwise leave unset. |

Entry semantics, shared by both lists
(`../../backends/web/allowlist.py`):

```
example.org        # example.org and every subdomain
www.example.org    # that exact host and its subdomains
*.example.org      # subdomains only, never the bare domain
```

Matching is case-insensitive on the URL hostname; ports don't
participate. The same entry format serves the blocklist. **Allowlist
unset** = every host allowed; **configured but empty** = every host
denied: the server is closed until the operator lists what is allowed.
**Blocklist unset** = no host blocked, and deny always wins over allow.

**Scope: one filter per server process.** With stdio each user's
client launches its own process, so per-user filters are a config per
user (admin-owned, user-readable), pointed at by that user's MCP
registration. A shared `http`/`sse` instance applies its one filter to
every caller — run separate instances for separate profiles.

## site_filter_edit (operator-enabled)

Registered only when the operator sets `filter.expose_edit_tool: true`
(env `WEB_FILTER_EDIT_TOOL`), a decision the setup procedure asks the
user explicitly, with the trade stated: once exposed, anything that can
call tools can edit the lists. Parameters: `list_name`
(`allowlist`|`blocklist`), `add`, `remove` (arrays or single domain
strings). Edits go into the live YAML config textually — comments and
everything else stay byte-identical — and apply on the next call.
Entries must look like domains; anything else is refused, as are
file-sourced lists and flow-style values, each with the reason named.
Returns `{ok, edited, entries, status}` or `{ok: false, error}`. The
SSRF guard is not editable through this or any tool.

## Accounting events

The search and fetch backends emit usage accounting events on every
call: the search provider and its pricing tier per query batch, and the
model usage of the LLM steps when `use_llm=true`. Inside a KDCube
deployment these land in the deployment's accounting storage and show up
in its usage reporting. Standalone, no accounting storage is bound, so
the events are dropped — nothing is written and nothing leaves the
process; the log lines about accounting context are harmless. Operators
who want a local record can bind the accounting system's file storage
before serving (see `kdcube_ai_app/infra/accounting/`).

## Enforcement model

The allowlist is enforced in code, on the server side — never by
instructions to the model. A denial is explained in-band (host, reason,
allowlist source) so the agent can tell the user what the operator would
need to change. Known limitation: the page fetcher follows HTTP
redirects, so a listed site that redirects off-domain can lead outside
the allowlist; list only sites you trust not to.
