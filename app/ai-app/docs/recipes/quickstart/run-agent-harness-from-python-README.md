---
id: repo:kdcube-ai-app/app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md
title: "Recipe: Run the Agent Harness from Python"
summary: "Run native ReAct, LangGraph, or Claude Code from SDK source with web research, isolated code execution, document rendering, durable conversations, and accounting."
status: current
tags: ["recipe", "quickstart", "agent-harness", "python", "native-react", "langgraph", "claude-code", "self-hosted", "web-search", "rendering"]
keywords: ["run agent harness", "direct SDK agent", "KDCube Web Search", "standard descriptors", "Redis", "Postgres", "Git transcript", "isolated code execution", "write_pdf", "write_docx", "write_pptx"]
updated_at: 2026-09-06
see_also:
  - repo:kdcube-ai-app/agents/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/direct-agent-instruction-profiles-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/arch/security-and-trust-model-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/configuration/assembly-descriptor-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/configuration/secrets-descriptor-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/exec/README-iso-runtime.md
  - repo:kdcube-ai-app/mcp/web-search/README.md
---
# Recipe: Run the Agent Harness from Python

Run an agent from a shell or IDE and give it KDCube Harness capabilities. The
sample process imports KDCube SDK source directly; it does not require a
running KDCube server. Redis and Postgres are independent support services.

Choose one complete example:

```text
agents/
  native/       KDCube ReactSolverV2
  langgraph/    LangGraph through KDCubeChatModel
  claude/       Claude Code through ClaudeCodeAgent
```

Each directory owns its agent, requirements, YAML behavior, platform
descriptors, skill, Compose services, and setup command.

## The user journey

```text
Open agents/
     |
     +---- native
     +---- langgraph
     +---- claude
              |
              v
       create .venv
       install requirements + Chromium
              |
              v
       create local descriptors
       start Redis + Postgres
       build isolated executor image
              |
              v
       --check -> --infra-check -> run
              |
              v
       research request is stored as an attachment
              |
              v
       web research -> next-turn continuity
              |
              v
       agent-authored Python -> isolated Docker execution
              |                         |
              |                         +-> XLSX
              |                         +-> renderer-ready HTML
              |                         +-> archived pkg/user_code.py
              v
       rendering_tools.write_pdf -> polished PDF
              |
              v
       inspect conversation, files, execution, and spend evidence
```

The model writes the research and Python. KDCube supplies the reusable
execution, rendering, persistence, accounting, and communicator boundaries.
This is the important document boundary: the model authors the content and
visual structure, while stable tools own PDF, DOCX, and PPTX conversion. The
agent can produce polished files without inventing a document-generation
program from scratch on every turn.

## 0. Prerequisites

Install Git, Python 3.11, and Docker Engine or Docker Desktop with Compose.

You also need one model interface:

- Native or LangGraph uses a provider API key.
- Claude uses an authenticated Claude Code CLI or an Anthropic key.

## 1. Get the source

```bash
git clone https://github.com/kdcube/kdcube.git
cd kdcube
export KDCUBE_SOURCE="$PWD"
```

`KDCUBE_SOURCE` is only a shell path used by this recipe. Runtime behavior is
descriptor-owned.

## 2. Choose one agent

```bash
export AGENT=native
cd "$KDCUBE_SOURCE/agents/$AGENT"
```

Set `AGENT` to `native`, `langgraph`, or `claude`. All remaining commands run
inside that selected directory unless a step says otherwise.

## 3. Install the Python process and renderer

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m playwright install chromium
cp config.template.yaml config.local.yaml
```

The requirements install this repository's SDK in editable mode. Chromium is
required by the default PDF demonstration and other browser-backed render
paths. DOCX consumes Markdown; PPTX consumes section-based HTML.

## 4. Create local descriptors

For Native or LangGraph:

```bash
.venv/bin/python setup_local.py --provider anthropic
```

Enter the provider key at the hidden prompt. For Claude authenticated through
its CLI:

```bash
.venv/bin/python setup_local.py --provider none
```

The setup command creates ignored local files:

```text
descriptors.local/
  assembly.yaml       model, Redis, Postgres, storage, executor, optional Git
  secrets.yaml        provider, Redis, Postgres, optional Git credentials
  economics.yaml      model prices and economics policy
  gateway.yaml        empty for this direct process
.env                   matching credentials for the two Compose services
```

Claude setup also initializes `output/claude-session-store.git`. These are
ordinary app-agnostic platform descriptors; this direct process does not need
`bundles.yaml`.

## 5. Build the isolated Python executor

The demonstration asks the model to author Python using `openpyxl`. That code
runs inside the executor image, not in the sample process.

```bash
cd "$KDCUBE_SOURCE"
docker build \
  -t py-code-exec:latest \
  -f app/ai-app/deployment/docker/all_in_one_kdcube/Dockerfile_Exec \
  app/ai-app
docker image inspect py-code-exec:latest >/dev/null
cd "$KDCUBE_SOURCE/agents/$AGENT"
```

The shipped descriptor sets `network_mode: none`. Research happens through
the governed Web Search tool; the agent passes selected findings into its
generated program.

## 6. Start Redis and Postgres

```bash
docker compose --env-file .env -f compose.yaml up -d --wait
docker compose --env-file .env -f compose.yaml ps
```

Both services must report healthy. The harness creates its conversation
tables. LangGraph also creates its checkpoint tables.

## 7. Inspect the agent YAML

`config.local.yaml` selects behavior, tools, skills, and the local output root.
This is the Native shape; LangGraph and Claude use framework-specific tool IDs
for the same capabilities:

```yaml
agent:
  topic: "the current stable Python release and its release date"
  instructions:
    profile: lite:core
  additional_instructions: |
    You are a research agent. Preserve public source URLs and follow enabled skills.
  run_directory: ./output
  tools:
    - id: web_tools.web_search
      enabled: true
      runtime: local
      settings:
        filter:
          allowlist:
            - python.org
          blocklist: []
          ssrf_guard: true
    - id: exec_tools.execute_code_python
      enabled: true
      runtime: docker
    - id: rendering_tools.write_pdf
      enabled: true
      runtime: local
    - id: rendering_tools.write_docx
      enabled: true
      runtime: local
    - id: rendering_tools.write_pptx
      enabled: true
      runtime: local
  skills:
    root: ./skills
    enabled:
      - demo.research-brief
```

The corresponding IDs are:

| Capability | Native | LangGraph | Claude Code |
| --- | --- | --- | --- |
| Web Search | `web_tools.web_search` | `web_search` | `mcp__kdcube_web_search__web_search` |
| Isolated Python | `exec_tools.execute_code_python` | `execute_python` | `mcp__kdcube_harness__execute_python` |
| PDF | `rendering_tools.write_pdf` | `write_pdf` | `mcp__kdcube_harness__write_pdf` |
| DOCX | `rendering_tools.write_docx` | `write_docx` | `mcp__kdcube_harness__write_docx` |
| PPTX | `rendering_tools.write_pptx` | `write_pptx` | `mcp__kdcube_harness__write_pptx` |

Unknown tool or skill IDs fail during `--check`. Tool settings stay on their
tool row. The Web Search row owns its domain allowlist, blocklist, and SSRF
policy. Change the allowlist when changing the sample topic.

The Native profile is `lite:core`; the SDK adds the existing ReAct exec,
rendering, and web blocks only when their tool families are enabled. LangGraph
and Claude use `workspace-files`, a framework-neutral profile that teaches the
direct current-turn artifact workspace without ReAct channel syntax. In all
three examples, `additional_instructions` is product-specific administrator
text appended after the standard profile, enabled-capability guidance, and
skills. It does not replace them.

```text
profile -> workspace/conduct -> enabled tool guidance -> skills
                                                        |
                                                        v
                                      additional_instructions (last)
```

Claude writes the composed profile to `CLAUDE.md` and materializes selected
skills under `.claude/skills`. LangGraph includes selected skill text in its
system prompt. Native keeps the normal ReAct skill gallery and strict protocol.
See
[Direct Agent Instruction Profiles](../../runtime/harness/direct-agent-instruction-profiles-README.md)
for supported Native profile IDs and the exact composition contract.

`descriptors.local/assembly.yaml` selects independent services and durable
storage:

```yaml
storage:
  kdcube: ../output/kdcube-storage

models:
  default_llm_model_id: claude-haiku-4-5-20251001

platform:
  services:
    proc:
      exec:
        py_code_exec_image: py-code-exec:latest
        py_code_exec_network_mode: none
```

Credentials live in `descriptors.local/secrets.yaml`. Model prices and
economics policy live in `descriptors.local/economics.yaml`. The three samples
use separate default ports, so their Compose projects can run independently.

## 8. Check and run

```bash
.venv/bin/python agent.py --check
.venv/bin/python agent.py --infra-check
.venv/bin/python agent.py
```

`--check` constructs the adapter, exact tool inventory, and skills without
contacting a provider or support service. `--infra-check` verifies Redis,
Postgres tables, KDCube storage, the executor image, and Playwright Chromium
without model spend. Claude also verifies its Git transcript store.

The full two-turn scenario:

1. Stores `research-request.md` as a user attachment and performs Web Search.
2. Continues the conversation, authors Python, and invokes isolated execution.
3. The program creates an XLSX evidence table and print-ready HTML.
4. The agent invokes `write_pdf`; KDCube renders and hosts the final PDF.

A successful run ends with:

```text
demonstration: PASS
```

## 9. Inspect durable evidence

Find the run index and deliverables:

```bash
find output/runs -type f \
  \( -name evidence.json -o -name research-brief.pdf -o -name research-data.xlsx \) \
  -print
```

Each `output/runs/<conversation>/evidence.json` is a navigation index. It points
to authoritative records under the configured `storage.kdcube` root:

```text
output/kdcube-storage/
  accounting/<tenant>/<project>/...                    durable usage events
  cb/tenants/<tenant>/projects/<project>/conversation/ turn-log payloads
  cb/tenants/<tenant>/projects/<project>/attachments/  inputs and produced files
  cb/tenants/<tenant>/projects/<project>/executions/   compressed turn workspaces
```

Inspect the generated program retained with the execution:

```bash
ARCHIVE="$(find output/kdcube-storage -path '*/executions/*' -name '*.zip' | sort | tail -1)"
unzip -l "$ARCHIVE"
unzip -p "$ARCHIVE" pkg/user_code.py | sed -n '1,220p'
```

The execution ZIP also contains the turn's `out/` tree. Postgres holds the
conversation index and turn metadata; LangGraph stores its graph checkpoints
there as well. Redis holds a live per-turn accounting mirror with a TTL. The
durable accounting JSON files remain in KDCube storage after that mirror
expires.

The local `communicator.jsonl` shows what the adapter streamed into the harness.
It is diagnostic evidence, not the durable conversation authority.

## 10. Try the other document operations

The default run demonstrates HTML-to-PDF because it gives an immediate visual
result. The same `rendering_tools` module is already selected in YAML:

- `write_pdf` renders HTML to PDF through Playwright and Chromium.
- `write_docx` renders structured Markdown to DOCX.
- `write_pptx` renders one HTML `<section>` per slide to PPTX.

To demonstrate DOCX or PPTX, alter the second prompt in `agent.py` so the
generated Python also contracts an internal `.md` or section-based `.html`
source, then call the matching renderer with an external output path. The
model authors content and structure; the reusable renderer owns the document
format instead of forcing every model to rebuild formatting code.

## 11. Understand continuity by adapter

Native adds a third turn in a different conversation and requires
`react.memsearch` to recover the earlier research for the same user.

LangGraph rebuilds its graph with turn-bound tools on every invocation while
reusing the same Postgres checkpoint thread. The fresh tool binding prevents a
later turn from writing into an earlier turn's artifact root.

Claude resumes its CLI transcript from a per-conversation Git branch. Inspect
the local branch with:

```bash
git --git-dir output/claude-session-store.git for-each-ref \
  --format='%(refname)' refs/heads/kdcube/claude/
```

To use a private remote, set `storage.claude_code_session.repo` in
`descriptors.local/assembly.yaml` and the matching
`platform.services.git.http_token` credential in `descriptors.local/secrets.yaml`.

## 12. Change the agent

Use `config.local.yaml` to change the instruction profile,
`additional_instructions`, topic, tool selection, skills, limits, and settings
attached to each tool row. Use the standard descriptors
to change model, infrastructure, storage, economics, executor, and secrets.

Native trusted tool sources live in `agent.py`'s `TOOL_SOURCES`. LangGraph
adapters live in `tools.py`. Claude's runner writes the selected stdio MCP
servers and exact allowed tool IDs into its workspace for each turn. Rerun
`--check` after any tool or skill change.

## 13. Know the boundary

This direct-host path gives one Python process model accounting, durable
conversation records, communicator events, skills, attachments, output
hosting, isolated Python, and document rendering. The process owner controls
the selected tools.

For a multi-user runtime with chat UI, authentication, managed tools,
tool-execution enforcement, delegated consent, isolated workspaces,
rate/spend policy, and app hosting, follow
[Quick Start: Run KDCube Locally](../../quick-start-README.md).

## 14. Stop services

```bash
docker compose --env-file .env -f compose.yaml down
```
