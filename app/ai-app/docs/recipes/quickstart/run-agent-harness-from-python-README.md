---
id: repo:kdcube-ai-app/app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md
title: "Recipe: Run the Agent Harness from Python"
summary: "Executable steps for running a YAML-configured native ReAct, LangGraph, or Claude Code agent from KDCube SDK source."
status: current
tags: ["recipe", "quickstart", "agent-harness", "python", "native-react", "langgraph", "claude-code", "self-hosted", "web-search"]
keywords: ["run agent harness", "direct SDK agent", "KDCube Web Search", "standard descriptors", "Redis", "Postgres", "Git transcript", "isolated code execution"]
updated_at: 2026-09-05
see_also:
  - repo:kdcube-ai-app/agents/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/arch/security-and-trust-model-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/configuration/assembly-descriptor-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/configuration/secrets-descriptor-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/exec/README-iso-runtime.md
  - repo:kdcube-ai-app/mcp/web-search/README.md
---
# Recipe: Run the Agent Harness from Python

Run one agent directly from a shell or IDE. The process imports the KDCube SDK
from source; it does not require a running KDCube server. Redis and Postgres are
independent support services.

Choose one self-contained directory:

```text
agents/
  native/       KDCube ReactSolverV2
  langgraph/    LangGraph through KDCubeChatModel
  claude/       Claude Code through ClaudeCodeAgent
```

Each directory contains its agent, requirements, behavior YAML, KDCube Web
Search policy, standard platform descriptors, skill, Compose services, and
setup command.

## The first-run journey

```text
Developer opens agents/
          |
          v
Choose one agent core
   |          |          |
   v          v          v
 native    langgraph    claude
   |          |          |
   +----------+----------+
              |
              v
Enter that agent's self-contained directory
              |
              v
Create a Python environment and install requirements
              |
              v
Create local YAML configuration and descriptors
              |
              v
Start the directory's Redis and Postgres services
              |
              v
Run --check, then --infra-check
              |
              v
Run the multi-turn demonstration
              |
              v
Web research -> retained context -> PDF + XLSX
              |
              v
Inspect output files, communicator events, and accounting
              |
              v
Change instructions, model, tools, skills, or limits in YAML
              |
              +-----------------------------+
              |                             |
              v                             v
Keep running the direct SDK agent    Adopt the full KDCube runtime
                                     for UI, authentication, governed
                                     tools, isolated workspaces, and
                                     application hosting
```

The direct path imports KDCube SDK source but does not require a running
KDCube server. The selected directory is the complete first-use boundary;
reusable hosting code remains in the SDK rather than appearing as another
choice beside the three agents.

## 0. Prerequisites

Install Git, Python 3.11, and Docker Engine or Docker Desktop with Compose.

You also need one model interface:

- native or LangGraph: a provider API key;
- Claude: an authenticated Claude Code CLI or Anthropic key.

## 1. Get the source

```bash
git clone https://github.com/kdcube/kdcube.git
cd kdcube
export KDCUBE_SOURCE="$PWD"
```

## 2. Choose an agent

Set one value:

```bash
export AGENT=native       # or langgraph / claude
cd "$KDCUBE_SOURCE/agents/$AGENT"
```

The rest of the basic procedure stays in this directory.

## 3. Install it

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
cp config.template.yaml config.local.yaml
```

The requirements install the SDK from
`app/ai-app/src/kdcube-ai-app` in editable mode.

## 4. Create local descriptors

For native or LangGraph:

```bash
.venv/bin/python configure.py --provider openai
```

Enter the provider key at the hidden prompt. For Claude authenticated through
its CLI:

```bash
.venv/bin/python configure.py --provider none
```

The command creates ignored local files in this agent directory:

```text
descriptors.local/
  assembly.yaml       infrastructure, storage, model, optional Git/executor
  secrets.yaml        provider, Redis, Postgres, optional Git credentials
  economics.yaml      model prices and economics policy
  gateway.yaml        empty for this direct process
.env                   matching Compose service credentials
```

The tracked `web-search.yaml` beside `config.template.yaml` owns the KDCube
Web Search allowlist, blocklist, and SSRF policy. It is part of the example,
not generated local state.

The Claude setup also initializes
`output/claude-session-store.git`; native and LangGraph do not carry that
Claude-specific surface.

These are ordinary app-agnostic KDCube descriptors. This direct process does
not use `bundles.yaml`. `.env` only supplies the same generated Redis/Postgres
credentials to Compose; agent configuration remains descriptor-owned.

## 5. Start Redis and Postgres

```bash
docker compose --env-file .env -f compose.yaml up -d --wait
docker compose --env-file .env -f compose.yaml ps
```

Both services must report healthy. The agent creates its required tables.

## 6. Inspect what YAML controls

`config.local.yaml` controls the agent itself:

```yaml
agent:
  topic: "the current stable Python release and its release date"
  instructions: |
    You are a research agent. Use the configured tools for public-web research
    and deliverable creation. Preserve source URLs and follow enabled skills.
  tools:
    - id: demo.web_search
      enabled: true
      runtime: local
    - id: demo.create_briefing
      enabled: true
      runtime: local
  skills:
    root: ./skills
    enabled:
      - demo.research-brief

web_search:
  config: ./web-search.yaml
```

The runner turns enabled rows into an exact tool allow-list and loads the named
skill from this directory. Unknown tool or skill IDs fail during construction.

Every example uses KDCube Web Search. Native and LangGraph adapt its Python
implementation to their native tool interfaces; Claude starts the same module
as a local stdio MCP server and denies Claude's ambient `WebSearch` and
`WebFetch`. The adapter-facing IDs differ, but the search implementation and
operator policy do not:

| Agent | Enabled search operation |
| --- | --- |
| Native ReAct | `demo.web_search` |
| LangGraph | `web_search` |
| Claude Code | `mcp__kdcube_web_search__web_search` |

`web-search.yaml` is the Web Search tool's own policy:

```yaml
filter:
  allowlist:
    - python.org
  blocklist: []
  ssrf_guard: true
```

The shipped topic concerns Python, so the initial allowlist admits
`python.org` and its subdomains. Change the list when you change the topic. A
tool call can narrow this policy with `sites`; it cannot widen it. The full
standalone tool contract is in the
[Web Search MCP quick start](../../../../../mcp/web-search/README.md).

`descriptors.local/assembly.yaml` controls the host:

```yaml
infra:
  postgres:
    host: 127.0.0.1
    port: 55432
  redis:
    host: 127.0.0.1
    port: 56379

storage:
  kdcube: ../output/conversation-store

models:
  default_llm_model_id: gpt-4o-mini
```

The exact default ports differ between the three directories so their Compose
projects can run independently. Credentials live in
`descriptors.local/secrets.yaml`; accounting prices and policy live in
`descriptors.local/economics.yaml`.

## 7. Check and run

```bash
.venv/bin/python agent.py --check
.venv/bin/python agent.py --infra-check
.venv/bin/python agent.py
```

`--check` constructs the adapter, exact tools, and skills without contacting a
provider. `--infra-check` verifies Redis, storage, and Postgres tables without
model spend. The full demonstration searches the web over multiple turns and
creates a real PDF and XLSX. Its final line is:

```text
demonstration: PASS
```

Inspect the result:

```bash
find output -type f \
  \( -name 'research-brief.pdf' -o -name 'research-data.xlsx' -o -name 'communicator.jsonl' \) \
  -print
```

Postgres holds the durable conversation index and turn metadata. Configured
storage holds turn payloads. Redis holds the per-turn accounted-event mirror.

## 8. Enable isolated Python execution

The native agent can replace its trusted local deliverable tool with the SDK's
isolated Python executor. Build the image first:

```bash
cd "$KDCUBE_SOURCE"
docker build \
  -t py-code-exec:latest \
  -f app/ai-app/deployment/docker/all_in_one_kdcube/Dockerfile_Exec \
  app/ai-app
docker image inspect py-code-exec:latest >/dev/null
cd agents/native
```

Change the tool rows in `config.local.yaml`:

```yaml
  tools:
    - id: demo.web_search
      enabled: true
      runtime: local
    - id: demo.create_briefing
      enabled: false
      runtime: local
    - id: exec_tools.execute_code_python
      enabled: true
      runtime: docker
```

Executor image, timeout, network, strategy, and workspace limits remain under
`platform.services.proc.exec` in `descriptors.local/assembly.yaml`. Run the
three commands from step 7 again. A missing image fails before model spend.

## 9. Agent-specific continuity

Native adds a third turn in another conversation and requires
`react.memsearch` to recover the first conversation for the same user.

LangGraph stores graph checkpoints in Postgres in addition to the common
KDCube conversation records. `KDCubeChatModel` routes its streamed model events
through harness accounting.

Claude stores its resumable CLI transcript on a per-conversation Git branch.
Its infrastructure check proves that branch is writable. Inspect it with:

```bash
git --git-dir output/claude-session-store.git for-each-ref \
  --format='%(refname)' refs/heads/kdcube/claude/
```

To use a private Git remote, edit
`descriptors.local/assembly.yaml` at
`storage.claude_code_session.repo`. Put an HTTPS token at
`services.git.http_token` in `descriptors.local/secrets.yaml`, or use the
standard SSH fields under `services.git` in `assembly.yaml`.

## 10. Change the agent

Use `config.local.yaml` to change instructions, topic, tools, skills, and
limits. Use the standard descriptors to change model, infrastructure, storage,
economics, executor, or credentials. Use `web-search.yaml` for Web Search
egress policy. Add other local tool implementations in `tools.py`, then select
their IDs in YAML and rerun `--check`.

YAML selects installed code. Every enabled tool and skill therefore has a real
implementation that construction validates.

## 11. Understand the boundary

This direct-host recipe gives one Python process harness persistence,
accounting, communicator events, skills, and an agent adapter. The process
owner controls its tools and configuration.

For a multi-user runtime with chat UI, authentication, delegated consent,
governed tool and side-effect enforcement, isolated workspaces, rate/spend
policy, and app hosting, follow
[Quick Start: Run KDCube Locally](../../quick-start-README.md).

## 12. Stop services

From the selected agent directory:

```bash
docker compose --env-file .env -f compose.yaml down
```
