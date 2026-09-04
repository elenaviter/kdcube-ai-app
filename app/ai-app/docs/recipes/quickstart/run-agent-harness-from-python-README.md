---
id: repo:kdcube-ai-app/app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md
title: "Recipe: Run the Agent Harness from Python"
summary: "Executable steps for running a YAML-configured native ReAct, LangGraph, or Claude Code agent from KDCube SDK source with standard platform descriptors."
status: current
tags: ["recipe", "quickstart", "agent-harness", "python", "native-react", "langgraph", "claude-code", "self-hosted"]
keywords: ["run agent harness", "direct SDK agent", "standard descriptors", "Redis", "Postgres", "Git transcript", "isolated code execution"]
updated_at: 2026-09-04
see_also:
  - repo:kdcube-ai-app/agents/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/arch/security-and-trust-model-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/configuration/assembly-descriptor-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/configuration/secrets-descriptor-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/exec/README-iso-runtime.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/components/chat-with-react-agent-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/components/chat-with-langgraph-agent-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/components/chat-with-claude-code-agent-README.md
---
# Recipe: Run the Agent Harness from Python

This recipe runs a real agent from a shell or IDE as a self-hosted direct SDK
process. It imports the KDCube SDK from this checkout. Redis and Postgres are
its independent support services.

The three runnable adapters share one ordinary platform descriptor set:

```text
agents/shared/descriptors.local/
  assembly.yaml       infrastructure, storage, default model, Git, executor
  secrets.yaml        provider, Redis, Postgres, and optional Git credentials
  economics.yaml      model prices and economics policy
  gateway.yaml        empty in this direct process
```

`bundles.yaml` belongs to the deployed-app path. These direct SDK hosts use the
app-agnostic descriptor set above. Each adapter has one small
`config.local.yaml` for its own instructions, tools, skills, limits, topic, and
output path.

## 0. Prerequisites

Install:

- Git;
- Python 3.11;
- Docker Engine or Docker Desktop with Compose;
- a provider API key for native ReAct or LangGraph, or an authenticated Claude
  Code CLI for the Claude example.

The direct path runs with the source checkout, Python, Git, Redis, Postgres, and
the selected model interface. The managed-runtime path adds the `kdcube` CLI,
gateway, processor, browser UI, and staged runtime.

## 1. Get the source

```bash
git clone https://github.com/kdcube/kdcube.git
cd kdcube
export KDCUBE_SOURCE="$PWD"
```

Each example's `requirements.txt` installs
`app/ai-app/src/kdcube-ai-app` from this checkout in editable mode.

## 2. Create local standard descriptors

For native ReAct or LangGraph, run:

```bash
cd "$KDCUBE_SOURCE/agents/shared"
python3 configure.py --provider openai
```

Enter the OpenAI key at the hidden prompt. For a Claude-only run authenticated
through an existing Claude CLI login, use:

```bash
python3 configure.py --provider none
```

The initializer creates ignored `descriptors.local/` and `.env`, generates the
Redis/Postgres passwords, and initializes
`output/claude-session-store.git`. Secret values stay in owner-only local
files.

The agent processes read configuration only from standard descriptors. The
`.env` file is input to Docker Compose so its Redis/Postgres containers receive
the same generated passwords.

## 3. Start Redis and Postgres

```bash
cd "$KDCUBE_SOURCE/agents/shared"
docker compose --env-file .env -f compose.yaml up -d --wait
docker compose --env-file .env -f compose.yaml ps
```

Both services must report healthy. The runners create their required tables
automatically.

## 4. Install the native example

```bash
cd "$KDCUBE_SOURCE/agents/native"
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
cp config.template.yaml config.local.yaml
```

## 5. Inspect the two configuration layers

The agent declaration in `config.local.yaml` is executable input:

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
    - id: exec_tools.execute_code_python
      enabled: false
      runtime: docker
  skills:
    root: ../shared/skills
    enabled:
      - demo.research-brief
```

The runner turns enabled rows into an exact tool allow-list and loads the named
skill from `agents/shared/skills/`. Every selected tool or skill ID must resolve
during construction.

Platform concerns stay in `../shared/descriptors.local/assembly.yaml`:

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
  claude_code_session:
    type: git
    repo: ../output/claude-session-store.git

models:
  default_llm_model_id: gpt-4o-mini
```

Canonical credentials are in the ignored `secrets.yaml`, for example
`services.openai.api_key`, `infra.postgres.password`, and
`infra.redis.password`. Accounting resolves prices from `economics.yaml`.

## 6. Prove construction and infrastructure

Run both checks before spending provider tokens:

```bash
.venv/bin/python agent.py \
  --config config.local.yaml \
  --descriptors ../shared/descriptors.local \
  --check

.venv/bin/python agent.py \
  --config config.local.yaml \
  --descriptors ../shared/descriptors.local \
  --infra-check
```

The first command constructs the adapter, tool inventory, and skills before
contacting a model provider. It ends with:

```text
mode: standalone SDK process
tools: demo.web_search, demo.create_briefing
skills: demo.research-brief
check: PASS
```

The second command connects to Redis and Postgres, creates the common
conversation tables, and opens configured storage. It ends with:

```text
infrastructure: Redis, Postgres conversation tables, and storage ready
infrastructure check: PASS
```

## 7. Run the demonstration

```bash
.venv/bin/python agent.py \
  --config config.local.yaml \
  --descriptors ../shared/descriptors.local
```

The native demonstration performs three turns:

1. `demo.web_search` gathers five findings and source URLs.
2. `demo.create_briefing` creates a PDF and XLSX from the retained findings.
3. A new conversation calls `react.memsearch` and recovers a source from the
   first conversation for the same user.

The command verifies the files, durable conversation records, storage payloads,
and recall tool event. Its last line is:

```text
demonstration: PASS
```

## 8. Inspect the evidence

```bash
find output -type f \
  \( -name 'research-brief.pdf' -o -name 'research-data.xlsx' -o -name 'communicator.jsonl' \) \
  -print
tail -n 20 output/communicator.jsonl
find ../shared/output/conversation-store -type f | head
```

Postgres stores the durable conversation index and turn metadata. Configured
storage contains the materialized turn payloads. Redis contains the per-turn
accounting/event mirror. The accounting events use the prices in the active
`economics.yaml`; its template keeps enforcement disabled for this local demo.

## 9. Enable isolated Python execution

Build the executor image once:

```bash
cd "$KDCUBE_SOURCE"
docker build \
  -t py-code-exec:latest \
  -f app/ai-app/deployment/docker/all_in_one_kdcube/Dockerfile_Exec \
  app/ai-app
docker image inspect py-code-exec:latest >/dev/null
```

In `agents/native/config.local.yaml`, disable the trusted local file builder and
enable the SDK executor:

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

Executor image, timeout, network, strategy, and workspace limits remain in the
normal platform location:

```yaml
platform:
  services:
    proc:
      exec:
        py_code_exec_image: py-code-exec:latest
        py_code_exec_network_mode: none
        py_code_exec_container_strategy: split
```

Rerun `--check`, `--infra-check`, and the full command from step 6. A missing
image stops construction before a model call. Generated Python runs in the
configured isolated executor and returns files through its explicit artifact
contract.

## 10. Run LangGraph

```bash
cd "$KDCUBE_SOURCE/agents/langgraph"
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp config.template.yaml config.local.yaml
.venv/bin/python agent.py --config config.local.yaml --descriptors ../shared/descriptors.local --check
.venv/bin/python agent.py --config config.local.yaml --descriptors ../shared/descriptors.local --infra-check
.venv/bin/python agent.py --config config.local.yaml --descriptors ../shared/descriptors.local
```

`KDCubeChatModel` sends LangGraph's streamed model calls through the accounted
harness bridge. The common KDCube conversation tables persist the two harness
turns; LangGraph creates additional Postgres checkpoint tables for its graph
thread.

## 11. Run Claude Code

Install and authenticate Claude Code, then run:

```bash
cd "$KDCUBE_SOURCE/agents/claude"
claude --version
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp config.template.yaml config.local.yaml
.venv/bin/python agent.py --config config.local.yaml --descriptors ../shared/descriptors.local --check
.venv/bin/python agent.py --config config.local.yaml --descriptors ../shared/descriptors.local --infra-check
.venv/bin/python agent.py --config config.local.yaml --descriptors ../shared/descriptors.local
```

The infrastructure check pushes a bootstrap commit to the local Git remote.
Inspect its per-conversation branch:

```bash
git --git-dir ../shared/output/claude-session-store.git for-each-ref \
  --format='%(refname)' refs/heads/kdcube/claude/
```

Claude's `research.json` and PDF/XLSX remain in its workspace. Claude's
resumable CLI JSONL transcript is bootstrapped from and published to the Git
branch by `run_claude_code_turn()`. The harness conversation remains in the
same Postgres/storage contract used by the other adapters. Run the full command
again to verify restoration in a new Python process.

To use a private Git remote, change only standard descriptor fields:

```yaml
# assembly.yaml
storage:
  claude_code_session:
    type: git
    repo: https://github.com/your-org/agent-transcripts.git
```

Put an HTTPS token at `services.git.http_token` in `secrets.yaml`, or configure
`services.git.git_ssh_key_path` and related SSH fields in `assembly.yaml`.
For key-based Claude execution, put the key at
`services.anthropic.claude_code_key` in `secrets.yaml`.

## 12. Change a model, tool, or skill

To select another registered model:

1. Change `models.default_llm_model_id` in `assembly.yaml`.
2. Put that provider's canonical key in `secrets.yaml`.
3. Add its price row to `economics.yaml`.

To add a native or LangGraph tool, implement it in the adapter's `tools.py`,
register its supported ID, select it in `config.local.yaml`, and run `--check`.
To add a skill, create its `SKILL.md` under the configured skills root, select
its qualified ID, and run `--check`.

YAML selects code that the process owner installed. Every selected ID therefore
has a corresponding tool or skill implementation.

## 13. Understand the direct-host boundary

This recipe gives one self-hosted Python process the Agent Harness persistence,
accounting, communicator, skills, and adapter contracts. The process owner is
the authority for its tools and configuration.

For a multi-user runtime with chat UI, authentication, delegated consent,
governed tool and side-effect enforcement, isolated workspaces, rate/spend
policy, and app hosting, follow
[Quick Start: Run KDCube Locally](../../quick-start-README.md).

## 14. Stop the support services

```bash
cd "$KDCUBE_SOURCE/agents/shared"
docker compose --env-file .env -f compose.yaml down
```
