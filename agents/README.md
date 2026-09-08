---
id: repo:kdcube-ai-app/agents/README.md
title: "Build and Run On-Premises Agents with the KDCube Harness"
summary: "Run Native ReAct, LangGraph, Claude Code, or your own agent on infrastructure you control with configurable models, durable conversations, tools, skills, isolated code execution, files, and usage evidence."
tags: ["agents", "harness", "native-react", "langgraph", "claude-code", "quickstart", "web-search", "web-fetch"]
keywords: ["agent examples", "DirectAgentHarness", "KDCube Web Search", "KDCube Web Fetch", "Redis", "Postgres", "PDF", "XLSX"]
updated_at: 2026-09-08
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/quick-start-README.md
  - repo:kdcube-ai-app/mcp/web-search/README.md
---
# Build and Run On-Premises Agents with the KDCube Harness

The KDCube Agent Harness assembles an agent implementation with configurable
model access, instructions, tools, skills, durable state, local filesystem or
S3 storage, and input channels. Use these examples to build and operate that
composition on infrastructure you control.

This directory gives you three complete, runnable starting points:

- **Native ReAct** is an included agent implementation you can use and change.
- **LangGraph** shows the same harness capabilities around a LangGraph agent.
- **Claude Code** shows the same harness capabilities around Claude Code.

Each example identifies the small adapter boundary where an agent loop joins
the harness. That boundary is also the integration point for an agent you
already operate or a new implementation you build. The direct on-premises
topology is an agent process, Redis, Postgres, local filesystem or S3 storage,
and either an on-host model endpoint or a selected provider API.

```text
Agent implementation
  KDCube Native ReAct | LangGraph | Claude Code | your adapter
                                |
                                v
KDCube Agent Harness, configured from YAML
  model | instructions | skills | tool sources + allowed operations
  conversation identity | local/S3 storage | terminal/Telegram input
                                |
          +---------------------+---------------------+
          |                     |                     |
          v                     v                     v
  durable conversations   isolated turn workspace   streamed evidence
  and earlier recall      and code execution         usage and cost
```

This is a constructor: each directory already runs, and its YAML lets you
select or replace one piece at a time. Begin with a working example, then
change the model, instructions, tools, skills, storage, channel, or agent
implementation independently.

## Choose your starting agent

| Start here when... | Ready implementation | Model path | Run it |
| --- | --- | --- | --- |
| You want a complete first agent, especially with a small on-host model | KDCube Native ReAct controls its own observe/reason/act loop and tool protocol | Provider API or on-host endpoint | [Native ReAct](native/README.md) |
| You want to construct the workflow as a LangGraph graph | LangGraph `create_agent` with durable checkpoints, KDCube tools, streaming, and accounting | Provider API or a compatible on-host endpoint | [LangGraph](langgraph/README.md) |
| You want a coding agent with Claude Code's own loop | `ClaudeCodeAgent` with a Git-backed transcript, workspace, KDCube tools, and harness evidence | Claude Code's Anthropic model path | [Claude Code](claude/README.md) |

## What the harness gives the agent

Choose the pieces your agent needs. A **tool** is a named capability the agent
can call during a turn; the YAML inventory determines which tools the agent
can see and use.

- **Durable conversations:** continue a conversation by stable user and
  conversation identity. The Native example also searches a different earlier
  conversation for that same user with `react.memsearch`.
- **Research tools:** use YAML-selected
  [KDCube Web Search and Web Fetch](../mcp/web-search/README.md) with explicit
  source policy.
- **Files and attachments:** receive a file, preserve it in local filesystem
  or S3 storage, and return generated files.
- **Isolated code execution:** the agent generates Python and calls the enabled
  code-execution tool. A trusted supervisor starts a separate executor in an
  isolated turn workspace and retains an archive of the exact program and
  declared outputs. The executor has no outbound network and receives neither
  model/provider credentials nor the Web Search credential; it receives only
  the scoped supervisor IPC needed for declared nested tool calls.
- **Tools from generated code:** generated Python can call an enabled catalog
  tool through `agent_io_tools.tool_call`. The trusted supervisor resolves that
  call from the same descriptor-selected tool catalog and enforces the same
  per-callable allow policy.
- **Document production:** the agent creates HTML or Markdown and calls an
  enabled rendering tool. The harness converts it into PDF, DOCX, or PPTX.
- **Instructions and skills:** select a maintained instruction profile, add
  product instructions, and enable reusable `SKILL.md` procedures from YAML.
- **Transparent execution:** stream communicator events and record model
  usage, cost, tool activity, conversation turns, files, and execution
  evidence.
- **Direct channels:** continue in a terminal or receive and answer Telegram
  messages through a local development webhook.

The agent implementation decides what to do. The harness supplies the
conversation, tool, workspace, file, streaming, and accountability contracts
around those decisions. YAML selects the capabilities available in each run.

## Use the ready composition

The included **research and report** flow exercises the pieces together. It
searches the web, inspects a source, carries findings into another turn,
authors Python, creates an XLSX workbook, and renders a PDF. Change the topic,
instructions, source policy, tools, skill, and output contract for your own
domain workflow.

The shell runner is a normal command-line job that can be invoked from a
scheduler or queue. A KDCube app carries the same composition into an
on-premises multi-user runtime with authenticated ingress, governed tool
execution, durable jobs, and administrator/user policy.

## Resource profile

The selected model and enabled tools determine the machine profile:

| Component | Resource use | Selection |
| --- | --- | --- |
| Model endpoint | VRAM or provider API capacity; weights, quantization, and context length define the local footprint | `descriptors.local/assembly.yaml` |
| Agent process | Host Python process for the selected Native, LangGraph, or Claude adapter | selected agent directory |
| Conversation and accounting services | Redis and Postgres preserve conversation and usage records; conversation files and artifacts use local filesystem or S3 storage | `compose.yaml` and `storage.kdcube` in `descriptors.local/assembly.yaml` |
| Isolated code execution | The agent generates Python and calls `exec_tools.execute_code_python`; the trusted supervisor starts the executor with the configured image, CPU, RAM, network, and isolated turn-workspace limits | select `execute_code_python` in the `exec_tools` source under `agent.tools`; configure execution under `platform.services.proc.exec` |
| Document production | The agent creates HTML or Markdown and calls `rendering_tools.write_pdf`, `write_docx`, or `write_pptx`; PDF and PPTX conversion uses Playwright/Chromium | select the required callable names in the `rendering_tools` source under `agent.tools` |

Enable the resource-bearing components required by the deployment. Isolated
code execution and rendering are selected through YAML tool sources; the model,
service, storage, and executor limits are selected through platform
descriptors. Native ReAct carries its action protocol in instructions and
parsing, which lets a capable text-generation model use configured tools
through an on-host endpoint.

## How do I try it?

Start with the complete Native ReAct agent and validate its local composition:

```bash
cd agents/native
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python setup_local.py --provider none
cp config.template.yaml config.local.yaml
.venv/bin/python agent.py --check
```

This prepares ignored local descriptors and validates the composition before a
model call. Next, select and start the on-host model endpoint, start Redis and
Postgres, and run the agent. The exact copyable sequence is in the
[Native README](native/README.md), including the provider-API alternative, and
in the complete
[Run the Agent Harness from Python recipe](../app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md).

For any of the three examples, the operating sequence is:

1. Create that directory's `.venv` and install its `requirements.txt`.
2. Run `setup_local.py` and copy `config.template.yaml` to the ignored
   `config.local.yaml`.
3. Select an on-host model or provider model in standard platform descriptors.
4. Select instructions, tools, tool settings, and skills in
   `config.local.yaml`.
5. Start Redis and Postgres. Build the isolated code executor and install
   Chromium when their tool sources are enabled.
6. Run `agent.py`, then inspect conversation, file, execution, event, usage,
   and generated-code evidence in the configured storage backend. The
   templates use `output/` on the local filesystem.

Choose the model through the standard platform descriptor:

| Agent | Model path | Start here |
| --- | --- | --- |
| Native ReAct | Provider API or on-host model through the KDCube model gateway | `descriptors.local/assembly.yaml` |
| LangGraph | Provider API or on-host model through the KDCube model gateway | `descriptors.local/assembly.yaml` |
| Claude Code | Claude Code's Anthropic model path | `descriptors.local/assembly.yaml` |

Each agent directory contains its runner, requirements, agent YAML, standard
platform descriptors, example skill, Redis/Postgres Compose file, and exact
commands. The command in each README executes that directory's visible
`agent.py` directly.

Create a `.venv` in the selected agent directory and install its
`requirements.txt`. The default research-and-report demonstration also uses
two explicit preparations:

- install Playwright Chromium with `.venv/bin/python -m playwright install chromium`
  because the PDF/PPTX renderers use it; and
- build `py-code-exec:latest` with the documented `docker build` command because
  model-authored Python runs in the isolated executor image.

Both commands are included in every runner's copyable setup block. A smaller
search-and-summary configuration can select only the Web tool source.

Every runner receives its caller and conversation explicitly:

```yaml
agent:
  input:
    user_id: demo-user
    user_type: regular
    session_id: local-session
    conversation_id: native-demo
```

Run the same example again with the same `user_id` and `conversation_id` to
continue that durable conversation. Use another `conversation_id` to start a
separate conversation. The shared conversation key is tenant, project, user,
and conversation; each adapter adds its stable `agent_id` to its private
checkpoint or transcript key. `session_id` identifies the calling session and
accounting lineage while the durable conversation key remains stable. The
values can also be overridden with `--user-id`, `--conversation-id`, and
`--session-id`.

From the selected agent directory, run:

```bash
.venv/bin/python agent.py \
  --user-id alice \
  --conversation-id release-research \
  --session-id terminal-1
```

This command replaces the three values under `agent.input` for that process.
Running it again continues `release-research` for `alice`; changing the user or
conversation selects another durable history. Changing only `session-id`
records a different calling/accounting session while keeping the same durable
conversation.

The built-in two-turn demonstration is illustrated by the **research and
report** flow below:

```text
research request
      |
      v
KDCube Web Search -> KDCube Web Fetch -> inspected source evidence
                                           |
                                           v
                              retained conversation context
                                           |
                                           v
agent authors Python -> isolated code execution -> XLSX + HTML
                              |
                              +-> generated code calls an enabled Web tool
                                  through the trusted supervisor
                              |
                              v
                    rendering_tools.write_pdf -> polished PDF
```

The YAML-selected renderer family also exposes HTML-to-PPTX and
Markdown-to-DOCX. Each run records communicator events, accounted model calls,
attachments, output files, conversation turns, and the execution ZIP that
contains the model-authored `pkg/user_code.py`.

The agent authors research, code, data, HTML, and Markdown. The isolated
executor runs its program, and KDCube's document tools own repeatable PDF,
DOCX, and PPTX conversion. This produces a concrete research-and-file agent
while keeping the selected agent loop replaceable.

Each YAML also selects an SDK-owned instruction profile. Native uses the
standard ReAct `lite:core` body plus blocks for its enabled tools. LangGraph and
Claude use the framework-neutral `workspace-files` body. Product behavior goes
in `additional_instructions`, after the workspace, capability, and skill
teaching. See
[Direct Agent Instruction Profiles](../app/ai-app/docs/runtime/harness/direct-agent-instruction-profiles-README.md).

After the first run, change these constructor inputs:

| Change | File |
| --- | --- |
| Research subject or workflow | `config.local.yaml#agent.topic` and `agent.additional_instructions` |
| Enabled capability and its policy | The exact `config.local.yaml#agent.tools[id=...]` row |
| Reusable procedure | `skills/<skill-id>/SKILL.md` and `agent.skills.enabled` |
| On-host or provider model | `descriptors.local/assembly.yaml` |
| Local filesystem or S3 storage | `storage.kdcube` in `descriptors.local/assembly.yaml` |
| Conversation identity | `agent.input` or the corresponding CLI flags |
| Terminal or local Telegram input | CLI mode and `agent.ingress.telegram` |

`agent.tools` declares tool sources, rather than one row for every callable.
For a Python source, `module` or `ref` identifies the implementation, `alias`
defines its tool-ID prefix, `allowed` selects exact callable names, and
`runtime` selects where each callable executes. `ToolSubsystem` introspects the
declared source and builds the catalog dynamically. Removing a callable from
`allowed` removes it from the model catalog and from generated-code supervisor
admission.

To add your own Python tool, put its module or bundle-relative file in one
source row and select the callable names the agent may use:

```yaml
agent:
  tools:
    - id: market-data
      kind: python
      ref: ./market_tools.py
      alias: market_tools
      discovery: semantic_kernel
      allowed: [latest_prices, supplier_snapshot]
      runtime:
        latest_prices: local
        supplier_snapshot: local
```

That row is the canonical discovery and execution policy. Native ReAct reads
the resulting catalog directly. LangGraph also needs a small `BaseTool` schema
adapter in `langgraph/tools.py`, and Claude Code needs an MCP adapter that
presents the callable to Claude. Those adapters translate model-facing names
and argument schemas; they do not create a second allowlist or bypass
`ToolSubsystem` enforcement.

The canonical discovery, naming, binding, and isolated-supervisor contract is
documented in [Tool Subsystem](../app/ai-app/docs/sdk/tools/tool-subsystem-README.md).
The complete shared command sequence is in
[Run the Agent Harness from Python](../app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md).
All three runners use the implementation surfaced by the
[Web Search and Web Fetch MCP package](../mcp/web-search/README.md). Native and
LangGraph bind its SDK functions in-process; Claude starts that package's
public launcher as a local stdio MCP server.

## Talk to the agent

Keep one durable conversation open in the terminal:

```bash
.venv/bin/python agent.py --interactive \
  --user-id alice \
  --conversation-id terminal-chat \
  --session-id terminal-1
```

Or point one Telegram bot at the local development hook:

```text
Telegram webhook + verified secret
              |
              v
      direct inline callback
              |
              v
    selected agent + DirectAgentHarness
              |
              +--> Postgres conversation
              +--> configured storage and files
              +--> Telegram text/file response
```

The Telegram hook uses `agent.ingress.telegram` and the ignored local secrets
descriptor. It maps the Telegram sender to `user_id=telegram_<sender-id>` and
the chat to `conversation_id=telegram_chat_<chat-id>`. It reuses KDCube's
Telegram update, attachment, and delivery SDK inside the standalone agent
process. Follow the exact setup in
[Run the Agent Harness from Python](../app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md#10-connect-a-local-telegram-bot).

This local experiment performs inline processing in one process. The process
prevents turns from overlapping; concurrent webhook arrival order remains
unspecified, update claims have process lifetime, and the HTTP request remains
open while the agent runs. KDCube's hosted chat ingress, or a durable queue
built around the callback, supplies ordering, retry recovery, asynchronous
execution, live controls, and multiple workers. Connection Hub links identity
and governs delegated tools in the hosted product; the app and chat runtime own
transport ingress.

## Serve the configured agent to users

Place the tested composition in a KDCube app and declare a chat, API, job, or
messaging surface. The hosted runtime supplies authenticated user and
conversation IDs, tool-execution enforcement, consent, rate/spend policy,
durable jobs, and multi-user ingress. Follow
[Settle Your Solution in KDCube](../app/ai-app/docs/recipes/apps/settle-your-solution-in-kdcube-README.md)
and the [KDCube Quick Start](../app/ai-app/docs/quick-start-README.md).
