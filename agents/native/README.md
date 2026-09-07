---
id: repo:kdcube-ai-app/agents/native/README.md
title: "Run the Native ReAct Agent"
summary: "Run KDCube native ReAct directly from Python with YAML-selected tools and skills."
tags: ["agents", "native-react", "harness", "standalone", "demonstration", "web-search"]
keywords: ["ReactSolverV2", "DirectAgentHarness", "KDCube Web Search", "Postgres conversation", "ChatCommunicator", "accounting"]
updated_at: 2026-09-07
see_also:
  - repo:kdcube-ai-app/agents/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md
---
# Run the Native ReAct Agent

## What it is

This directory runs KDCube's `ReactSolverV2` in your Python process.
`config.local.yaml` selects agent behavior and tool settings;
`descriptors.local/` selects the model, storage, and support services. No
KDCube server is required.

## Run it

```bash
cd agents/native
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m playwright install chromium
.venv/bin/python setup_local.py --provider anthropic
cp config.template.yaml config.local.yaml
docker compose --env-file .env -f compose.yaml up -d --wait
cd ../..
docker build -t py-code-exec:latest -f app/ai-app/deployment/docker/all_in_one_kdcube/Dockerfile_Exec app/ai-app
cd agents/native
.venv/bin/python agent.py --check
.venv/bin/python agent.py --infra-check
.venv/bin/python agent.py
```

The default run uses `agent.input.user_id: demo-user` and
`agent.input.conversation_id: native-demo`. Run it again with those values to
continue that conversation, or choose them explicitly:

```bash
.venv/bin/python agent.py \
  --user-id alice \
  --conversation-id release-research \
  --session-id terminal-1
```

The provider key prompt is hidden. Local secrets and generated descriptors are
ignored by Git. `setup_local.py` is this one-time preparation command; it does
not participate in a turn or own agent configuration.

To use an on-host model, prepare the same directory with no provider secret:

```bash
.venv/bin/python setup_local.py --provider none
cp config.template.yaml config.local.yaml
```

Then select the exact model tag in `descriptors.local/assembly.yaml`:

```yaml
models:
  default_llm_provider: custom
  default_llm_model_id: <model-tag-loaded-by-your-local-runtime>

services:
  llm:
    custom:
      endpoint: http://127.0.0.1:11500/generate
      num_ctx: 32768
```

Start Ollama in one terminal:

```bash
ollama serve
```

Load the selected model and start the KDCube protocol gateway in a second
terminal:

```bash
cd agents/native
ollama pull <model-tag-loaded-by-your-local-runtime>
.venv/bin/python -m uvicorn \
  kdcube_ai_app.apps.models_gateway.app:app \
  --host 127.0.0.1 --port 11500
```

`agent.py --check` must print the exact `custom/<model-tag>`, endpoint, and
context budget before a model call is made. Native ReAct supplies its action
protocol through instructions and parsing; the model provider does not need a
provider-native tool-calling API. The selected model must still follow that
protocol reliably within the configured context window.

## What the demo shows

Turn one hosts a research-request attachment and searches through KDCube Web
Search. Turn two makes the model author Python that uses `openpyxl`; KDCube
executes that code in the configured Docker image to create an XLSX and HTML.
The agent then calls `rendering_tools.write_pdf` to turn the HTML into a
polished PDF. It does not generate PDF bytes in Python.

The YAML also enables `write_docx` for Markdown and `write_pptx` for
section-based HTML. The default demonstration calls only `write_pdf`; alter the
second prompt to exercise either sibling operation.

Native adds a third turn in another conversation and uses `react.memsearch` to
recover its earlier research. A successful run ends with `demonstration: PASS`.
The second conversation is the explicit
`agent.input.recall_conversation_id`. Inspect
`output/runs/<user>/<conversation>/<run>/evidence.json`; it points to durable
records in `output/kdcube-storage`, including the execution ZIP containing
`pkg/user_code.py`.

## Change the demo

Edit `config.local.yaml` to change the `lite:core` instruction profile,
`additional_instructions`, run directory, tools, skills, topic, or limits.
Its `agent.input` section selects the local caller session, durable
conversation, and recall conversation. Tenant and project come from
`descriptors.local/assembly.yaml`; this runner's stable agent ID is `native`.
The SDK preserves the ReAct protocol and automatically adds standard blocks for
the enabled exec, rendering, and web tool families. Web Search's allowlist,
blocklist, and SSRF policy are under the `web_tools.web_search` tool row's
`settings`. Edit
`descriptors.local/assembly.yaml` for the model provider and ID, storage, and isolated executor;
edit `descriptors.local/secrets.yaml` for credentials. The shipped model is
`claude-haiku-4-5-20251001`.

For an on-host run, `services.llm.custom.endpoint` is the shared model gateway
and `num_ctx` is the context budget sent on every request. Model-specific
override lists are not part of this example; change the single selected model
and shared context budget directly.

To add a local Python tool, implement it in `tools.py`, register its trusted
source in `agent.py`'s `TOOL_SOURCES`, and select its ID in
`config.local.yaml`. The SDK resolves that registry into Native tool bindings;
this example does not carry a separate planner module.

Executor image, timeout, network, and workspace limits live under
`platform.services.proc.exec` in `descriptors.local/assembly.yaml`. The full
storage and inspection procedure is in
[the executable recipe](../../app/ai-app/docs/recipes/quickstart/run-agent-harness-from-python-README.md).
Profile choices and composition order are in
[Direct Agent Instruction Profiles](../../app/ai-app/docs/runtime/harness/direct-agent-instruction-profiles-README.md).
