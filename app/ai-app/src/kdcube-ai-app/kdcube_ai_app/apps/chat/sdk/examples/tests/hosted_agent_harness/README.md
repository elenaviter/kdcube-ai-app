---
id: repo:kdcube-ai-app/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/tests/hosted_agent_harness/README.md
title: "Run Optional Hosted Acceptance"
summary: "Verify a prepared KDCube deployment over its real chat and SSE boundary after the direct agent example works."
tags: ["agents", "integration-testing", "sse", "hosted-runtime"]
keywords: ["hosted acceptance", "runtime workdir", "browser bearer", "communicator evidence"]
updated_at: 2026-09-04
see_also:
  - repo:kdcube-ai-app/agents/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/procedures/platform-source-testing-README.md
---
# Run Optional Hosted Acceptance

## What it is

These scripts test an already prepared KDCube deployment through its public
chat and SSE transport. They are an SDK acceptance fixture, separate from the
three direct runnable agents under the repository's `agents/` directory.

## Run it

Prepare the runtime, stage the matching descriptor patch under
`config/<adapter>/`, provide a browser bearer, then run from the repository
root:

```bash
python -m venv .venv
.venv/bin/pip install -e app/ai-app/src/kdcube-ai-app
.venv/bin/pip install -r app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/examples/tests/hosted_agent_harness/requirements.txt
.venv/bin/python -m kdcube_ai_app.apps.chat.sdk.examples.tests.hosted_agent_harness.native \
  --workdir ~/.kdcube/kdcube-runtime/<tenant>__<project>
```

Substitute `langgraph` or `claude` for `native`. Use
`--preflight-only` to check the runtime, descriptor, tools, and execution image
before submitting a turn.

## What the demo shows

A full run proves that the deployed agent can:

- accept two turns through HTTP/SSE;
- stream communicator events;
- retain the first turn for the second;
- account for model use; and
- publish the required PDF and XLSX conversation files.

The runner writes its raw evidence to `events.jsonl` and exits with an error
when a required event or file is missing.

## Change the demo

- Pass `--research-prompt` and `--artifact-prompt` to change the two turns.
- Pass `--conversation-id` to continue an existing conversation.
- Pass `--raw-events` to print complete SSE envelopes.
- Edit `config/<adapter>/` to change the staged app or agent capabilities.
- Edit `demo.py` to change shared assertions for all three adapters.

Native and LangGraph use the configured isolated Python executor for file
creation. Build its image when preflight reports it missing:

```bash
docker build -t py-code-exec:latest \
  -f app/ai-app/deployment/docker/all_in_one_kdcube/Dockerfile_Exec app/ai-app
```
