---
id: repo:kdcube-ai-app/agents/shared/README.md
title: "Shared Agent Example Setup"
summary: "Standard descriptors, support services, skills, and Python helpers shared by the three runnable agents."
tags: ["agents", "descriptors", "redis", "postgres", "skills"]
keywords: ["agent shared setup", "assembly.yaml", "secrets.yaml", "economics.yaml"]
updated_at: 2026-09-04
see_also:
  - repo:kdcube-ai-app/agents/README.md
---
# Shared Agent Example Setup

## What it is

This directory holds the setup used by `../native`, `../langgraph`, and
`../claude`: standard descriptors, Redis/Postgres Compose services, one sample
skill, and small shared Python helpers.

## Run it

```bash
python3 configure.py --provider openai
docker compose --env-file .env -f compose.yaml up -d --wait
```

## What the demo shows

The shared setup gives each agent the same model, storage, economics,
infrastructure, executor, and transcript-store contract.

## Change the demo

Change platform settings in `descriptors.local/`. Change agent behavior in the
chosen agent's `config.local.yaml`.
