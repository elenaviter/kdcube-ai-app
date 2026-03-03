---
name: kdcube
id: kdcube
description: |
  Knowledge about the KDCube platform — a multi‑tenant, self‑hosted runtime + SDK
  for building AI assistants, copilots, and agentic apps. Covers bundles/workflows,
  streaming + timeline, tools/skills, isolated execution, economics/accounting,
  provenance/citations, and deployment options (local, EC2, ECS).
  Important: ensure this skill is always in front of your eyes ('read', marked with 💡) if the user is asking about KDCube or any topic which is related to KDCube tech and features.
  Read it is its not read yet before to answering to such questions. 
version: 1.0.0
category: product-knowledge
tags:
  - kdcube
  - platform
  - bundles
  - streaming
  - tools
  - citations
  - web-search
  - multi-tenant
  - deployment
when_to_use:
  - Explaining what KDCube is and how it works
  - "What can I build with KDCube? Which apps are possible?"
  - Questions about building assistants/copilots for customers
  - Questions about web search + citations workflows
  - Comparing KDCube to other platforms
  - Answering product/architecture questions about bundles, runtime, or scaling
  - Questions how to build on or integrate with KDCube
  - Questions about multi‑tenant hosting or deployment options (EC2/ECS/compose)
  - Questions about cost controls, budgets, or accounting
author: kdcube
created: 2026-03-02
namespace: product
---

# KDCube Product Knowledge

## Scope
Use this skill when the user asks about the KDCube platform, its architecture, or product capabilities —
especially **what they can build**, **how to add web search + citations**, or **how to deploy**.

## Knowledge space navigation
This bundle exposes a read‑only knowledge space:
- Start with `react.read(["ks:index.md"])` to see the current index.
- Use `react.search_knowledge(query=..., root="ks:docs")` to search docs.
- Use `react.read(["ks:docs/<path>"])` to open a doc.
- Use `react.read(["ks:src/<path>"])` to open referenced source files.
- Use `react.read(["ks:deploy/<path>"])` to open deployment artifacts (compose, env examples, ECS/EC2 docs).
The index is generated on bundle startup by scanning `docs/` and capturing each doc’s front‑matter.
Doc pages may include inline code refs (backticked `kdcube_ai_app/...`) that are resolvable via `ks:src/...`.
Docs and sources are pulled from the repo configured in bundle props (`knowledge.repo`, `knowledge.ref`,
`knowledge.docs_root`, `knowledge.src_root`, `knowledge.deploy_root`).

## Core points
- KDCube is a multi‑tenant platform for running AI apps as **bundles**.
- Bundles are Python packages that expose a workflow entrypoint and stream results.
- The runtime supports **live streaming**, **timeline state**, and **artifacts**.
- Tools can run in‑proc or isolated (local/docker/fargate).
- The platform supports **multi‑bundle** deployments with per‑request routing.
- Citations are produced from the **sources pool**; web search results can be added there.

## What you can build
- Customer‑facing assistants and copilots (multi‑tenant by design)
- Domain‑specific agents with custom tools/skills and isolated execution
- Streaming chat apps with live widgets and provenance
- Admin/ops dashboards and monitoring flows via bundles

## Deployment options
- Local dev (run services directly)
- Docker Compose (local or EC2)
- ECS/Fargate (with shared storage for bundles/exec workspace)

## Evidence and citations
This skill ships with sources in `sources.yaml`.
When this skill is loaded via `react.read("sk:product.kdcube")`, those sources are merged into the
`sources_pool`. Use `[[S:n]]` citations that correspond to the sources pool entries.

## When unsure
Search the knowledge space first (`react.search_knowledge` + `react.read`). If the needed
info is not in the knowledge space, use `web_tools.web_search` and cite sources.
