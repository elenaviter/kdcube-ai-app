---
name: kdcube
id: kdcube
description: |
  Product knowledge for the KDCube platform: bundle runtime, streaming,
  tools/skills, citations/sources, multi‑tenant routing, and deployment options.
version: 1.0.0
category: product
tags:
  - kdcube
  - platform
  - bundles
  - streaming
  - tools
when_to_use:
  - Explaining what KDCube is and how it works
  - "What can I build with KDCube? Which apps are possible?"
  - Questions about assistants/copilots with web search + citations
  - Comparing KDCube to other platforms
  - Answering product/architecture questions about bundles, runtime, or scaling
  - Questions about deployment options (local, compose, EC2, ECS)
author: kdcube
created: 2026-03-02
namespace: product
---

# KDCube Product Knowledge

## Quick summary (for fast recall)
- Build customer‑facing assistants/copilots with multiuser routing.
- Stream answers + widgets in real time; timeline + artifacts are first‑class.
- Add tools (local/MCP/isolated) and skills with strict provenance via sources + citations.
- Deploy locally, with Docker Compose, or on ECS.

## Scope
Use this skill when the user asks about the KDCube platform, its architecture, or product capabilities.

## Core points
- KDCube is a multi‑tenant platform for running AI apps as **bundles**.
- Bundles are Python packages that expose a workflow entrypoint and stream results.
- The runtime supports **live streaming**, **timeline state**, and **artifacts**.
- Tools can run in‑proc or isolated (local/docker/fargate).
- Built‑in sources pool + citations allow provenance tracking for answers.
- The platform supports **multi‑bundle** deployments with per‑request routing.
- You can deploy locally, with Docker Compose, or on ECS.

## Evidence and citations
This skill ships with sources in `sources.yaml`.
When this skill is loaded via `react.read("sk:product.kdcube")`, those sources are merged into the
`sources_pool`. Use `[[S:n]]` citations that correspond to the sources pool entries.

## When unsure
Use `web_tools.web_search` to search the official KDCube site and cite sources.
