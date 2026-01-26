---
name: url-gen
description: |
  Generate clean, human-facing URLs for fetch tools when external evidence is needed.
version: 1.0.0
category: research
tags:
  - urls
  - fetch
  - discovery
when_to_use:
  - You need external evidence and must provide URLs for fetch tools
  - You are forming a list of sources to retrieve with fetch_url_contents
author: kdcube
created: 2026-01-16
namespace: public
---

# URL Generation

## Overview
Generate URLs that are relevant and likely accessible for fetch tools. Prefer human-facing pages and avoid deep, speculative paths.

## When to Use This Skill
- You need external evidence and must provide URLs for fetch tools.
- You are forming a list of sources to retrieve with `fetch_url_contents`.

## Rules
1. Relevance
   - Only suggest URLs clearly relevant to the task.
   - Do not invent deep paths if you are unsure they exist.

2. Prefer human-facing pages
   - Prefer normal, human-facing pages over programmatic endpoints.
   - If multiple paths lead to the same info, prefer the one without segments like api, v1, v2, json, rest, graphql.
   - Example: prefer https://openai.com/pricing over https://openai.com/api/pricing.

3. Avoid machine-only endpoints unless requested
   - Do not suggest /api/, .json, .xml, or /graphql unless the user explicitly asks for APIs or raw data.

## Hard Rule for fetch_context
- Generated URLs do NOT exist in context, so you MUST NOT put them in fetch_context.path.
- Put generated URLs directly into tool_call.params (e.g., {"urls": ["https://..."]}).
- fetch_context is only for reusing existing strings from context.
