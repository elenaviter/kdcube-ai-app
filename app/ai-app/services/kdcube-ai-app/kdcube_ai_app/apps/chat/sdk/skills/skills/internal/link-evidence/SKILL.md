---
name: link-evidence
description: |
  Prevents fabricated URLs. Only use links that appear in the sources pool or are provided by the user.
version: 1.0.0
category: research-integrity
tags:
  - links
  - citations
  - evidence
  - sourcing
when_to_use:
  - The task requires links or citations
  - You are generating reports that mention companies or technologies
  - You are summarizing web research and need to avoid fabricated URLs
author: kdcube
created: 2026-01-16
namespace: internal
---

# Link Evidence Policy

## Overview
This skill prevents fabricated URLs and enforces evidence-only linking.

## Core Rules
1. Evidence links only
   - A URL may appear only if it is present in the sources pool or explicitly provided by the user.
   - If a company/technology is mentioned but no URL appears in sources, do NOT invent one.

2. Two link types
   - Evidence links: URLs from sources pool (SIDs). These are allowed for citations and sources sections.
   - Explicit links: URLs provided by the user in the prompt or prior artifacts. These are allowed as-is.

3. No inferred domains
   - Do not invent company-name.com or similar guesses.

## What to do when a link is missing
- Keep the company name as plain text (no hyperlink).
- If links are required, rely on sources pool URLs only.

## Citation discipline
- When citations are required, use only SIDs from sources pool.
- Never cite an entity without an actual source URL present in the pool.
