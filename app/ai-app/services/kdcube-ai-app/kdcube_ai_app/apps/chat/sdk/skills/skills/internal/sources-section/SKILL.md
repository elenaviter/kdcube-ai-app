---
name: sources-section
description: |
  Appends a Sources/References section using only web sources from the sources pool.
version: 1.0.0
category: reporting
tags:
  - sources
  - citations
  - references
  - web
namespace: internal
when_to_use:
  - The user asks for sources, references, or citations
  - You are producing a report that must include a sources section
  - You are generating HTML/Markdown that should list web evidence
author: kdcube
created: 2026-01-16
---

# Sources Section (Web Sources Only)

## Overview
When sources are requested, append a Sources or References section at the end of the document.
Only include web sources that exist in the sources pool and were actually used.
Do NOT include attachments or local files in the sources list.

## Rules
1. Web sources only
   - Include only sources with web URLs from sources pool.
   - Exclude attachments and files, even if they were used.

2. Use real URLs and titles
   - Use the exact URL and title as shown in sources pool entries.
   - Never invent or guess a URL.

3. Placement
   - Always place the Sources/References section at the end of the document.

## Markdown Template
```
## Sources
1. [Title from source](https://example.com) — S:12
2. [Another source](https://example.org) — S:18
```

## HTML Template
```
<section class="sources">
  <h2>Sources</h2>
  <ol>
    <li><a href="https://example.com">Title from source</a> <span class="sid">S:12</span></li>
    <li><a href="https://example.org">Another source</a> <span class="sid">S:18</span></li>
  </ol>
</section>
```

## Notes
- SIDs should correspond to sources pool entries actually used in the content.
- If no web sources were used, do not fabricate a sources section.
