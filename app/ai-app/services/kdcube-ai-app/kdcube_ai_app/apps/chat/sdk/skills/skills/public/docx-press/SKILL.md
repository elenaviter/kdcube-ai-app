---
name: docx-press
description: |
  Teaches agents how to author Markdown that renders cleanly into DOCX, with
  heading structure, tables, and citation handling.
version: 1.0.0
category: document-creation
tags:
  - docx
  - markdown
  - tables
  - citations
when_to_use:
  - Generating Markdown for write_docx
  - Building structured DOCX reports with headings and tables
  - Including citations and a references section
author: kdcube
created: 2026-01-16
namespace: public
import:
  - internal.link-evidence
  - internal.sources-section
---

# DOCX Authoring for Reports

## Overview
This skill teaches how to produce Markdown that renders cleanly into DOCX.
Use consistent heading levels, compact spacing, and valid tables.

## Core Rules
- Use headings (#, ##, ###) to structure sections.
- Prefer short paragraphs and concise lists.
- Use pipe tables with a header row; keep column counts modest.
- Avoid nested tables or complex HTML; keep to Markdown primitives.

## Citations
- Use [[S:n]] tokens inline after factual claims.
- If sources are required, include a References section at the end.
- Only include web sources that exist in the sources pool.

## Recommended Structure
# Title
## Executive Summary
## Findings
## Recommendations
## References

## Example (Markdown)
# Market Update
## Executive Summary
Short summary with citations [[S:1]].

## Findings
- Key point [[S:2]]
- Another point [[S:3]]

## Data Table
| Metric | Value | Source |
| --- | --- | --- |
| Growth | 12% | [[S:2]] |

## References
1. Source title (S:1)
2. Source title (S:2)
