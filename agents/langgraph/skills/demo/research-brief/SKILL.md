---
name: research-brief
id: research-brief
description: "Turn sourced web findings into a concise PDF brief and an auditable XLSX evidence table."
version: 1.0.0
category: research
tags: ["research", "sources", "pdf", "xlsx"]
when_to_use:
  - Summarizing public-web research into a brief
  - Producing a spreadsheet that preserves source URLs
namespace: demo
---
# Research Brief

Keep the evidence reusable:

1. Preserve the title, finding, and source URL for every selected result.
2. Distinguish observed source facts from your synthesis.
3. Author Python for structured computation and XLSX creation, then execute it through the isolated Python tool.
4. Author renderer-ready source separately: HTML for PDF/PPTX, or Markdown for DOCX.
5. Use the matching KDCube rendering tool instead of generating document-format bytes in Python.
6. Verify every promised output exists and is non-empty before reporting completion.
