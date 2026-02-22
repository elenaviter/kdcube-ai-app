---
name: png-press
description: |
  Teaches agents how to author content and configure write_png so PNGs render
  at the correct size, with tight cropping, readable text, and consistent scaling.
version: 1.0.0
category: document-creation
tags:
  - png
  - rendering
  - html
  - markdown
  - mermaid
when_to_use:
  - Rendering Mermaid diagrams to PNG
  - Rendering HTML layouts to PNG
  - Rendering Markdown snippets to PNG
  - Fixing tiny or mostly-blank PNG output
  - Improving readability via zoom/scale
  - Verifying Mermaid renders correctly (syntax + visual check) when needed
author: kdcube
created: 2026-02-18
namespace: public
---

# PNG Authoring (write_png)

## Overview
The `write_png` tool renders HTML, Markdown, or Mermaid into PNG via headless Chromium.
The most common failure is **tiny diagrams centered in a large blank canvas**.
Use **fit='content'** (default) + correct selectors/width/zoom to crop and scale properly.

## Golden Rules

1) **Always control sizing and cropping**
- Use `fit='content'` to crop to the real content bounds.
- For HTML, set `content_selector` to a stable wrapper (e.g. `#render-root`).
- Add `padding_px` for margin around the crop.

2) **Use the correct format**
- Mermaid diagrams: `format='mermaid'` with raw Mermaid text (no ``` fences).
- Mixed text + diagram: use `format='markdown'` and fenced ```mermaid blocks.
- Custom layout: `format='html'` with a single wrapper element.

3) **Scale for readability**
- Increase viewport `width` for wide diagrams (2200–3200px typical).
- Use `zoom=1.2–1.8` when text is too small.
- Raise `device_scale_factor` (2 or 3) for crisper output.
- For Mermaid, prefer `mermaid_font_size_px` (e.g. 16–22) and/or `mermaid_scale` (1.1–1.6).
- Optionally set `mermaid_font_family` for consistent typography.

4) **Allow time for layout**
- Mermaid and JS charts need time to settle.
- Use `render_delay_ms=1000–2000` for complex diagrams.

5) **Avoid base64 images**
- Reference local assets under OUT_DIR using relative paths.
- Base64 data URIs can fail or bloat headless rendering.

## HTML Mode: Recommended Structure
Wrap your content in a root container so cropping is reliable.

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    body { margin: 0; background: white; }
    #render-root {
      display: inline-block;
      padding: 16px;
      font-family: Arial, sans-serif;
    }
  </style>
</head>
<body>
  <div id="render-root">
    <!-- content here -->
  </div>
</body>
</html>
```

Then call:
- `format='html'`
- `content_selector='#render-root'`
- `fit='content'`

## Mermaid Mode: Recommended Use
- Provide **raw Mermaid** only.
- Use `format='mermaid'` and set `width` high enough for the diagram.
- If diagram is small: increase `zoom`.
- When needed to validate Mermaid syntax visually, render to PNG and check the output for parse errors or missing nodes.

Example:
```
write_png(
  path="diagram.png",
  content="graph LR\nA[\"Start\"]-->B[\"Process\"]",
  format="mermaid",
  width=2400,
  zoom=1.4,
  mermaid_font_size_px=18,
  device_scale_factor=3,
  fit="content",
  render_delay_ms=1200
)
```

If Mermaid syntax breaks, use the **public.mermaid** skill to fix labels/quotes.

## Markdown Mode: When to Use
- Use for mixed text + diagrams.
- Prefer small markdown documents; large layouts are better in HTML.

Example:
```
write_png(
  path="summary.png",
  format="markdown",
  content="""
# Summary\n\n```mermaid\nflowchart LR\nA-->B\n```\n""",
  fit="content",
  width=2000
)
```

## Troubleshooting

**Problem: Tiny diagram in the middle of a blank page**
- Use `fit='content'` (default).
- Provide `content_selector` for HTML.
- Increase `zoom` and/or `width`.

**Problem: Diagram clipped on the right**
- Increase `width` or reduce diagram complexity.
- Avoid `max-width: 100%` on SVG if it shrinks.

**Problem: Text too small**
- Increase `zoom` to 1.4–1.8.
- Set `device_scale_factor=2` or `3`.
- For Mermaid, set `mermaid_font_size_px` (16–22) or `mermaid_scale` (1.2–1.5).

**Problem: Mermaid fails to render**
- Fix syntax (quote labels with punctuation).
- Use the `public.mermaid` skill.
