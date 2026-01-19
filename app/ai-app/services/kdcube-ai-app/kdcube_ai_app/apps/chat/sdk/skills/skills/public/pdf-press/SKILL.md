---
name: pdf-press
description: |
  Teaches agents how to generate HTML and Markdown that renders beautifully to PDF 
  with proper page breaks, compact professional layouts, domain-adaptive color schemes, 
  and multi-column support for technical reports, scientific papers, and magazine-style 
  documents.
version: 1.0.0
category: document-creation
tags:
  - pdf
  - html
  - css
  - layout
  - typography
  - print-design
when_to_use:
  - Creating technical reports with multiple pages
  - Building scientific papers with two-column layouts
  - Designing magazine-style documents with varied column counts
  - Ensuring content doesn't split awkwardly across page boundaries
  - Generating professional PDFs with domain-adaptive color schemes
  - Working with write_pdf tool for HTML rendering
author: kdcube
created: 2026-01-16
namespace: public
import:
  - internal.link-evidence 
  - internal.sources-section
---

# PDF Authoring for Professional Documents

## Overview
This skill teaches Claude how to generate HTML and Markdown content that renders beautifully
to PDF using Playwright + headless Chromium. It covers professional layout patterns,
page-break control, compact spacing, domain-adaptive color schemes, and multi-column layouts
for technical reports, scientific papers, and magazine-style documents.

## When to Use This Skill
- Generating HTML for the write_pdf tool
- Creating multi-page technical reports that need proper pagination
- Building scientific papers with two-column layouts
- Designing magazine-style documents with varied column counts
- Ensuring content doesn't split awkwardly across page boundaries
- Adapting color schemes to match document domain (tech, medical, business, etc.)

## Core Principles

### 1. Page-Aware Layout
Always consider the printable page height (A4 portrait ≈ 257mm) when structuring content:
- Wrap logical units in `break-inside:avoid` containers
- Keep individual sections under 220mm height
- Split large content into multiple breakable sections

### 2. Compact Professional Spacing
Avoid magazine-style decorative layouts for technical documents:
- Use tight padding (10-14px, not 20-40px)
- Minimize section margins (12-16px, not 24-30px)
- Avoid banner-style headers that waste 50mm+ of vertical space
- Budget vertical space: 80% of PDF failures are due to excessive spacing

### 3. Domain-Adaptive Color Schemes
Choose color palettes that match the document's subject matter:
- Tech/Engineering: Blues and slate grays
- Business/Finance: Navy and emerald
- Medical/Health: Teal and clinical blues
- Creative/Design: Purple and coral
- Define colors as CSS variables for consistency

### 4. Semantic Break Points
Let the browser break naturally between sections, not within them:
- Each section = one cohesive idea
- Always wrap `<h2>` + content in same `break-inside:avoid` container
- Prevents heading splits like "Simulation &" / "Optimization" across pages

## Implementation Guidelines

### Basic Document Structure

**Single-column technical report (most common):**
```html
<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8'>
  <style>
  @page { size: A4 portrait; margin: 20mm; }
  
  :root {
    --primary: #1e3a8a;
    --text: #1f2937;
    --bg-alt: #f8f9fa;
    --border: #e5e7eb;
    --text-muted: #6b7280;
  }
  
  * { margin: 0; padding: 0; box-sizing: border-box; }
  
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 10pt;
    line-height: 1.5;
    color: var(--text);
  }
  
  h1 {
    font-size: 18pt;
    color: var(--primary);
    margin: 0 0 6px;
    padding-bottom: 6px;
    border-bottom: 2px solid var(--primary);
  }
  
  h2 {
    font-size: 14pt;
    color: var(--primary);
    margin: 16px 0 8px;
  }
  
  h3 {
    font-size: 11pt;
    font-weight: 600;
    margin: 12px 0 6px;
  }
  
  p { margin: 6px 0; }
  
  .metadata {
    font-size: 8pt;
    color: var(--text-muted);
    margin: 4px 0 16px;
  }
  
  section {
    break-inside: avoid;
    page-break-inside: avoid;
    margin-bottom: 16px;
  }
  
  .card {
    background: var(--bg-alt);
    border-left: 3px solid var(--primary);
    padding: 10px 14px;
    margin: 12px 0;
    break-inside: avoid;
  }
  
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 8.5pt;
    line-height: 1.3;
    margin: 8px 0;
  }
  
  thead {
    background: var(--primary);
    color: white;
  }
  
  th, td {
    padding: 4px 6px;
    border: 1px solid var(--border);
    text-align: left;
    vertical-align: top;
  }
  
  .table-wrapper {
    break-inside: avoid;
    margin: 12px 0;
  }
  </style>
</head>
<body>
  <h1>Document Title</h1>
  <p class='metadata'>Author • 2026-01-16 • Report Type</p>
  
  <section>
    <h2>Introduction</h2>
    <p>Content that stays together with its heading...</p>
  </section>
  
  <section>
    <h2>Analysis</h2>
    <div class='card'>
      <h3>Key Finding</h3>
      <p>Important information in compact card...</p>
    </div>
    
    <div class='table-wrapper'>
      <h3>Data Summary</h3>
      <table>
        <thead>
          <tr><th>Metric</th><th>Value</th></tr>
        </thead>
        <tbody>
          <tr><td>Item 1</td><td>100</td></tr>
        </tbody>
      </table>
    </div>
  </section>
</body>
</html>
```

### Color Scheme Selection

Choose based on document domain:

**Tech/Engineering:**
```css
:root {
  --primary: #1e3a8a;
  --accent: #3b82f6;
  --text: #1f2937;
  --bg-alt: #f8fafc;
  --border: #cbd5e1;
}
```

**Business/Finance:**
```css
:root {
  --primary: #1e40af;
  --accent: #059669;
  --text: #1f2937;
  --bg-alt: #f0f9ff;
  --border: #bae6fd;
}
```

**Medical/Health:**
```css
:root {
  --primary: #0d9488;
  --accent: #0ea5e9;
  --text: #1f2937;
  --bg-alt: #f0fdfa;
  --border: #99f6e4;
}
```

### Preventing Page Breaks

**Critical pattern - always wrap heading + content:**
```html
<!-- ❌ BAD: Heading can split from content -->
<h2>Scenario 3: AI-Driven Simulation & Optimization</h2>
<p>Description of the scenario...</p>

<!-- ✅ GOOD: Heading and content stay together -->
<section style='break-inside:avoid;'>
  <h2>Scenario 3: AI-Driven Simulation & Optimization</h2>
  <p>Description of the scenario...</p>
</section>
```

### Compact Spacing Measurements

Budget your vertical space wisely:

| Element | Compact (✅) | Wasteful (❌) | Savings |
|---------|-------------|---------------|---------|
| Document title | 30-40mm | 60-100mm | 30-60mm |
| Card padding | 10-14px | 20-40px | 10-26px per card |
| Section margins | 12-16px | 24-30px | 12-14px per section |
| Table cell padding | 4-6px | 8-10px | 4-8px per row |
| Line-height (body) | 1.5 | 1.6-1.8 | 1mm per 10 lines |
| Line-height (tables) | 1.3 | 1.5 | 2mm per row |

**Vertical space budget for A4 portrait (257mm printable):**
- Compact header: 30-40mm
- Executive summary: 60-80mm
- Remaining for content: 137-167mm
- Each section: ~40-80mm (fits 2-4 per page)

### Table Best Practices

**Compact table that fits on one page:**
```html
<div class='table-wrapper' style='break-inside:avoid; margin:12px 0;'>
  <h3 style='font-size:11pt; margin-bottom:6px;'>Staffing Profile</h3>
  <table style='width:100%; border-collapse:collapse; font-size:8.5pt; line-height:1.3;'>
    <thead style='background:var(--primary); color:white;'>
      <tr>
        <th style='padding:4px 6px;'>Role</th>
        <th style='padding:4px 6px;'>FTE</th>
        <th style='padding:4px 6px;'>Months</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style='border:1px solid var(--border); padding:4px 6px;'>ML Engineer</td>
        <td style='border:1px solid var(--border); padding:4px 6px;'>2</td>
        <td style='border:1px solid var(--border); padding:4px 6px;'>10</td>
      </tr>
    </tbody>
  </table>
</div>
```

**Maximum rows before splitting:**
- 8.5pt font + 4px padding + 1.3 line-height ≈ 5-6mm per row
- A4 portrait content area ≈ 220mm safe zone
- Maximum: ~12-15 rows per table before requiring split

## Common Patterns

### Pattern 1: Multi-Section Technical Report
```html
<!-- Compact header -->
<h1>Comprehensive Staffing Model</h1>
<p class='metadata'>Author • 2026-01-16 • Resource Planning</p>

<!-- Executive summary in compact card -->
<div class='card' style='break-inside:avoid;'>
  <strong>Executive Summary:</strong> This document provides detailed 
  staffing models for five scenarios...
</div>

<!-- Each scenario as breakable section -->
<section style='break-inside:avoid;'>
  <h2>Scenario 1: Automated Design</h2>
  <p>Description...</p>
  
  <div class='table-wrapper' style='break-inside:avoid;'>
    <h3>Staffing Profile</h3>
    <table>...</table>
  </div>
</section>

<section style='break-inside:avoid;'>
  <h2>Scenario 2: CAD Modeling</h2>
  <p>Description...</p>
</section>
```

### Pattern 2: Scientific Paper (Two-Column)
```html
<style>
.paper-header { max-width: 700px; margin: 0 auto 20px; text-align: center; }
.abstract { 
  max-width: 600px; 
  margin: 0 auto 16px; 
  font-size: 9.5pt; 
  border: 1px solid var(--border); 
  padding: 12px 16px; 
  background: var(--bg-alt); 
}
.two-column-body {
  column-count: 2;
  column-gap: 20px;
  max-width: 700px;
  margin: 0 auto;
  text-align: justify;
}
.two-column-body h2 {
  column-span: all;
  font-size: 12pt;
  margin: 16px 0 10px;
}
.two-column-body figure {
  column-span: all;
  break-inside: avoid;
  margin: 16px 0;
}
</style>

<div class='paper-header'>
  <h1 style='font-size:16pt;'>Paper Title</h1>
  <p class='metadata'>Authors • Institution</p>
</div>

<div class='abstract'>
  <strong>Abstract.</strong> Summary text...
</div>

<div class='two-column-body'>
  <h2>1. Introduction</h2>
  <p>Text flows in two columns...</p>
  
  <figure>
    <img src='files/fig1.png' style='max-width:100%; height:auto;'>
    <figcaption>Figure 1: Description</figcaption>
  </figure>
  
  <h2>2. Methods</h2>
  <p>More content...</p>
</div>
```

## Examples

### ✅ Good Example: Compact Professional Report
```html
<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8'>
  <style>
  @page { size: A4 portrait; margin: 20mm; }
  :root { --primary: #1e3a8a; --text: #1f2937; --bg-alt: #f8f9fa; --border: #e5e7eb; }
  body { font-family: -apple-system, sans-serif; font-size: 10pt; line-height: 1.5; }
  h1 { font-size: 18pt; color: var(--primary); margin: 0 0 6px; padding-bottom: 6px; border-bottom: 2px solid var(--primary); }
  h2 { font-size: 14pt; color: var(--primary); margin: 16px 0 8px; }
  section { break-inside: avoid; margin-bottom: 16px; }
  .card { background: var(--bg-alt); border-left: 3px solid var(--primary); padding: 10px 14px; margin: 12px 0; }
  </style>
</head>
<body>
  <h1>Quarterly Report</h1>
  <p style='font-size:8pt; color:#6b7280; margin:4px 0 16px;'>Q4 2025 • Finance Team</p>
  
  <section>
    <h2>Revenue Summary</h2>
    <p>Total revenue increased by 15%...</p>
    <div class='card'>
      <strong>Key Metric:</strong> $2.5M ARR achieved
    </div>
  </section>
</body>
</html>
```
**Why this is good:**
- Compact header (only 30mm)
- Sections wrapped in break-inside:avoid
- Tight padding (10-14px)
- Professional color scheme
- Will fit 3-4 sections per page

### ❌ Bad Example: Wasteful Banner Layout
```html
<!DOCTYPE html>
<html>
<head>
  <style>
  /* No @page definition */
  body { font-family: Arial; font-size: 12pt; }  /* Font too large */
  </style>
</head>
<body>
  <!-- Giant banner wastes 100mm -->
  <div style='background:#2c5aa0; color:white; padding:50px; text-align:center;'>
    <h1 style='font-size:28pt; margin:40px 0;'>Quarterly Report</h1>
    <p style='font-size:14pt; margin:20px 0;'>Q4 2025</p>
  </div>
  
  <div style='padding:30px; margin:30px 0;'>  <!-- No break-inside:avoid -->
    <h2 style='font-size:20pt;'>Revenue Summary</h2>  <!-- Not wrapped with content -->
  </div>
  <p style='margin:20px 0;'>Total revenue...</p>  <!-- Can split from heading -->
</body>
</html>
```
**Why this is bad:**
- Banner header wastes 100mm+ of first page
- No @page definition
- No break-inside:avoid (heading will split from content)
- Excessive padding (50px, 30px)
- Excessive margins (30px, 40px)
- Font sizes too large (28pt, 20pt, 14pt)
- Will only fit 1-2 sections per page

## Common Mistakes to Avoid

1. **Banner-style headers**
    - ❌ `<div style='padding:40px; background:#blue;'><h1>Title</h1></div>` (wastes 100mm)
    - ✅ `<h1 style='padding-bottom:6px; border-bottom:2px solid;'>Title</h1>` (uses 20mm)

2. **Not wrapping headings with content**
    - ❌ `<h2>Long Title</h2><p>Content</p>` (heading can split mid-word across pages)
    - ✅ `<section style='break-inside:avoid;'><h2>Long Title</h2><p>Content</p></section>`

3. **Excessive padding and margins**
    - ❌ Card: `padding:24px 40px; margin:30px 0;` (wastes 40mm per card)
    - ✅ Card: `padding:10px 14px; margin:12px 0;` (saves 30mm per card)

4. **Tables without break-inside:avoid wrapper**
    - ❌ `<table>...</table>` (can split between title and table)
    - ✅ `<div style='break-inside:avoid;'><h3>Table Title</h3><table>...</table></div>`

5. **Using body line-height >1.6**
    - ❌ `body { line-height: 1.8; }` (reduces content density)
    - ✅ `body { line-height: 1.5; }` (readable + space-efficient)

6. **Table line-height >1.4**
    - ❌ `table { line-height: 1.6; }` (tables become too tall)
    - ✅ `table { line-height: 1.3; }` (compact rows)

7. **Not defining @page**
    - ❌ Missing `@page { size: A4 portrait; margin: 20mm; }`
    - ✅ Always define page size and margins

8. **Using base64 images**
    - ❌ `<img src='data:image/png;base64,...'>` (crashes multi-page PDFs)
    - ✅ `<img src='files/chart.png'>` (relative file paths)

## Quick Reference

### Essential CSS Properties
```css
/* Page setup */
@page { size: A4 portrait; margin: 20mm; }

/* Break control */
break-inside: avoid;
page-break-inside: avoid;
page-break-before: always;  /* Force new page */
page-break-after: always;

/* Column layouts */
column-count: 2;
column-gap: 20px;
column-span: all;  /* Break out of columns */
```

### Spacing Standards (Compact Professional)
```css
/* Typography */
body: 10pt, line-height: 1.5
h1: 18pt max (16pt for academic)
h2: 14pt (12pt for academic)
h3: 11-12pt
Captions: 8.5pt
Tables: 8.5pt, line-height: 1.3

/* Spacing */
Card padding: 10-14px
Section margins: 12-16px
Table cell padding: 4-6px
Paragraph margins: 6px
```

### Height Budgets
```
A4 Portrait (257mm printable):
- Header: 30-40mm
- Summary: 60-80mm
- Section: 40-80mm (2-4 per page)
- Table row: 5-6mm (max 12-15 rows)

A4 Landscape (177mm printable):
- Section: 30-60mm
- Table row: 5-6mm (max 8-10 rows)
```

### Domain Color Palettes
```css
/* Tech/Engineering */
--primary: #1e3a8a; --accent: #3b82f6; --bg-alt: #f8fafc;

/* Business/Finance */
--primary: #1e40af; --accent: #059669; --bg-alt: #f0f9ff;

/* Medical/Health */
--primary: #0d9488; --accent: #0ea5e9; --bg-alt: #f0fdfa;

/* Creative/Design */
--primary: #7c3aed; --accent: #f97316; --bg-alt: #faf5ff;

/* Legal/Formal */
--primary: #1f2937; --accent: #991b1b; --bg-alt: #fef3c7;

/* Academic/Research */
--primary: #1e3a8a; --accent: #f59e0b; --bg-alt: #f8f9fa;
```

## Advanced Topics

### Multi-Column Layouts

**Two-column scientific paper:**
- Full-width header and abstract
- Two-column body with column-span for figures
- Each column ~330px wide
- Max unbreakable element: 180mm tall

**Three-column magazine:**
- Landscape orientation only
- Each column ~280px wide
- Max unbreakable element: 120mm tall
- Use pull-quotes and sidebars for variety

### Responsive Images
```html
<figure style='break-inside:avoid; margin:16px 0; text-align:center;'>
  <img src='files/chart.png' alt='Revenue Chart' 
       style='max-width:100%; height:auto; display:block; margin:0 auto;'>
  <figcaption style='font-size:8.5pt; color:#6b7280; margin-top:6px;'>
    Figure 1: Quarterly revenue trends
  </figcaption>
</figure>
```

### Complex Tables
For tables >12 rows:
1. Add subheadings to create semantic breaks
2. Split into multiple tables
3. Each table wrapped in break-inside:avoid
4. Consider landscape orientation for wide tables

## Remember
- **Budget vertical space** - 80% of PDF issues are spacing-related
- **Wrap headings with content** - Prevents "Simulation &" / "Optimization" splits
- **Use compact measurements** - 10-14px padding, 12-16px margins, 4-6px table cells
- **Choose domain colors** - Tech=blue, Medical=teal, Business=navy, etc.
- **Test page breaks** - Preview PDF to ensure sections don't split awkwardly
- **Prioritize content density** - Technical reports need information, not decoration
