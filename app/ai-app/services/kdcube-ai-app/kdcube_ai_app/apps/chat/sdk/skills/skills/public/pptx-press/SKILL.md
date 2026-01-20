---
name: pptx-press
description: |
  Teaches agents how to author slide-structured HTML that renders professionally to PPTX
  with proper sizing, styling, color schemes, and citation handling for business presentations,
  technical decks, and executive briefings.
version: 1.0.0
category: presentation-creation
tags:
  - pptx
  - slides
  - html
  - presentations
  - business
  - executive
when_to_use:
  - Generating HTML for write_pptx tool
  - Creating professional slide decks
  - Building citation-aware presentations
  - Designing executive briefings
  - Making technical presentations with data
author: kdcube
created: 2026-01-16
namespace: public
tools:
  - id: agent_tools.write_pptx
    role: presentation-generation
    purpose: Renders slide-structured HTML to PPTX using python-pptx
---

# PPTX Authoring for Professional Slide Decks

## Overview
This skill teaches how to produce HTML that renders cleanly into PPTX slides with professional
styling, proper content density, and reliable layouts. Focus: business presentations, technical
decks, and executive briefings.

## Core Principles

### 1. Content Density
Slides auto-scale content down (to ~70% minimum) if it exceeds available space. Always budget
content to fit comfortably within slide dimensions without requiring aggressive scaling.

### 2. Professional Styling
Use domain-appropriate color schemes, clear hierarchy, and balanced layouts. Avoid decorative
elements that don't serve the message.

### 3. Citation Integration
Citations should be concise inline markers `[n]` with full details in a Sources slide, not
disruptive to slide flow.

## HTML Structure

### Slide Container
```html
<section id="slide-1">
  <h1>Slide Title</h1>
  <p class="subtitle">Optional one-line subtitle</p>
  
  <!-- Body content here -->
</section>
```

**Rules:**
- One `<section>` per slide
- `id` can be any unique identifier
- `<h1>` is the slide title (required for most slides)
- `<p class="subtitle">` for optional subtitle
- All body content goes after title/subtitle

### Supported Body Elements

**Headings:**
```html
<h2>Section Heading</h2>
<h3>Subsection Heading</h3>
```

**Paragraphs:**
```html
<p>Body text with <strong>bold</strong> and <em>italic</em> formatting.</p>
```

**Lists:**
```html
<ul>
  <li><strong>Point:</strong> Short explanation</li>
  <li><strong>Another:</strong> Keep bullets concise</li>
</ul>
```

**Callout Boxes:**
```html
<div class="highlight-box">
  <h3>Key Insight</h3>
  <p>Important information that needs visual emphasis.</p>
</div>
```

**Two-Column Layout:**
```html
<div class="two-column">
  <div class="column">
    <h3>Left Side</h3>
    <ul>
      <li>Point one</li>
      <li>Point two</li>
    </ul>
  </div>
  <div class="column">
    <h3>Right Side</h3>
    <p>Comparison or complementary content.</p>
  </div>
</div>
```

**Tables:**
```html
<table>
  <thead>
    <tr><th>Metric</th><th>Q3</th><th>Q4</th></tr>
  </thead>
  <tbody>
    <tr><td>Revenue</td><td>$2.5M</td><td>$3.1M</td></tr>
  </tbody>
</table>
```

**Images:**
```html
<img src="turn_id/files/chart.png" width="640" alt="Revenue Chart">
<!-- or with inline style -->
<img src="images/diagram.png" style="width:5in; height:3in;" alt="Architecture">
```

## Supported CSS

### Colors (hex only: #RGB or #RRGGBB)
```css
color: #333333;
background: #f0f4f8;
background-color: #0066cc;
```

### Typography
```css
font-size: 18pt;  /* or 1.2em, 16px */
line-height: 1.3; /* number only */
```

### Spacing
```css
padding: 0.2in;
padding-left: 0.3in;
padding-top: 0.1in;
/* Supports: px, pt, in */
```

### Borders (for accents and underlines)
```css
border-bottom: 2px solid #0066cc;  /* Title underline */
border-left: 4px solid #ff6b35;    /* Callout accent */
```

### Two-Column Specific
```css
.two-column { gap: 0.3in; }
.column { 
  background: #f8f9fa; 
  padding: 0.2in;
  border-left: 3px solid #0066cc;
}
```

### Table Styling
```css
th { 
  background-color: #0066cc; 
  color: #ffffff; 
}
tr:nth-child(even) { 
  background-color: #f0f4f8; 
}
```

## DO NOT USE (Ignored by Renderer)

❌ **Layout:**
- `min-height`, `100vh`, `max-height`
- `flex`, `grid` (except implicit in `.two-column`)
- `position: absolute/fixed`
- `page-break-*` properties

❌ **Visual Effects:**
- `box-shadow`, `border-radius`, gradients
- `transform`, `transition`, `animation`
- `opacity` (partial support only)

❌ **Global Resets:**
- `* { ... }` selectors
- Large wrapper divs with generic styling

## Content Budgets (Critical for Proper Fitting)

### Standard Slide
- **1 heading** (h2 or h3)
- **6 short bullets** OR **2 short paragraphs** (~25-40 words each)
- **1 callout box** (~25-40 words)
- Tables: ≤6 columns, ≤8 rows

### Two-Column Slide
- Each column: **1 h3 + 3 bullets** OR **2 short paragraphs**
- Maximum ~12 lines per column
- Keep columns balanced in height

### Tables
- ≤6 columns (wider gets cramped)
- ≤8 rows (taller requires scaling)
- No merged cells
- Keep cell content concise

### Titles & Subtitles
- Title: ≤8 words (one line preferred)
- Subtitle: ≤12 words (one sentence max)

## Professional Color Schemes

### Business/Corporate
```css
:root {
  --primary: #0066cc;      /* IBM Blue */
  --accent: #00a86b;       /* Success Green */
  --text: #333333;         /* Dark Gray */
  --bg-light: #f0f4f8;     /* Light Blue-Gray */
  --bg-callout: #e8f2ff;   /* Soft Blue */
}

h1 { color: var(--primary); border-bottom: 3px solid var(--primary); }
.highlight-box { background: var(--bg-callout); border-left: 4px solid var(--primary); }
th { background-color: var(--primary); color: white; }
```

### Tech/Engineering
```css
:root {
  --primary: #1e3a8a;      /* Deep Blue */
  --accent: #7c3aed;       /* Purple */
  --text: #1f2937;
  --bg-light: #f8fafc;
  --bg-callout: #ede9fe;   /* Light Purple */
}
```

### Financial
```css
:root {
  --primary: #1e40af;      /* Navy */
  --accent: #059669;       /* Emerald */
  --text: #111827;
  --bg-light: #f0f9ff;
  --bg-callout: #dbeafe;
}
```

### Healthcare/Medical
```css
:root {
  --primary: #0d9488;      /* Teal */
  --accent: #0ea5e9;       /* Sky Blue */
  --text: #1f2937;
  --bg-light: #f0fdfa;
  --bg-callout: #ccfbf1;
}
```

### Executive/Premium
```css
:root {
  --primary: #1f2937;      /* Charcoal */
  --accent: #991b1b;       /* Burgundy */
  --text: #111827;
  --bg-light: #fef3c7;     /* Cream */
  --bg-callout: #fef3c7;
}
```

## Citations

### Inline Citation Format
```html
<p>
  Global EV sales grew ~35% YoY in 2024
  <sup class="cite" data-sids="1,3">[[S:1,3]]</sup>.
</p>
```

**Rules:**
- Add immediately after the factual claim
- `data-sids` contains numeric source IDs (comma-separated or range like 2-4)
- Inner text `[S:...]` must mirror `data-sids`
- Renders as concise `[1] · [3]` with hyperlinks

### Alternative: Footnotes Block
```html
<div class="footnotes">
  <p>Sources: [S:1], [S:3], [S:5]</p>
</div>
```

### Sources Slide
When sources are provided and `include_sources_slide=True`, a final "Sources" slide is
auto-generated with:
- [n] Title (domain)
- Clickable hyperlinks to sources

## Complete Slide Templates

### Template 1: Executive Summary
```html
<section id="slide-1">
  <h1>Executive Summary</h1>
  <p class="subtitle">Q4 2025 Performance Highlights</p>
  
  <h2>Key Achievements</h2>
  <ul>
    <li><strong>Revenue:</strong> $3.2M (+28% YoY)</li>
    <li><strong>Growth:</strong> 450 new customers</li>
    <li><strong>Efficiency:</strong> 35% cost reduction</li>
  </ul>
  
  <div class="highlight-box">
    <h3>Strategic Priority</h3>
    <p>Expand into EMEA market Q1 2026 with localized offerings.</p>
  </div>
</section>
```

### Template 2: Data Comparison (Two-Column)
```html
<section id="slide-2">
  <h1>Market Analysis</h1>
  
  <div class="two-column">
    <div class="column">
      <h3>Opportunities</h3>
      <ul>
        <li>Growing EV demand (+40%)</li>
        <li>Policy tailwinds in EU</li>
        <li>Tech partnerships</li>
      </ul>
    </div>
    <div class="column">
      <h3>Challenges</h3>
      <ul>
        <li>Supply chain volatility</li>
        <li>Intense competition</li>
        <li>Regulatory uncertainty</li>
      </ul>
    </div>
  </div>
</section>
```

### Template 3: Data Table
```html
<section id="slide-3">
  <h1>Quarterly Metrics</h1>
  
  <table>
    <thead>
      <tr>
        <th>Metric</th>
        <th>Q3 2025</th>
        <th>Q4 2025</th>
        <th>Change</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Revenue</td>
        <td>$2.5M</td>
        <td>$3.2M</td>
        <td>+28%</td>
      </tr>
      <tr>
        <td>Customers</td>
        <td>1,200</td>
        <td>1,650</td>
        <td>+38%</td>
      </tr>
    </tbody>
  </table>
</section>
```

### Template 4: With Citations
```html
<section id="slide-4">
  <h1>Market Trends</h1>
  
  <h2>Electric Vehicle Adoption</h2>
  <p>
    Global EV sales reached 14M units in 2024, representing ~18% of total 
    auto sales <sup class="cite" data-sids="1">[[S:1]]</sup>. Growth is 
    accelerated by policy incentives and falling battery costs 
    <sup class="cite" data-sids="2,3">[[S:2,3]]</sup>.
  </p>
</section>
```

## Common Mistakes to Avoid

### ❌ Content Overload
```html
<!-- BAD: 12 bullets won't fit -->
<ul>
  <li>Point 1 with very long explanation...</li>
  <li>Point 2 with even more text...</li>
  <!-- 10 more bullets -->
</ul>
```
**Fix:** Split into 2-3 slides with 4-6 bullets each.

### ❌ Tiny Columns
```html
<!-- BAD: Each column has 8 bullets -->
<div class="two-column">
  <div class="column">
    <ul><li>...</li><!-- 8 items --></ul>
  </div>
  <div class="column">
    <ul><li>...</li><!-- 8 items --></ul>
  </div>
</div>
```
**Fix:** Max 3-4 bullets per column.

### ❌ Unsupported CSS
```html
<style>
/* BAD: These are ignored */
.box { 
  box-shadow: 0 2px 8px rgba(0,0,0,0.1); 
  border-radius: 8px;
  transform: scale(1.1);
}
</style>
```
**Fix:** Use only supported properties.

### ❌ Base64 Images
```html
<!-- BAD: Will trigger warning -->
<img src="data:image/png;base64,iVBORw0KG...">
```
**Fix:** Use file paths: `<img src="turn_id/files/chart.png">`

### ❌ Long Titles
```html
<!-- BAD: Title wraps to 3 lines -->
<h1>Comprehensive Analysis of Q4 2025 Financial Performance and Strategic Market Positioning</h1>
```
**Fix:** `<h1>Q4 2025 Financial Performance</h1>` (use subtitle for details)

## Professional Practices

### 1. Visual Hierarchy
```html
<style>
h1 { font-size: 36pt; color: var(--primary); }
h2 { font-size: 28pt; color: var(--primary); margin-top: 0.2in; }
h3 { font-size: 22pt; color: var(--text); }
p { font-size: 18pt; line-height: 1.3; }
</style>
```

### 2. Balanced Layouts
- Single-column: Center-aligned, clear hierarchy
- Two-column: Equal visual weight, related content
- Mixed: Vary layouts between slides for visual interest

### 3. Callout Usage
Use callouts for:
- Key takeaways
- Action items
- Important warnings
- Strategic priorities

Avoid:
- Decorative boxes with no purpose
- Multiple callouts per slide (use max 1-2)

### 4. Color Consistency
- Stick to 2-3 colors per deck (primary + accent + neutrals)
- Use primary for headings and key elements
- Use accent sparingly for emphasis
- Keep backgrounds light for readability

## Image Guidelines

### File Paths (Required)
```html
<!-- Relative from OUT_DIR -->
<img src="turn_id/files/revenue_chart.png" width="640">

<!-- With explicit dimensions -->
<img src="images/architecture.png" style="width:6in; height:4in;">
```

### Sizing
- Specify width OR width+height
- Use inches for print consistency: `width:5in; height:3in;`
- Use pixels for web-friendly: `width="640"` (converts to ~6.7in at 96dpi)
- Images auto-fit to slide width if too large

### Placement
- Top-level images appear below title
- Images in columns appear inline with column content
- Images in callouts appear inside the callout box

## Performance Tips

1. **Keep HTML compact** - Remove unnecessary whitespace
2. **Minimize CSS** - Define common colors as variables, reuse classes
3. **Consolidate slides** - Don't split content unnecessarily
4. **Test content density** - If unsure, err on less content per slide

## Remember
- Content auto-scales down (min 70%) if too large - avoid this by budgeting properly
- Renderer measures text to determine fit - longer text = more scaling
- Citations render as `[n]` inline - keep factual claims concise
- Sources slide is auto-generated - don't create manually
- Two-column backgrounds/borders are per-column - style individually