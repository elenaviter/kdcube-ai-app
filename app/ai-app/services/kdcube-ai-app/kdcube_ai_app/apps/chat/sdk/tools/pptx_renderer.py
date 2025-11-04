# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# chat/sdk/tools/pptx_renderer.py
# FINAL FIXED VERSION - H3 headers, proper height tracking, overflow prevention

from __future__ import annotations

import pathlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import re

from pptx import Presentation
from pptx.util import Pt, Inches
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR

from kdcube_ai_app.apps.chat.sdk.runtime.workdir_discovery import resolve_output_dir
import kdcube_ai_app.apps.chat.sdk.tools.md_utils as md_utils


# GPT
# Layout heuristics (tuned to avoid overlap)
TITLE_BAND_IN   = 0.90   # reserved height below slide title
BOTTOM_MARGIN_IN = 0.50  # keep off the footer area
BLOCK_GAP_IN     = 0.12  # vertical rhythm between blocks

# line-height multipliers (approx)
LINE_SPACING = 1.25
EMU_PER_IN = 914400

def _emu_to_in(v: int) -> float:
    return float(v) / EMU_PER_IN

def _in_to_emu(inches: float) -> int:
    return int(round(inches * EMU_PER_IN))

def _chars_per_line(width_in: float, font_pt: float, indent_level: int = 0) -> int:
    """
    Very conservative char-per-line estimate.
    Rough model: avg char width ≈ 0.55em; with bullets/indent we reduce effective width.
    """
    # effective width reduction per indent level (~0.35" per level)
    indent_in = max(0, indent_level) * 0.35
    eff_in = max(1.0, width_in - indent_in)
    # points per inch = 72; approx chars per em = 2; fudge factor 0.8 for safety
    chars = int((eff_in * 72 / font_pt) * 2 * 0.8)
    # cap to a sane range
    return max(25, min(95, chars))

def _lines_for_text(text: str, width_in: float, font_pt: float, indent_level: int = 0) -> int:
    # strip markdown emphasis markers for width calc
    stripped = re.sub(r"(\*\*|\*)", "", text)
    cpl = _chars_per_line(width_in, font_pt, indent_level)
    # very conservative break: count words/characters
    return max(1, (len(stripped) + cpl - 1) // cpl)

def _pt_to_in(pts: float) -> float:
    return pts / 72.0

# def _estimate_text_block_height(lines: List[str], width_in: float, font_pt: float) -> float:
#     """
#     Estimate height (in inches) for mixed bullet levels text block.
#     Each logical line is wrapped separately with a conservative model.
#     """
#     if not lines:
#         return 0.0
#     total_lines = 0
#     for ln in lines:
#         lvl, txt = _parse_bullet_level(ln)
#         total_lines += _lines_for_text(txt, width_in, font_pt, lvl)
#     line_height_in = _pt_to_in(font_pt) * LINE_SPACING
#     # add small top+bottom padding
#     return total_lines * line_height_in + 0.12
PARA_SPACE_AFTER_PT = 2      # you already use this in _style_paragraph
TF_MARGIN_IN = 0.10          # 0.05 top + 0.05 bottom (from _add_textbox)
HEIGHT_FUDGE = 1.10          # safety multiplier for long tokens/URLs

def _estimate_text_block_height(lines: List[str], width_in: float, font_pt: float) -> float:
    """
    Estimate height (in inches) for a list of paragraph lines (each becomes its own paragraph).
    Accounts for wrapping, per-paragraph spacing, text frame margins, and a safety fudge.
    """
    if not lines:
        return 0.0

    line_height_in = _pt_to_in(font_pt) * LINE_SPACING
    para_space_in = _pt_to_in(PARA_SPACE_AFTER_PT)

    total_visual_lines = 0
    nonempty_paras = 0

    for ln in lines:
        if not ln.strip():
            continue
        lvl, txt = _parse_bullet_level(ln)
        # be more conservative on width for deep indents
        visual_lines = _lines_for_text(txt, width_in * 0.95, font_pt, lvl)
        total_visual_lines += visual_lines
        nonempty_paras += 1

    if nonempty_paras == 0:
        return 0.0

    content_h = total_visual_lines * line_height_in
    # add paragraph spacing between paragraphs (n-1 gaps)
    spacing_h = max(0, nonempty_paras - 1) * para_space_in
    # add textframe top+bottom margins
    margins_h = TF_MARGIN_IN

    return (content_h + spacing_h + margins_h) * HEIGHT_FUDGE
# GPT


_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_CIT_RE  = re.compile(r"\[\[S:(\d+)\]\]")

def _outdir() -> pathlib.Path:
    return resolve_output_dir()

def _basename_only(path: str, default_ext: str = ".pptx") -> str:
    name = Path(path).name
    if default_ext and not name.lower().endswith(default_ext):
        name += default_ext
    return name

def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _domain_of(url: str) -> str:
    try:
        from urllib.parse import urlparse
        net = urlparse(url).netloc
        return net or url
    except Exception:
        return url

def _split_markdown_sections(md: str) -> List[Tuple[str, List[str]]]:
    """Split markdown into sections by ## headings."""
    lines = (md or "").splitlines()
    slides: List[Tuple[str, List[str]]] = []
    cur_title: Optional[str] = None
    cur_body: List[str] = []

    for ln in lines:
        if ln.startswith("## "):
            if cur_title is not None:
                slides.append((cur_title.strip(), cur_body))
            cur_title = ln[3:]
            cur_body = []
        elif ln.startswith("# "):
            if cur_title is None and not slides:
                cur_title = ln[2:]
                cur_body = []
            else:
                cur_body.append(ln)
        else:
            cur_body.append(ln)

    if cur_title is None:
        nonempty = next((l for l in lines if l.strip()), "Slides")
        cur_title = nonempty.lstrip("# ").strip() or "Slides"

    slides.append((cur_title.strip(), cur_body))
    return slides

# Styling
PALETTE = {
    "fg": RGBColor(20, 24, 31),
    "muted": RGBColor(95, 106, 121),
    "accent": RGBColor(31, 111, 235),
    "quote_bg": RGBColor(245, 247, 250),
    "code_bg": RGBColor(250, 250, 252),
    "rule": RGBColor(220, 224, 230),
    # "table_row_bg": RGBColor(255, 255, 255),      # even rows (base)
    "table_header_bg": RGBColor(240, 244, 252),
    # "table_header_fg": RGBColor(20, 24, 31),   # dark text in header
    # "table_row_alt_bg": RGBColor(232, 237, 245)  # light zebra stripe
}

TYPE_SCALE = {
    "title": Pt(36),
    "slide_title": Pt(26),
    "h3": Pt(20),       # For ### headers
    "body": Pt(15),     # Slightly smaller for better fit
    "code": Pt(12),
    "caption": Pt(11),
}

PAGE = {
    "content_left": Inches(0.8),
    "content_top": Inches(1.4),
    "content_width": Inches(8.4),  # 10.0 - 0.8 - 0.8
    "slide_height": Inches(7.5),
}

MONO_FALLBACK = "Consolas"

# Regex patterns
_CODE_FENCE_RE = re.compile(r"^```(\w+)?\s*$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$")
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s+(.*)$")

def _style_run(run, *, size: Pt, bold=False, italic=False, color: RGBColor = None, mono=False):
    run.font.size = size
    run.font.bold = bold
    run.font.italic = italic
    if mono:
        run.font.name = MONO_FALLBACK
    if color:
        run.font.color.rgb = color

def _style_paragraph(p, *, level=0, space_after=Pt(6), align=PP_ALIGN.LEFT):
    p.level = max(0, min(level, 4))
    p.space_after = space_after
    p.line_spacing = 1.15
    p.alignment = align

def _add_textbox(slide, left, top, width, height):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.0)
    tf.margin_right = Inches(0.0)
    tf.margin_top = Inches(0.05)
    tf.margin_bottom = Inches(0.05)
    return tb

def _add_title_slide(prs: Presentation, text: str) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    tbox = _add_textbox(slide, PAGE["content_left"], Inches(2.0), PAGE["content_width"], Inches(2.0))
    p = tbox.text_frame.paragraphs[0]
    _style_paragraph(p, align=PP_ALIGN.LEFT)
    r = p.add_run()
    r.text = (text or "Presentation").strip()
    _style_run(r, size=TYPE_SCALE["title"], bold=True, color=PALETTE["fg"])

def _add_slide_title(slide, title: str):
    tbox = _add_textbox(slide, PAGE["content_left"], PAGE["content_top"], PAGE["content_width"], Inches(0.6))
    p = tbox.text_frame.paragraphs[0]
    _style_paragraph(p, align=PP_ALIGN.LEFT, space_after=Pt(2))
    r = p.add_run()
    r.text = (title or "").strip()
    _style_run(r, size=TYPE_SCALE["slide_title"], bold=True, color=PALETTE["fg"])

def _parse_bullet_level(line: str) -> tuple[int, str]:
    """Extract bullet level and text."""
    m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
    if not m:
        return 0, line.strip()
    spaces, _bullet, text = m.groups()
    lvl = min(len(spaces) // 2, 4)
    return lvl, text.strip()

def _is_table_separator(cell: str) -> bool:
    """Check if a table cell is a separator (contains dashes)."""
    stripped = cell.strip()
    if not stripped:
        return False
    dash_count = stripped.count('-')
    return dash_count >= 3 and dash_count >= len(stripped) - 2

def _parse_table(lines: List[str]) -> Optional[List[List[str]]]:
    """Parse markdown table into rows/columns."""
    table_lines = [ln.strip() for ln in lines if _TABLE_ROW_RE.match(ln)]
    if len(table_lines) < 2:
        return None

    def split_row(line: str) -> List[str]:
        return [cell.strip() for cell in line.strip('|').split('|')]

    rows = [split_row(line) for line in table_lines]

    if len(rows) < 2:
        return None

    # Check if second row is separator
    separator_row = rows[1]
    if not all(_is_table_separator(cell) for cell in separator_row):
        return None

    # Return header + data rows (skip separator)
    header = rows[0]
    data_rows = rows[2:] if len(rows) > 2 else []

    return [header] + data_rows

def _emit_text_with_formatting(paragraph, text: str, sources_map: Dict[int, Dict[str, str]], resolve_citations: bool):
    """Add text to paragraph with bold/italic/links/citations."""
    # Split by bold first
    parts = re.split(r"(\*\*[^*]+\*\*)", text)

    for part in parts:
        if not part:
            continue

        if part.startswith("**") and part.endswith("**"):
            # Bold text
            inner = part[2:-2]
            # Check for italic within bold
            italic_parts = re.split(r"(\*[^*]+\*)", inner)
            for ipart in italic_parts:
                if ipart.startswith("*") and ipart.endswith("*"):
                    r = paragraph.add_run()
                    r.text = ipart[1:-1]
                    _style_run(r, size=TYPE_SCALE["body"], bold=True, italic=True, color=PALETTE["fg"])
                elif ipart:
                    r = paragraph.add_run()
                    r.text = ipart
                    _style_run(r, size=TYPE_SCALE["body"], bold=True, color=PALETTE["fg"])
        else:
            # Check for italic, links, citations
            italic_parts = re.split(r"(\*[^*]+\*)", part)
            for ipart in italic_parts:
                if not ipart:
                    continue

                if ipart.startswith("*") and ipart.endswith("*"):
                    r = paragraph.add_run()
                    r.text = ipart[1:-1]
                    _style_run(r, size=TYPE_SCALE["body"], italic=True, color=PALETTE["fg"])
                else:
                    # Handle links and citations
                    _emit_links_and_citations(paragraph, ipart, sources_map, resolve_citations)

def _emit_links_and_citations(paragraph, text: str, sources_map: Dict[int, Dict[str, str]], resolve_citations: bool):
    """Handle links [text](url) and citations [[S:n]]."""
    while text:
        m_link = _LINK_RE.search(text)
        m_cit = _CIT_RE.search(text) if resolve_citations else None

        matches = []
        if m_link:
            matches.append(("link", m_link))
        if m_cit:
            matches.append(("cit", m_cit))

        if not matches:
            # Plain text
            if text:
                r = paragraph.add_run()
                r.text = text
                _style_run(r, size=TYPE_SCALE["body"], color=PALETTE["fg"])
            return

        # Get earliest match
        kind, m = min(matches, key=lambda x: x[1].start())

        # Add text before match
        if m.start() > 0:
            r = paragraph.add_run()
            r.text = text[:m.start()]
            _style_run(r, size=TYPE_SCALE["body"], color=PALETTE["fg"])

        # Add match
        if kind == "link":
            r = paragraph.add_run()
            r.text = m.group(1)
            _style_run(r, size=TYPE_SCALE["body"], color=PALETTE["accent"])
            try:
                r.hyperlink.address = m.group(2)
            except:
                pass
        else:  # citation
            sid = int(m.group(1))
            rec = sources_map.get(sid, {})
            label = rec.get("title") or f"[{sid}]"
            url = rec.get("url", "")
            r = paragraph.add_run()
            r.text = label
            _style_run(r, size=TYPE_SCALE["body"], color=PALETTE["accent"])
            if url:
                try:
                    r.hyperlink.address = url
                except:
                    pass

        text = text[m.end():]

def _new_content_slide(prs: Presentation, title: str):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_slide_title(slide, title)

    # Keep anchors as EMUs
    top_anchor = PAGE["content_top"] + Inches(TITLE_BAND_IN)          # EMU int
    bottom_anchor = PAGE["slide_height"] - Inches(BOTTOM_MARGIN_IN)   # EMU int

    # Available height *in inches* for our estimators
    available_height_in = _emu_to_in(bottom_anchor - top_anchor)

    y_offset_in = 0.0  # we track runtime offset in inches, convert to EMU only when placing
    return slide, top_anchor, available_height_in, y_offset_in

def _peek_next_block(body_lines: List[str], start_idx: int) -> Tuple[str, int, List[str]]:
    """
    Return (kind, end_index, payload_lines) starting from start_idx.
    kind in {"h3","code","table","quote","text","blank"}
    end_index is the index AFTER the block.
    """
    n = len(body_lines)
    i = start_idx
    if i >= n:
        return "blank", i, []

    ln = body_lines[i]
    if not ln.strip():
        return "blank", i+1, []

    if ln.startswith("### "):
        return "h3", i+1, [ln[4:].strip()]

    if _CODE_FENCE_RE.match(ln):
        i += 1
        code = []
        while i < n and not _CODE_FENCE_RE.match(body_lines[i]):
            code.append(body_lines[i]); i += 1
        if i < n and _CODE_FENCE_RE.match(body_lines[i]):
            i += 1
        return "code", i, code

    if _TABLE_ROW_RE.match(ln):
        table_lines = []
        while i < n and _TABLE_ROW_RE.match(body_lines[i]):
            table_lines.append(body_lines[i]); i += 1
        return "table", i, table_lines

    if _BLOCKQUOTE_RE.match(ln):
        quotes = []
        while i < n and _BLOCKQUOTE_RE.match(body_lines[i]):
            m = _BLOCKQUOTE_RE.match(body_lines[i])
            quotes.append(m.group(1)); i += 1
        return "quote", i, quotes

    # text block
    text = []
    while i < n:
        curr = body_lines[i]
        if not curr.strip():
            i += 1
            continue
        if curr.startswith("### ") or _CODE_FENCE_RE.match(curr) or _TABLE_ROW_RE.match(curr) or _BLOCKQUOTE_RE.match(curr):
            break
        text.append(curr); i += 1
    return "text", i, text

def _count_lines_height(lines: List[str], width_in: float, font_pt: float) -> Tuple[int, float]:
    """
    Return (visual_lines_count, height_in) for the given lines using same estimator pieces.
    """
    if not lines:
        return 0, 0.0
    line_height_in = _pt_to_in(font_pt) * LINE_SPACING
    para_space_in = _pt_to_in(PARA_SPACE_AFTER_PT)

    total_visual = 0
    nonempty = 0
    for ln in lines:
        if not ln.strip():
            continue
        lvl, txt = _parse_bullet_level(ln)
        total_visual += _lines_for_text(txt, width_in * 0.95, font_pt, lvl)
        nonempty += 1

    if nonempty == 0:
        return 0, 0.0
    content_h = total_visual * line_height_in
    spacing_h = max(0, nonempty - 1) * para_space_in
    margins_h = TF_MARGIN_IN
    return total_visual, (content_h + spacing_h + margins_h) * HEIGHT_FUDGE

def _pack_lines_to_height(lines: List[str], width_in: float, font_pt: float, max_h_in: float) -> int:
    """
    Return how many leading lines from `lines` can fit into `max_h_in`.
    Greedy: add one line at a time with conservative height calc.
    """
    if max_h_in <= 0:
        return 0
    lo, hi = 0, len(lines)
    # binary search for speed on big chunks
    while lo < hi:
        mid = (lo + hi + 1) // 2
        _, h = _count_lines_height(lines[:mid], width_in, font_pt)
        if h <= max_h_in:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _render_section_across_slides(
        prs: Presentation,
        title: str,
        body_lines: List[str],
        sources_map: Dict[int, Dict[str, str]],
        resolve_citations: bool
) -> None:
    """
    Robust renderer that splits a section across multiple slides as needed.
    Prevents overlaps by estimating block heights conservatively and
    pushing blocks that don't fit to the next slide.
    """
    # initial slide
    slide, top_anchor, avail_in, y = _new_content_slide(prs, title)

    i = 0
    n = len(body_lines)

    def need_new_slide(cont: bool = True):
        nonlocal slide, top_anchor, avail_in, y
        cont_title = f"{title} (cont.)" if cont else title
        slide, top_anchor, avail_in, y = _new_content_slide(prs, cont_title)

    while i < n:
        # skip blank lines
        if not body_lines[i].strip():
            i += 1
            continue

        # --- H3 header ---
        if body_lines[i].startswith("### "):
            header_text = body_lines[i][4:].strip()
            header_h = 0.42  # header band (incl. spacing/padding)

            # Look ahead to the *next* block and estimate its full height.
            kind, nxt_end, payload = _peek_next_block(body_lines, i+1)
            need_after_h = 0.0
            if kind == "text":
                need_after_h = _estimate_text_block_height(payload, _emu_to_in(PAGE["content_width"]), TYPE_SCALE["body"].pt)
            elif kind == "code":
                need_after_h = 0.40 + max(1, min(20, len(payload))) * 0.18
            elif kind == "table":
                tbl = _parse_table(payload)
                if tbl:
                    rows = len(tbl)
                    need_after_h = 0.30 + rows * 0.45
            elif kind == "quote":
                qh = _estimate_text_block_height(payload, _emu_to_in(PAGE["content_width"]), TYPE_SCALE["body"].pt)
                need_after_h = max(0.60, qh + 0.20)
            else:
                need_after_h = 0.30  # minimal stub if blank/unknown

            # If header + next block won't fit, start a new slide *before* placing header
            if y + header_h + need_after_h > avail_in:
                need_new_slide(cont=True)

            # place header
            tbox = _add_textbox(slide, PAGE["content_left"], top_anchor + Inches(y),
                                PAGE["content_width"], Inches(header_h - 0.10))
            tf = tbox.text_frame
            tf.word_wrap = True
            tf.auto_size = MSO_AUTO_SIZE.NONE
            p = tf.paragraphs[0]
            _style_paragraph(p, space_after=Pt(1))
            r = p.add_run()
            r.text = header_text
            _style_run(r, size=TYPE_SCALE["h3"], bold=True, color=PALETTE["fg"])
            y += header_h + BLOCK_GAP_IN
            i += 1
            continue

        # --- Code fence ---
        if _CODE_FENCE_RE.match(body_lines[i]):
            i += 1
            code = []
            while i < n and not _CODE_FENCE_RE.match(body_lines[i]):
                code.append(body_lines[i])
                i += 1
            # consume closing fence if present
            if i < n and _CODE_FENCE_RE.match(body_lines[i]):
                i += 1

            lines = code[:20]
            per_line = 0.18
            needed = 0.40 + max(1, len(lines)) * per_line
            if y + needed > avail_in:
                need_new_slide(cont=True)

            rect = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE,
                PAGE["content_left"],
                top_anchor + Inches(y),
                PAGE["content_width"],
                Inches(needed - 0.06)
            )
            rect.fill.solid()
            rect.fill.fore_color.rgb = PALETTE["code_bg"]
            rect.line.color.rgb = PALETTE["rule"]
            rect.line.width = Pt(0.5)

            tf = rect.text_frame
            tf.word_wrap = True
            tf.margin_left = Inches(0.12)
            tf.margin_right = Inches(0.12)
            tf.margin_top = Inches(0.10)
            tf.margin_bottom = Inches(0.10)

            p = tf.paragraphs[0]
            for j, code_line in enumerate(lines):
                if j > 0:
                    p = tf.add_paragraph()
                r = p.add_run()
                r.text = code_line
                _style_run(r, size=TYPE_SCALE["code"], mono=True, color=PALETTE["fg"])

            y += needed + BLOCK_GAP_IN
            continue

        # --- Table block ---
        if _TABLE_ROW_RE.match(body_lines[i]):
            table_lines = []
            while i < n and _TABLE_ROW_RE.match(body_lines[i]):
                table_lines.append(body_lines[i])
                i += 1

            table_data = _parse_table(table_lines)
            if not table_data:
                continue

            rows, cols = len(table_data), len(table_data[0])
            row_h = 0.45
            needed = 0.30 + rows * row_h
            if y + needed > avail_in:
                need_new_slide(cont=True)

            shape = slide.shapes.add_table(
                rows, cols,
                PAGE["content_left"],
                top_anchor + Inches(y),
                PAGE["content_width"],
                Inches(needed - 0.10)
            )
            tbl = shape.table

            # column widths
            col_width = int(PAGE["content_width"] / cols)
            for j in range(cols):
                tbl.columns[j].width = col_width

            # taller header row
            try:
                tbl.rows[0].height = Inches(0.55)
            except Exception:
                pass

            # fill cells + styles
            for r_idx, row_data in enumerate(table_data):
                for c_idx, cell_text in enumerate(row_data):
                    cell = tbl.cell(r_idx, c_idx)
                    cell.text = cell_text

                    # Make header background light
                    if r_idx == 0:
                        cell.fill.solid()
                        cell.fill.fore_color.rgb = PALETTE["table_header_bg"]

                    # text frame hygiene
                    tf = cell.text_frame
                    tf.word_wrap = True
                    tf.auto_size = MSO_AUTO_SIZE.NONE
                    tf.vertical_anchor = MSO_ANCHOR.MIDDLE

                    for p in tf.paragraphs:
                        p.alignment = PP_ALIGN.LEFT
                        for rr in p.runs:
                            rr.font.size = TYPE_SCALE["body"]
                            if r_idx == 0:
                                rr.font.bold = True
                                rr.font.color.rgb = PALETTE["fg"]   # dark text in header
                            else:
                                # Leave body text color/weight as-is so user/theme/zebra can control it later
                                pass

            y += needed + BLOCK_GAP_IN
            continue

        # --- Blockquote ---
        if _BLOCKQUOTE_RE.match(body_lines[i]):
            quotes = []
            while i < n and _BLOCKQUOTE_RE.match(body_lines[i]):
                m = _BLOCKQUOTE_RE.match(body_lines[i])
                quotes.append(m.group(1))
                i += 1

            # estimate quote height as normal text with italics
            q_height = _estimate_text_block_height(quotes, PAGE["content_width"].inches, TYPE_SCALE["body"].pt)
            q_height = max(0.60, q_height + 0.20)  # padding

            if y + q_height > avail_in:
                need_new_slide(cont=True)

            rect = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE,
                PAGE["content_left"],
                top_anchor + Inches(y),
                PAGE["content_width"],
                Inches(q_height - 0.08)
            )
            rect.fill.solid()
            rect.fill.fore_color.rgb = PALETTE["quote_bg"]
            rect.line.fill.background()

            bar = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE,
                PAGE["content_left"],
                top_anchor + Inches(y),
                Inches(0.08),
                Inches(q_height - 0.08)
            )
            bar.fill.solid()
            bar.fill.fore_color.rgb = PALETTE["rule"]
            bar.line.fill.background()

            tf = rect.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            r = p.add_run()
            r.text = "\n".join(quotes)
            _style_run(r, size=TYPE_SCALE["body"], italic=True, color=PALETTE["muted"])

            y += q_height + BLOCK_GAP_IN
            continue

        # --- Regular text block (collect until next special) ---
        kind, j, chunk = _peek_next_block(body_lines, i)
        if kind != "text":
            # nothing to render here
            i = max(i + 1, j)
            continue

        width_in = _emu_to_in(PAGE["content_width"])
        font_pt = TYPE_SCALE["body"].pt

        # how much space remains
        remain_in = max(0.0, avail_in - y)

        # If nothing fits on this slide, start a new one
        if remain_in < 0.35:
            need_new_slide(cont=True)
            remain_in = avail_in

        # Pack as many lines as we can into the remaining space
        fit_count = _pack_lines_to_height(chunk, width_in, font_pt, remain_in - BLOCK_GAP_IN)
        if fit_count == 0:
            need_new_slide(cont=True)
            remain_in = avail_in
            fit_count = _pack_lines_to_height(chunk, width_in, font_pt, remain_in - BLOCK_GAP_IN)

        to_render = chunk[:fit_count]
        _, est_h = _count_lines_height(to_render, width_in, font_pt)

        tbox = _add_textbox(
            slide,
            PAGE["content_left"],
            top_anchor + Inches(y),
            PAGE["content_width"],
            Inches(est_h)
        )
        tf = tbox.text_frame
        tf.word_wrap = True
        tf.auto_size = MSO_AUTO_SIZE.NONE

        p = tf.paragraphs[0]
        first = True
        for line in to_render:
            if not first:
                p = tf.add_paragraph()
            first = False
            lvl, txt = _parse_bullet_level(line)
            _style_paragraph(p, level=lvl, space_after=Pt(PARA_SPACE_AFTER_PT))
            _emit_text_with_formatting(p, txt, sources_map, resolve_citations)

        y += est_h + BLOCK_GAP_IN
        i += fit_count  # advance within the same logical block; we'll loop back for the remainder
        continue

        # # --- Regular text block (collect until next special) ---
        # chunk = []
        # j = i
        # while j < n:
        #     curr = body_lines[j]
        #     if not curr.strip():
        #         j += 1
        #         continue
        #     if curr.startswith("### ") or _CODE_FENCE_RE.match(curr) or _TABLE_ROW_RE.match(curr) or _BLOCKQUOTE_RE.match(curr):
        #         break
        #     chunk.append(curr)
        #     j += 1
        #
        # # nothing to render
        # if not chunk:
        #     i = max(i + 1, j)
        #     continue
        #
        # # estimate height conservatively for body text
        # est_h = _estimate_text_block_height(chunk, PAGE["content_width"].inches, TYPE_SCALE["body"].pt)
        # # if it doesn't fit, move to a new slide (but re-render same chunk)
        # if y + est_h > avail_in:
        #     need_new_slide(cont=True)
        #
        # # render the text chunk
        # tbox = _add_textbox(
        #     slide,
        #     PAGE["content_left"],
        #     top_anchor + Inches(y),
        #     PAGE["content_width"],
        #     Inches(est_h - 0.06)
        # )
        # tf = tbox.text_frame
        # tf.word_wrap = True
        # # DISABLE AUTOSIZE
        # tf.auto_size = MSO_AUTO_SIZE.NONE
        # p = tf.paragraphs[0]
        # first = True
        # for line in chunk:
        #     if not first:
        #         p = tf.add_paragraph()
        #     first = False
        #     lvl, txt = _parse_bullet_level(line)
        #     _style_paragraph(p, level=lvl, space_after=Pt(2))
        #     _emit_text_with_formatting(p, txt, sources_map, resolve_citations)
        #
        # y += est_h + BLOCK_GAP_IN
        # i = j  # advance to the next unread line

def _add_sources_slide(prs: Presentation, sources_map: Dict[int, Dict[str, str]], order: List[int]) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _add_slide_title(slide, "Sources")

    tbox = _add_textbox(
        slide,
        PAGE["content_left"],
        PAGE["content_top"] + Inches(0.8),
        PAGE["content_width"],
        Inches(5.0)
    )

    tf = tbox.text_frame
    tf.auto_size = MSO_AUTO_SIZE.NONE
    p = tf.paragraphs[0]

    for idx, sid in enumerate(order):
        if idx > 0:
            p = tf.add_paragraph()

        src = sources_map.get(sid, {})
        title = src.get("title", f"Source {sid}")
        url = src.get("url", "")

        _style_paragraph(p, space_after=Pt(4))

        r1 = p.add_run()
        r1.text = f"[{sid}] "
        _style_run(r1, size=TYPE_SCALE["body"], bold=True, color=PALETTE["fg"])

        r2 = p.add_run()
        r2.text = title
        _style_run(r2, size=TYPE_SCALE["body"], color=PALETTE["fg"])

        if url:
            r3 = p.add_run()
            r3.text = f" ({_domain_of(url)})"
            _style_run(r3, size=TYPE_SCALE["caption"], color=PALETTE["accent"])

def render_pptx(
        path: str,
        content_md: str,
        *,
        title: Optional[str] = None,
        base_dir: Optional[str] = None,
        sources: Optional[str] = None,
        resolve_citations: bool = False,
        include_sources_slide: bool = False
) -> str:
    """
    Render PPTX from Markdown - FINAL FIXED VERSION.

    FIXES:
    - Table parsing (checks for dashes correctly)
    - Word wrapping (proper textbox settings)
    - Width overflow (8.4" content area)
    - Height tracking (proper y_offset management)
    - H3 headers (### parsed and styled)
    - Overflow prevention (stops when slide is full)
    - All markdown features work
    """
    basename = _basename_only(path, ".pptx")
    outdir = _outdir()
    outfile = outdir / basename
    _ensure_parent(outfile)

    sources_map: Dict[int, Dict[str, str]] = {}
    order: List[int] = []
    if sources:
        sources_map, order = md_utils._normalize_sources(sources)

    sections = _split_markdown_sections(content_md or "")
    prs = Presentation()

    # Title slide
    if title:
        _add_title_slide(prs, title)
    else:
        fst_title, _ = sections[0]
        _add_title_slide(prs, fst_title)

    # Content slides
    for stitle, body in sections:
        # _add_content_slide(prs, stitle, body, sources_map, resolve_citations)
        _render_section_across_slides(prs, stitle, body, sources_map, resolve_citations)

    # Sources slide
    if include_sources_slide and sources_map:
        _add_sources_slide(prs, sources_map, order)

    prs.save(str(outfile))
    return basename