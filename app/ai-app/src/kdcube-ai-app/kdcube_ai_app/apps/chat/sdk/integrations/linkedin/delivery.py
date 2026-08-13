from __future__ import annotations

import re

LINKEDIN_POST_MAX_CHARS = 3000


def utf16_length(text: str) -> int:
    """Length as LinkedIn counts it: UTF-16 code units (emoji cost 2)."""
    return sum(2 if ord(ch) > 0xFFFF else 1 for ch in str(text or ""))


def strip_markdown(text: str) -> str:
    """Convert markdown-flavoured text to plain text suitable for a LinkedIn post.

    LinkedIn text posts do not render HTML or markdown; raw syntax characters
    would appear verbatim.  This function removes the most common patterns that
    Claude agents produce so the post reads naturally.
    """
    value = str(text or "")

    # Setext headings (underline style) — drop the underline row
    value = re.sub(r"\n[=\-]{2,}\n", "\n", value)

    # ATX headings  # Heading → Heading
    value = re.sub(r"^#{1,6}\s+(.+)$", r"\1", value, flags=re.MULTILINE)

    # Bold / italic  **text** / *text* / __text__ / _text_
    value = re.sub(r"\*{1,3}([^*\n]+)\*{1,3}", r"\1", value)
    value = re.sub(r"_{1,3}([^_\n]+)_{1,3}", r"\1", value)

    # Inline code  `code`
    value = re.sub(r"`([^`\n]+)`", r"\1", value)

    # Fenced code blocks  ``` … ``` or ~~~ … ~~~
    value = re.sub(r"```[^\n]*\n(.*?)\n```", r"\1", value, flags=re.DOTALL)
    value = re.sub(r"~~~[^\n]*\n(.*?)\n~~~", r"\1", value, flags=re.DOTALL)

    # Markdown images  ![alt](url) → alt
    value = re.sub(r"!\[([^\]\n]*)\]\([^)]*\)", r"\1", value)

    # Markdown links  [label](url) → label (url)
    value = re.sub(r"\[([^\]\n]+)\]\(([^)\s]+)\)", r"\1 (\2)", value)

    # Blockquotes  > text → text
    value = re.sub(r"^>\s*", "", value, flags=re.MULTILINE)

    # Horizontal rules
    value = re.sub(r"^\s*[-*_]{3,}\s*$", "", value, flags=re.MULTILINE)

    # Unordered list bullets  - item / * item / + item → item
    value = re.sub(r"^[\-*+]\s+", "", value, flags=re.MULTILINE)

    # Ordered list  1. item → item
    value = re.sub(r"^\d+\.\s+", "", value, flags=re.MULTILINE)

    # Strikethrough  ~~text~~ → text
    value = re.sub(r"~~([^~\n]+)~~", r"\1", value)

    # Collapse runs of blank lines to a single blank line
    value = re.sub(r"\n{3,}", "\n\n", value)

    return value.strip()


# LinkedIn commentary is "little text" format: these characters are control
# characters there (mention/hashtag templates, emphasis), and RAW occurrences
# corrupt the parse - observed live: an unescaped "(" truncated the rendered
# post at that character. "#" stays raw: bare hashtags are the one little-text
# construct authors write literally.
_LITTLE_TEXT_RESERVED = "\\|{}@[]()<>*_~"


def escape_little_text(text: str) -> str:
    """Backslash-escape little-text control characters for the commentary."""
    value = str(text or "")
    out = []
    for ch in value:
        if ch in _LITTLE_TEXT_RESERVED:
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def truncate_post_text(text: str, max_chars: int = LINKEDIN_POST_MAX_CHARS, suffix: str = "…") -> str:
    """Truncate *text* to *max_chars* UTF-16 code units, appending *suffix*.

    LinkedIn measures the limit in UTF-16 code units, so an emoji costs 2 —
    a code-point count undercounts exactly the posts that carry emoji. The cut
    is made on a word boundary where possible so the post does not end
    mid-word.
    """
    value = str(text or "")
    if utf16_length(value) <= max_chars:
        return value
    budget = max_chars - utf16_length(suffix)
    # Walk to the last code-point index whose prefix fits the UTF-16 budget.
    units = 0
    fit = 0
    for index, ch in enumerate(value):
        units += 2 if ord(ch) > 0xFFFF else 1
        if units > budget:
            break
        fit = index + 1
    # Try to break on a word boundary (space or newline)
    cut = value.rfind(" ", 0, fit + 1)
    if cut <= 0:
        cut = value.rfind("\n", 0, fit + 1)
    if cut <= 0:
        cut = fit
    return value[:cut].rstrip() + suffix


def format_post_text(text: str, max_chars: int = LINKEDIN_POST_MAX_CHARS) -> str:
    """Strip markdown and enforce the LinkedIn character limit.

    Convenience wrapper: call this on any text before passing it to
    ``create_linkedin_post``.
    """
    return truncate_post_text(escape_little_text(strip_markdown(text)), max_chars=max_chars)
