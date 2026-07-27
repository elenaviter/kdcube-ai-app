# SPDX-License-Identifier: MIT
"""Text helpers for public-content rendering — one source of truth for the two
ways a summary is used.

A ``summary`` is authored as prose and may carry light HTML (an article lede
like ``<p>Moonshot's <strong>Kimi K3</strong> ships today.</p>``). It surfaces
in two very different roles, and each needs its own treatment:

- ``plain_text`` — a machine DESCRIPTION (``<meta name="description">``,
  ``og:description``, ``twitter:description``, JSON-LD ``description``). These
  must be plain text: no markup, entities resolved, whitespace collapsed.

- ``sanitize_inline_html`` — VISIBLE prose on a card or the generated summary
  paragraph. Inline emphasis renders; everything else is neutralized. This is
  public-facing output, so the sanitizer is an allowlist, not a blocklist:
  only a fixed set of inline formatting tags survive, always WITHOUT
  attributes (which removes the ``on*`` / ``style`` / ``href`` injection
  surface entirely), and all text is escaped. Block elements (``<p>``,
  ``<div>``, headings, lists) and links are UNWRAPPED — their tags dropped,
  their text kept — because a clamped card has no room for block flow and a
  rail card already wraps its summary in an ``<a>`` (a nested link would be
  invalid and an href is an injection vector). Unclosed tags are auto-closed
  so a card can never bleed formatting into the rest of the page.
"""

import html
import re
from html.parser import HTMLParser

_TAG_RE = re.compile(r"<[^>]+>")

# Inline emphasis only. No links, no images, no block/structural elements.
_INLINE_ALLOWED = frozenset({
    "strong", "b", "em", "i", "u", "s", "code", "mark", "sup", "sub", "abbr", "small",
})


def plain_text(value: str) -> str:
    """Flatten possibly-HTML text to a single clean line (no markup)."""
    text = _TAG_RE.sub(" ", str(value or ""))
    text = html.unescape(text)
    return " ".join(text.split()).strip()


class _InlineSanitizer(HTMLParser):
    def __init__(self) -> None:
        # convert_charrefs=True: entities in text become characters, which we
        # then re-escape — so a literal "&lt;p&gt;" in the source stays visible
        # as text and is never reinterpreted as a tag.
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.stack: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in _INLINE_ALLOWED:
            self.out.append(f"<{tag}>")  # attributes intentionally dropped
            self.stack.append(tag)
        else:
            # Unwrap: drop the tag but keep a space so a block boundary
            # (</p><p>) does not fuse adjacent words. Collapsed away later.
            self.out.append(" ")

    def handle_startendtag(self, tag: str, attrs) -> None:
        # A self-closing inline emphasis tag carries no content; any other
        # self-closing tag (e.g. <br/>, <img/>) becomes a word boundary.
        if tag not in _INLINE_ALLOWED:
            self.out.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _INLINE_ALLOWED and tag in self.stack:
            # Close down to (and including) the matching open tag; tolerates
            # simple misnesting by closing the inner tags first.
            while self.stack:
                top = self.stack.pop()
                self.out.append(f"</{top}>")
                if top == tag:
                    break
        elif tag not in _INLINE_ALLOWED:
            self.out.append(" ")  # closing block boundary -> word separator

    def handle_data(self, data: str) -> None:
        self.out.append(html.escape(data, quote=False))


def sanitize_inline_html(value: str) -> str:
    """Return a safe inline-HTML rendering of ``value`` (see module docstring).

    Empty/whitespace-only input returns "". Whitespace runs are collapsed to
    single spaces while inline tags are preserved.
    """
    parser = _InlineSanitizer()
    parser.feed(str(value or ""))
    parser.close()
    # Auto-close anything left open so the fragment is well-formed.
    while parser.stack:
        parser.out.append(f"</{parser.stack.pop()}>")
    return " ".join("".join(parser.out).split()).strip()
