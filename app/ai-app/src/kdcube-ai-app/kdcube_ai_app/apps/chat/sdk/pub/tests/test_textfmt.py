# SPDX-License-Identifier: MIT
"""Security boundary for the public-content summary sanitizer.

sanitize_inline_html is an allowlist: only inline emphasis tags survive, always
without attributes, all text escaped, block/link tags unwrapped, unclosed tags
auto-closed. These tests assert known XSS vectors are neutralized."""
import pytest

from kdcube_ai_app.apps.chat.sdk.pub.textfmt import plain_text, sanitize_inline_html as S


def test_allowed_emphasis_survives_without_attributes():
    assert S("<strong>Kimi</strong>") == "<strong>Kimi</strong>"
    assert S('<em class="x" style="color:red">a</em>') == "<em>a</em>"
    assert S("<code>x=1</code> and <mark>y</mark>") == "<code>x=1</code> and <mark>y</mark>"


def test_block_wrapper_is_unwrapped():
    assert S("<p>Moonshot <strong>K3</strong></p>") == "Moonshot <strong>K3</strong>"
    assert S("<div><h2>t</h2><ul><li>a</li></ul></div>") == "t a"


@pytest.mark.parametrize("payload", [
    '<script>alert(1)</script>',
    '<img src=x onerror=alert(1)>',
    '<a href="javascript:alert(1)">x</a>',
    '<iframe src="//evil"></iframe>',
    '<svg/onload=alert(1)>',
    '<strong onclick="steal()">b</strong>',
    '<style>body{display:none}</style>',
    '<object data="x"></object>',
])
def test_no_executable_or_link_surface_survives(payload):
    out = S(payload)
    lower = out.lower()
    # no live tags of any disallowed kind, no event handlers, no js: urls
    for bad in ("<script", "<img", "<iframe", "<svg", "<style", "<object", "<a ", "<a>",
                "onerror", "onclick", "onload", "javascript:", "href=", "src=", "data="):
        assert bad not in lower, (payload, out, bad)


def test_disallowed_tag_text_is_kept_but_escaped():
    # <script> is unwrapped; its text content stays as ESCAPED text, inert
    out = S("a <script>alert(1)</script> b")
    assert "<script" not in out
    assert "alert(1)" in out
    assert "&lt;" not in out or "alert" in out  # inner text present, not a live tag


def test_pre_escaped_source_stays_visible_not_reinterpreted():
    # literal "&lt;p&gt;" in the source must render as visible text "<p>"
    out = S("&lt;p&gt;hi&lt;/p&gt;")
    assert out == "&lt;p&gt;hi&lt;/p&gt;"


def test_unclosed_tag_is_auto_closed():
    assert S("<strong>bold and more") == "<strong>bold and more</strong>"
    # stray/misnested closers do not bleed
    assert S("<em><strong>x</em>y") == "<em><strong>x</strong></em>y"
    assert S("</strong>just text") == "just text"


def test_angle_bracket_text_is_escaped():
    assert S("5 < 7 and 8 > 3") == "5 &lt; 7 and 8 &gt; 3"


def test_plain_text_strips_all_markup():
    assert plain_text("<p>Moonshot <strong>K3</strong> &amp; more</p>") == "Moonshot K3 & more"
    assert plain_text("") == ""
