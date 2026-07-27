# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

"""Catalog serving tests: listing pages, search (hook + lexical degrade),
pagination, item-page chrome/rail shell, sitemap catalog URLs, and the
hot-index schema upgrade path."""
from __future__ import annotations

import asyncio
import copy
import json
import pathlib

from kdcube_ai_app.apps.chat.sdk.pub.model import INDEX_SCHEMA, PublicContentItem
from kdcube_ai_app.apps.chat.sdk.pub.registry import PublicContentRegistry
from kdcube_ai_app.apps.chat.sdk.pub.service import (
    resolve_alias_configs,
    serve_public_content,
)
from kdcube_ai_app.apps.chat.sdk.storage.bundle_artifact_storage import BundleArtifactStorage
from kdcube_ai_app.infra.plugin.bundle_loader import public_content, public_content_search


class _App:
    BUNDLE_ID = "news@test"

    @public_content(alias="news")
    async def news_items(self):
        return []

    @public_content_search(alias="news")
    async def news_search(self, query: str, *, prefix: str = "", limit: int = 50):
        # Deterministic, order-bearing hook: worst-to-best reversed.
        if "boom" in query:
            raise RuntimeError("engine down")
        return [
            f"{prefix}/2026-07-01-alpha",
            f"{prefix}/2026-07-03-gamma",
        ]


class _AppNoHook:
    BUNDLE_ID = "news@test"

    @public_content(alias="news")
    async def news_items(self):
        return []


_PROPS = {
    "public_content": {
        "news": {
            "enabled": True,
            "canonical_base": "https://kdcube.tech/news",
            "sitemap": True,
            "og_defaults": {"site_name": "KDCube"},
            "catalogs": {
                "kdcube/blogs": {
                    "title": "Engineering blog",
                    "nav_label": "Blogs",
                    "eyebrow": "KDCube Press",
                    "accent": "#01BEB2",
                    "page_size": 2,
                },
                "kdcube/journal": {
                    "title": "Our Journal",
                    "nav_label": "Journal",
                    "accent": "#0969DA",
                    "background": "#F4F9FF",
                },
            },
            "chrome": {
                "brand_label": "KDCube",
                "brand_href": "/",
                "links": [
                    {"label": "Home", "href": "/"},
                    {"label": "News", "href": "/news"},
                    {"label": "Blog", "href": "/blog"},
                ],
            },
        },
    }
}

_BASE = "http://localhost:8010/api/integrations/bundles/t1/p1/news@test/public/__content__"


def _registry(tmp_path) -> PublicContentRegistry:
    return PublicContentRegistry(
        alias="news",
        durable=BundleArtifactStorage(
            tenant="t1", project="p1", bundle_id="news@test",
            storage_uri=(tmp_path / "durable").as_uri(),
        ),
        hot_root=tmp_path / "hot",
    )


def _seed(tmp_path) -> None:
    items = [
        PublicContentItem(
            alias="news",
            slug=f"kdcube/blogs/2026-07-0{i}-{name}",
            title=f"{name.title()} article",
            summary=f"The {name} summary about delegated credentials." if name == "alpha" else f"The {name} summary.",
            tags=["mcp", "oauth"] if name == "alpha" else ["platform"],
            kicker="Deep",
            section="blogs",
            body_html=f"<p>{name} body</p>",
            published_at=f"2026-07-0{i}",
            lastmod=f"2026-07-0{i}",
        )
        for i, name in ((1, "alpha"), (2, "beta"), (3, "gamma"))
    ]
    items.append(PublicContentItem(
        alias="news", slug="kdcube/journal/2026-07-04-delta", title="Delta note",
        summary="Journal note.", section="journal", body_html="<p>delta</p>",
        published_at="2026-07-04", lastmod="2026-07-04",
    ))
    items.append(PublicContentItem(
        alias="news", slug="industry/ai/2026-07-05-eps", title="Industry digest",
        body_html="<p>digest</p>", published_at="2026-07-05", lastmod="2026-07-05",
    ))
    asyncio.run(_registry(tmp_path).publish_many(items))


def _serve(tmp_path, path_tail: str, *, workflow=None, query_params=None, props=None):
    import kdcube_ai_app.apps.chat.sdk.pub.service as service_mod

    original = service_mod.BundleArtifactStorage

    def _factory(**kwargs):
        kwargs["storage_uri"] = (tmp_path / "durable").as_uri()
        return original(**kwargs)

    service_mod.BundleArtifactStorage = _factory
    try:
        return asyncio.run(serve_public_content(
            workflow=workflow or _App(),
            tenant="t1", project="p1", bundle_id="news@test",
            props=props or _PROPS,
            hot_root=tmp_path / "hot",
            path_tail=path_tail,
            serving_base_url=_BASE,
            query_params=query_params or {},
        ))
    finally:
        service_mod.BundleArtifactStorage = original


def test_catalogs_mapping_form_parses():
    config = resolve_alias_configs(_PROPS)["news"]
    assert [c.prefix for c in config.catalogs] == ["kdcube/blogs", "kdcube/journal"]
    assert config.catalog("kdcube/blogs").label == "Blogs"
    assert config.chrome and config.chrome.links[2].label == "Blog"


def test_empty_prefix_declares_alias_root_catalog():
    props = copy.deepcopy(_PROPS)
    props["public_content"]["news"]["catalogs"] = {
        "": {"title": "All writing", "nav_label": "All"},
        "kdcube/blogs": {"title": "Engineering"},
    }
    config = resolve_alias_configs(props)["news"]
    assert [c.prefix for c in config.catalogs] == ["", "kdcube/blogs"]
    assert config.catalog("").covers("industry/ai/2026-07-05-eps") is True
    assert config.canonical_url("") == "https://kdcube.tech/news"


def test_alias_root_can_redirect_to_a_declared_catalog(tmp_path):
    _seed(tmp_path)
    props = copy.deepcopy(_PROPS)
    props["public_content"]["news"]["root_redirect"] = "industry/ai"
    props["public_content"]["news"]["catalogs"]["industry/ai"] = {
        "title": "AI Industry News",
    }
    response = _serve(tmp_path, "news", props=props)
    assert response.status_code == 302
    assert response.headers["Location"] == "https://kdcube.tech/news/industry/ai"


def test_alias_root_redirect_must_name_a_non_empty_catalog():
    props = copy.deepcopy(_PROPS)
    props["public_content"]["news"]["root_redirect"] = "industry/ai"
    assert "news" not in resolve_alias_configs(props)

    props["public_content"]["news"]["catalogs"][""] = {"title": "All"}
    props["public_content"]["news"]["catalogs"]["industry/ai"] = {
        "title": "AI Industry News",
    }
    assert "news" not in resolve_alias_configs(props)


def test_alias_root_catalog_lists_every_published_item(tmp_path):
    _seed(tmp_path)
    props = copy.deepcopy(_PROPS)
    props["public_content"]["news"]["catalogs"] = {
        "": {"title": "All writing", "nav_label": "All", "page_size": 10},
        "kdcube/blogs": {"title": "Engineering"},
        "kdcube/journal": {"title": "Journal"},
    }
    page = _serve(tmp_path, "news", props=props).content.decode("utf-8")
    assert "All writing" in page
    assert "Gamma article" in page
    assert "Delta note" in page
    assert "Industry digest" in page
    assert 'rel="canonical" href="https://kdcube.tech/news"' in page

    payload = json.loads(_serve(tmp_path, "", props=props).content)
    catalog_sitemaps = payload["sitemaps"][0]["catalog_sitemaps"]
    assert all(row["prefix"] for row in catalog_sitemaps)


def test_catalog_page_lists_newest_first_with_pagination(tmp_path):
    _seed(tmp_path)
    resp = _serve(tmp_path, "news/kdcube/blogs")
    assert resp.status_code == 200 and resp.media_type.startswith("text/html")
    page = resp.content.decode("utf-8")
    # newest two of three (page_size=2), fold pills with counts, chrome
    assert "Gamma article" in page and "Beta article" in page and "Alpha article" not in page
    assert "Blogs · 3" in page and "Journal · 1" in page
    assert "kdcpub-header" in page and "Engineering blog" in page
    assert 'aria-controls="kdcpub-site-nav"' in page
    assert 'aria-label="Primary navigation"' in page
    assert "1–2 of 3" in page
    assert 'href="/news" class="kdcpub-active" aria-current="page"' in page
    assert 'href="/" class="kdcpub-active"' not in page
    # second page
    page2 = _serve(tmp_path, "news/kdcube/blogs", query_params={"offset": "2"}).content.decode("utf-8")
    assert "Alpha article" in page2 and "Gamma article" not in page2


def test_catalog_summary_renders_safe_inline_html(tmp_path):
    """A summary carrying light HTML (an article lede) renders inline emphasis
    on catalog cards: <strong> survives, the block <p> wrapper is unwrapped,
    text is escaped, and no literal tags leak. See sanitize_inline_html."""
    item = PublicContentItem(
        alias="news",
        slug="kdcube/blogs/2026-07-09-html-summary",
        title="HTML summary article",
        summary="<p>Moonshot's <strong>Kimi K3</strong> publishes weights &amp; a license.</p>",
        section="blogs",
        body_html="<p>body</p>",
        published_at="2026-07-09",
        lastmod="2026-07-09",
    )
    asyncio.run(_registry(tmp_path).publish_many([item]))
    page = _serve(tmp_path, "news/kdcube/blogs").content.decode("utf-8")
    # emphasis renders inside the card; the block <p> wrapper was unwrapped
    assert "<strong>Kimi K3</strong> publishes weights &amp; a license." in page
    # no literal <p>/<strong> markup from the summary leaked as escaped text
    assert "&lt;p&gt;" not in page
    assert "&lt;strong&gt;" not in page


def test_catalog_summary_neutralizes_unsafe_html(tmp_path):
    """Public-facing render: disallowed tags/attributes are stripped. A script
    and an onclick never survive to the page; their text is escaped."""
    item = PublicContentItem(
        alias="news",
        slug="kdcube/blogs/2026-07-10-xss",
        title="XSS attempt",
        summary='Safe <strong onclick="steal()">bold</strong> then <script>alert(1)</script> end.',
        section="blogs",
        body_html="<p>body</p>",
        published_at="2026-07-10",
        lastmod="2026-07-10",
    )
    asyncio.run(_registry(tmp_path).publish_many([item]))
    page = _serve(tmp_path, "news/kdcube/blogs").content.decode("utf-8")
    assert "<strong>bold</strong>" in page       # tag kept, onclick dropped
    assert "onclick" not in page
    # the injected script was unwrapped: no live <script> from the summary,
    # its text is inert (the page's own nav <script> is unrelated chrome)
    assert "<script>alert(1)" not in page
    assert "alert(1)" in page                     # inner text survives, escaped/inert


def test_catalog_search_uses_declared_hook_order(tmp_path):
    _seed(tmp_path)
    page = _serve(
        tmp_path, "news/kdcube/blogs", query_params={"q": "credentials"}
    ).content.decode("utf-8")
    # hook returns alpha then gamma; beta is not a result
    assert page.index("Alpha article") < page.index("Gamma article")
    assert "Beta article" not in page
    assert "results" in page and "noindex" in page
    # tier-honest hint: the app's engine answered
    assert "full article text" in page


def test_catalog_search_degrades_to_lexical_without_hook(tmp_path):
    _seed(tmp_path)
    page = _serve(
        tmp_path, "news/kdcube/blogs", workflow=_AppNoHook(),
        query_params={"q": "delegated credentials"},
    ).content.decode("utf-8")
    assert "Alpha article" in page and "Beta article" not in page
    # tier-honest hint: the basic index-card match answered
    assert "titles, summaries and tags" in page


def test_catalog_search_degrades_to_lexical_when_hook_raises(tmp_path):
    _seed(tmp_path)
    page = _serve(
        tmp_path, "news/kdcube/blogs", query_params={"q": "boom delegated"}
    ).content.decode("utf-8")
    # hook raised; lexical still matches alpha via summary
    assert "Alpha article" in page


def test_item_under_catalog_gets_chrome_and_rail(tmp_path):
    _seed(tmp_path)
    page = _serve(tmp_path, "news/kdcube/blogs/2026-07-02-beta").content.decode("utf-8")
    assert "kdcpub-rail-card" in page and "kdcpub-header" in page
    assert "kdcpub-now" in page  # active rail item
    assert 'rel="canonical" href="https://kdcube.tech/news/kdcube/blogs/2026-07-02-beta"' in page
    assert "<p>beta body</p>" in page
    # crawlable metadata intact alongside the shell
    assert "application/ld+json" in page


def test_item_outside_catalogs_stays_plain(tmp_path):
    _seed(tmp_path)
    page = _serve(tmp_path, "news/industry/ai/2026-07-05-eps").content.decode("utf-8")
    assert "kdcpub-rail-card" not in page and "kdcpub-header" not in page


def test_sitemap_includes_catalog_urls(tmp_path):
    _seed(tmp_path)
    xml = _serve(tmp_path, "news/sitemap.xml").content.decode("utf-8")
    assert "<loc>https://kdcube.tech/news/kdcube/blogs</loc>" in xml
    assert "<loc>https://kdcube.tech/news/kdcube/journal</loc>" in xml
    assert "<loc>https://kdcube.tech/news/kdcube/blogs/2026-07-01-alpha</loc>" in xml


def test_catalog_sitemap_contains_only_its_catalog(tmp_path):
    _seed(tmp_path)
    xml = _serve(tmp_path, "news/kdcube/journal/sitemap.xml").content.decode("utf-8")
    assert "<loc>https://kdcube.tech/news/kdcube/journal</loc>" in xml
    assert "<loc>https://kdcube.tech/news/kdcube/journal/2026-07-04-delta</loc>" in xml
    assert "kdcube/blogs" not in xml
    assert "industry/ai" not in xml


def test_catalog_sitemap_is_discoverable_from_descriptor_list(tmp_path):
    _seed(tmp_path)
    payload = json.loads(_serve(tmp_path, "").content)
    catalogs = payload["sitemaps"][0]["catalog_sitemaps"]
    journal = next(item for item in catalogs if item["prefix"] == "kdcube/journal")
    assert journal == {
        "prefix": "kdcube/journal",
        "label": "Journal",
        "sitemap_url": f"{_BASE}/news/kdcube/journal/sitemap.xml",
        "item_count": 1,
        "lastmod": "2026-07-04",
    }


def test_unknown_catalog_sitemap_is_404(tmp_path):
    _seed(tmp_path)
    assert _serve(tmp_path, "news/unknown/sitemap.xml").status_code == 404


def test_presentation_theme_tokens_reach_the_page(tmp_path):
    """Alias presentation sets base tokens; the catalog's accent shorthand and
    its own presentation.theme override them (later wins), and accent tints
    derive from the winning accent."""
    _seed(tmp_path)
    props = copy.deepcopy(_PROPS)
    props["public_content"]["news"]["presentation"] = {
        "theme": {"display": "'Playfair Display',serif", "width": "1000px", "accent": "#333333"},
    }
    props["public_content"]["news"]["catalogs"]["kdcube/blogs"]["presentation"] = {
        "theme": {"ink": "#101010"},
    }
    page = _serve(tmp_path, "news/kdcube/blogs", props=props).content.decode("utf-8")
    assert "--kdcpub-display:'Playfair Display',serif;" in page
    assert "--kdcpub-width:1000px;" in page
    assert "--kdcpub-ink:#101010;" in page
    # catalog accent shorthand (#01BEB2) wins over the alias theme accent
    assert "--kdcpub-accent:#01BEB2;" in page
    assert "--kdcpub-accent-rgb:1,190,178;" in page


def test_presentation_stylesheets_load_after_sdk_styles(tmp_path):
    """App stylesheets (alias first, then catalog) come after the SDK style +
    theme token blocks so they can override anything via kdcpub-* classes."""
    _seed(tmp_path)
    props = copy.deepcopy(_PROPS)
    props["public_content"]["news"]["presentation"] = {
        "stylesheets": ["/assets/pub-brand.css"],
    }
    props["public_content"]["news"]["catalogs"]["kdcube/blogs"]["presentation"] = {
        "stylesheets": ["/assets/pub-blogs.css"],
    }
    page = _serve(tmp_path, "news/kdcube/blogs", props=props).content.decode("utf-8")
    alias_link = '<link rel="stylesheet" href="/assets/pub-brand.css">'
    fold_link = '<link rel="stylesheet" href="/assets/pub-blogs.css">'
    assert page.rindex("--kdcpub-accent") < page.index(alias_link) < page.index(fold_link)
    # the item-page shell carries the same app styles (shared primitives)
    item = _serve(
        tmp_path, "news/kdcube/blogs/2026-07-02-beta", props=props
    ).content.decode("utf-8")
    assert alias_link in item and fold_link in item


def test_presentation_rejects_unsafe_theme_tokens():
    props = copy.deepcopy(_PROPS)
    props["public_content"]["news"]["presentation"] = {
        "theme": {"ink": "#111}</style><script>x()</script>"},
    }
    assert "news" not in resolve_alias_configs(props)


def test_empty_fold_renders_deliberate_empty_state(tmp_path):
    _seed(tmp_path)
    props = copy.deepcopy(_PROPS)
    props["public_content"]["news"]["catalogs"]["recipes"] = {"title": "Recipes"}
    page = _serve(tmp_path, "news/recipes", props=props).content.decode("utf-8")
    assert "kdcpub-empty" in page and "No articles yet." in page
    # no 0–0 pager row in an empty band (the class only remains in the CSS)
    assert '<div class="kdcpub-pager">' not in page


def test_old_schema_hot_index_is_rebuilt_on_catalog_serve(tmp_path):
    _seed(tmp_path)
    hot_dir = tmp_path / "hot" / "_public_content" / "news"
    # Simulate a hot tier written by the previous release: schema-1 index
    # (entries without card fields) and the old gen-only signature format.
    index_path = hot_dir / "index.json"
    index_path.write_text(
        '{"alias":"news","generation":1,"updated_at":"2026-07-01T00:00:00+00:00",'
        '"entries":[{"slug":"kdcube/blogs/2026-07-03-gamma","title":"Gamma article",'
        '"lastmod":"2026-07-03","published_at":"2026-07-03","state":"published"}]}',
        encoding="utf-8",
    )
    (hot_dir / ".index.signature").write_text("gen:1\n", encoding="utf-8")
    page = _serve(tmp_path, "news/kdcube/blogs").content.decode("utf-8")
    # rebuilt from durable: all three blog items back, with summaries (schema 2)
    assert "Gamma article" in page and "Beta article" in page
    assert "The gamma summary." in page
    rebuilt = (hot_dir / "index.json").read_text(encoding="utf-8")
    assert f'"index_schema":{INDEX_SCHEMA}' in rebuilt
    assert pathlib.Path(hot_dir / ".index.signature").read_text().startswith("gen:")
