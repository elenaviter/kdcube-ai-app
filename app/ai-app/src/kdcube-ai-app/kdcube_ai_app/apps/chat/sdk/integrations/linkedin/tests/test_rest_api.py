# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""LinkedIn versioned REST payload/response contract."""

from __future__ import annotations

import pytest

from kdcube_ai_app.apps.chat.sdk.integrations.linkedin import rest_api


class _Response:
    def __init__(self, headers: dict[str, str] | None = None) -> None:
        self.headers = headers or {}


def test_person_urn_accepts_subject_or_full_urn():
    assert rest_api.person_urn("dE5aOhH-ap") == "urn:li:person:dE5aOhH-ap"
    assert rest_api.person_urn("urn:li:person:abc") == "urn:li:person:abc"


def test_person_urn_rejects_empty_subject():
    with pytest.raises(rest_api.LinkedInPayloadError):
        rest_api.person_urn("")


def test_rest_headers_carry_version_and_protocol():
    headers = rest_api.rest_headers(access_token="T", api_version="202601", json_body=True)
    assert headers["LinkedIn-Version"] == "202601"
    assert headers["X-Restli-Protocol-Version"] == "2.0.0"
    assert headers["Authorization"] == "Bearer T"
    assert headers["Content-Type"] == "application/json"


def test_rest_headers_require_a_version():
    with pytest.raises(rest_api.LinkedInPayloadError):
        rest_api.rest_headers(access_token="T", api_version="")


def test_text_post_body_is_the_flat_posts_shape():
    body = rest_api.build_post_body(author_urn="urn:li:person:X", commentary="hello")
    assert body == {
        "author": "urn:li:person:X",
        "commentary": "hello",
        "visibility": "PUBLIC",
        "distribution": {
            "feedDistribution": "MAIN_FEED",
            "targetEntities": [],
            "thirdPartyDistributionChannels": [],
        },
        "lifecycleState": "PUBLISHED",
        "isReshareDisabledByAuthor": False,
    }
    assert "content" not in body


def test_one_image_uses_content_media():
    content = rest_api.build_post_content([{"image_urn": "urn:li:image:A", "alt_text": "chart"}])
    assert content == {"media": {"id": "urn:li:image:A", "altText": "chart"}}


def test_several_images_use_content_multi_image():
    content = rest_api.build_post_content(
        [{"image_urn": "urn:li:image:A"}, {"image_urn": "urn:li:image:B"}]
    )
    assert content == {
        "multiImage": {"images": [{"id": "urn:li:image:A"}, {"id": "urn:li:image:B"}]}
    }


def test_no_images_produces_no_content_block():
    assert rest_api.build_post_content([]) is None


def test_image_count_over_the_limit_is_rejected():
    entries = [{"image_urn": f"urn:li:image:{i}"} for i in range(rest_api.MULTI_IMAGE_MAX + 1)]
    with pytest.raises(rest_api.LinkedInPayloadError):
        rest_api.build_post_content(entries)


def test_commentary_limits_are_enforced():
    with pytest.raises(rest_api.LinkedInPayloadError):
        rest_api.build_post_body(author_urn="u", commentary="   ")
    with pytest.raises(rest_api.LinkedInPayloadError):
        rest_api.build_post_body(
            author_urn="u", commentary="x" * (rest_api.LINKEDIN_POST_MAX_CHARS + 1)
        )


def test_created_urn_comes_from_the_response_header():
    assert rest_api.created_urn_from_response(
        _Response({"x-restli-id": "urn:li:share:7123"})
    ) == "urn:li:share:7123"
    assert rest_api.created_urn_from_response(_Response({})) == ""


def test_image_upload_init_is_parsed():
    parsed = rest_api.parse_image_upload_init(
        {"value": {"uploadUrl": "https://up", "image": "urn:li:image:A", "uploadUrlExpiresAt": 5}}
    )
    assert parsed == {"upload_url": "https://up", "image_urn": "urn:li:image:A", "expires_at": 5}


def test_comments_url_uses_the_unversioned_endpoint():
    url = rest_api.social_actions_comments_url("urn:li:share:7123")
    assert url == (
        "https://api.linkedin.com/v2/socialActions/urn%3Ali%3Ashare%3A7123/comments"
    )
    with pytest.raises(rest_api.LinkedInPayloadError):
        rest_api.social_actions_comments_url("")


def test_legacy_headers_carry_no_api_version():
    headers = rest_api.legacy_headers(access_token="T", json_body=True)
    assert "LinkedIn-Version" not in headers
    assert headers["X-Restli-Protocol-Version"] == "2.0.0"
    assert headers["Authorization"] == "Bearer T"
    assert headers["Content-Type"] == "application/json"


def test_comment_body_shape():
    assert rest_api.build_comment_body(
        actor_urn="urn:li:person:X", object_urn="urn:li:share:7", text="hi"
    ) == {
        "actor": "urn:li:person:X",
        "object": "urn:li:share:7",
        "message": {"text": "hi"},
    }


def test_permalink_is_derived_from_the_post_urn():
    assert rest_api.post_permalink("urn:li:share:7123") == (
        "https://www.linkedin.com/feed/update/urn:li:share:7123"
    )
    assert rest_api.post_permalink("") == ""


def test_comment_urn_prefers_what_linkedin_returned():
    assert rest_api.comment_urn_from_body({"commentUrn": "urn:li:comment:(a,1)"}) == (
        "urn:li:comment:(a,1)"
    )
    assert rest_api.comment_urn_from_body({"$URN": "urn:li:comment:(b,2)"}) == (
        "urn:li:comment:(b,2)"
    )


def test_comment_urn_composite_uses_the_response_thread():
    # The response `object` can differ from the request target; keying on the
    # target would produce an invalid comment URN.
    assert rest_api.comment_urn_from_body(
        {"object": "urn:li:activity:99"}, comment_id="7"
    ) == "urn:li:comment:(urn:li:activity:99,7)"


def test_comment_urn_is_empty_without_a_thread():
    assert rest_api.comment_urn_from_body({}, comment_id="7") == ""
    assert rest_api.comment_urn_from_body({"object": "urn:li:activity:99"}) == ""
    assert rest_api.comment_urn_from_body(None) == ""


# --- Article card (content.article) + UTF-16 length -------------------------


def test_utf16_length_counts_emoji_as_two():
    from kdcube_ai_app.apps.chat.sdk.integrations.linkedin.rest_api import utf16_length

    assert utf16_length("abc") == 3
    assert utf16_length("🚀") == 2          # astral plane
    assert utf16_length("a🚀b") == 4
    assert utf16_length("👩‍💻") == 5        # person(2) + ZWJ(1) + laptop(2)


def test_build_article_content_full_card():
    from kdcube_ai_app.apps.chat.sdk.integrations.linkedin import rest_api

    article = rest_api.build_article_content(
        source="https://example.test/blog/post",
        title="A title",
        description="A description",
        thumbnail_urn="urn:li:image:abc",
        thumbnail_alt="Cover",
    )
    assert article == {
        "source": "https://example.test/blog/post",
        "title": "A title",
        "description": "A description",
        "thumbnail": "urn:li:image:abc",
        "thumbnailAltText": "Cover",
    }


def test_build_article_content_requires_source_and_title():
    from kdcube_ai_app.apps.chat.sdk.integrations.linkedin import rest_api

    with pytest.raises(rest_api.LinkedInPayloadError):
        rest_api.build_article_content(source="", title="t")
    with pytest.raises(rest_api.LinkedInPayloadError):
        rest_api.build_article_content(source="https://x.test", title="")
    with pytest.raises(rest_api.LinkedInPayloadError):
        rest_api.build_article_content(source="https://x.test", title="x" * 401)


def test_post_body_carries_article_and_never_both():
    from kdcube_ai_app.apps.chat.sdk.integrations.linkedin import rest_api

    article = rest_api.build_article_content(source="https://x.test", title="t")
    body = rest_api.build_post_body(
        author_urn="urn:li:person:p", commentary="hello", article=article
    )
    assert body["content"] == {"article": article}
    with pytest.raises(rest_api.LinkedInPayloadError):
        rest_api.build_post_body(
            author_urn="urn:li:person:p",
            commentary="hello",
            article=article,
            images=[{"image_urn": "urn:li:image:i"}],
        )


def test_post_body_limit_is_utf16_aware():
    from kdcube_ai_app.apps.chat.sdk.integrations.linkedin import rest_api

    # 1499 chars + 751 emoji = 1499 + 1502 = 3001 UTF-16 units -> rejected,
    # although len() sees only 2250 code points.
    text = "x" * 1499 + "🚀" * 751
    assert len(text) < 3000
    with pytest.raises(rest_api.LinkedInPayloadError):
        rest_api.build_post_body(author_urn="urn:li:person:p", commentary=text)


# --- Post delete / partial update + comment delete ---------------------------


class _CapturingClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def post(self, url, *, json=None, headers=None, params=None):
        self.calls.append({"method": "POST", "url": url, "json": json,
                           "headers": headers, "params": params})
        return _Response({})

    async def delete(self, url, *, headers=None, params=None):
        self.calls.append({"method": "DELETE", "url": url, "headers": headers,
                           "params": params})
        return _Response({})


def test_post_entity_url_encodes_the_urn():
    assert rest_api.post_entity_url("urn:li:share:7123") == (
        "https://api.linkedin.com/rest/posts/urn%3Ali%3Ashare%3A7123"
    )
    with pytest.raises(rest_api.LinkedInPayloadError):
        rest_api.post_entity_url("")


def test_commentary_patch_sets_commentary_and_nothing_else():
    assert rest_api.build_commentary_patch("new text") == {
        "patch": {"$set": {"commentary": "new text"}}
    }


def test_commentary_patch_enforces_the_utf16_limit():
    with pytest.raises(rest_api.LinkedInPayloadError):
        rest_api.build_commentary_patch("   ")
    # 3001 UTF-16 units although len() sees only 2250 code points.
    text = "x" * 1499 + "🚀" * 751
    assert len(text) < 3000
    with pytest.raises(rest_api.LinkedInPayloadError):
        rest_api.build_commentary_patch(text)


@pytest.mark.asyncio
async def test_delete_post_sends_the_restli_delete_method():
    client = _CapturingClient()
    await rest_api.delete_post(
        client, access_token="T", api_version="202601", post_urn="urn:li:share:7123"
    )
    call = client.calls[0]
    assert call["method"] == "DELETE"
    assert call["url"] == "https://api.linkedin.com/rest/posts/urn%3Ali%3Ashare%3A7123"
    assert call["headers"]["X-RestLi-Method"] == "DELETE"
    assert call["headers"]["LinkedIn-Version"] == "202601"


@pytest.mark.asyncio
async def test_update_post_commentary_is_a_partial_update_post():
    client = _CapturingClient()
    await rest_api.update_post_commentary(
        client,
        access_token="T",
        api_version="202601",
        post_urn="urn:li:ugcPost:9",
        commentary="edited",
    )
    call = client.calls[0]
    assert call["method"] == "POST"
    assert call["url"] == "https://api.linkedin.com/rest/posts/urn%3Ali%3AugcPost%3A9"
    assert call["headers"]["X-RestLi-Method"] == "PARTIAL_UPDATE"
    assert call["headers"]["Content-Type"] == "application/json"
    assert call["json"] == {"patch": {"$set": {"commentary": "edited"}}}


def test_comment_delete_url_encodes_thread_and_comment():
    url = rest_api.social_actions_comment_url("urn:li:share:7123", "654 3")
    assert url == (
        "https://api.linkedin.com/v2/socialActions/urn%3Ali%3Ashare%3A7123/comments/654%203"
    )
    with pytest.raises(rest_api.LinkedInPayloadError):
        rest_api.social_actions_comment_url("urn:li:share:7123", "")
    with pytest.raises(rest_api.LinkedInPayloadError):
        rest_api.social_actions_comment_url("", "6543")


@pytest.mark.asyncio
async def test_delete_comment_targets_v2_with_the_actor():
    client = _CapturingClient()
    await rest_api.delete_comment(
        client,
        access_token="T",
        actor_urn="urn:li:person:X",
        object_urn="urn:li:share:7123",
        comment_id="6543",
    )
    call = client.calls[0]
    assert call["method"] == "DELETE"
    assert call["url"] == (
        "https://api.linkedin.com/v2/socialActions/urn%3Ali%3Ashare%3A7123/comments/6543"
    )
    assert call["params"] == {"actor": "urn:li:person:X"}
    # Unversioned endpoint: legacy headers carry no LinkedIn-Version.
    assert "LinkedIn-Version" not in call["headers"]
    assert call["headers"]["Authorization"] == "Bearer T"


@pytest.mark.asyncio
async def test_delete_comment_requires_the_actor():
    with pytest.raises(rest_api.LinkedInPayloadError):
        await rest_api.delete_comment(
            _CapturingClient(),
            access_token="T",
            actor_urn="",
            object_urn="urn:li:share:7123",
            comment_id="6543",
        )


def test_commentary_escapes_little_text_control_characters():
    """Live finding 2026-08-13: a raw "(" in an API-written commentary made
    LinkedIn truncate the rendered post at that character. Commentary is
    little-text format; reserved characters must be backslash-escaped, while
    bare hashtags stay raw."""
    from kdcube_ai_app.apps.chat.sdk.integrations.linkedin.delivery import (
        escape_little_text,
        format_post_text,
    )

    out = format_post_text("works. (edit leg too.)\n\n#PipelineTest")
    assert "\\(edit leg too.\\)" in out
    assert "#PipelineTest" in out and "\\#" not in out
    assert escape_little_text("a|b{c}@d[e](f)<g>*h_i~\\") == (
        "a\\|b\\{c\\}\\@d\\[e\\]\\(f\\)\\<g\\>\\*h\\_i\\~\\\\"
    )
