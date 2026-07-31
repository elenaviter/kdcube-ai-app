# SPDX-License-Identifier: MIT
"""Google Docs proxy: bounded operations over mocked Docs + Drive REST.

The proxy owns no credentials and speaks raw REST; these tests mock the httpx
transport so every operation is exercised without a live Google endpoint, and
assert the serializable envelope contract (ok/error/ret, normalized failures,
token never leaked)."""
from __future__ import annotations

import asyncio
import base64
import json
from typing import Any, Callable

import httpx
import pytest

from kdcube_ai_app.apps.chat.sdk.integrations.google import docs_proxy


def _run(operation: str, payload: dict[str, Any], handler: Callable[[httpx.Request], httpx.Response],
         *, token: str = "tok-123"):
    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    orig = docs_proxy.httpx.AsyncClient
    docs_proxy.httpx.AsyncClient = _factory  # type: ignore[assignment]
    try:
        return asyncio.run(
            docs_proxy.execute_google_docs_operation(
                operation=operation, access_token=token, payload=payload
            )
        )
    finally:
        docs_proxy.httpx.AsyncClient = orig  # type: ignore[assignment]


def _json_response(request: httpx.Request, body: dict[str, Any], status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=body, request=request)


def test_missing_token_returns_credential_error():
    out = _run("get", {"document_ref": "abc"}, lambda r: _json_response(r, {}), token="")
    assert out["ok"] is False
    assert out["error"]["code"] == "credential_missing_access_token"


def test_unsupported_operation():
    out = _run("frobnicate", {}, lambda r: _json_response(r, {}))
    assert out["ok"] is False and out["error"]["code"] == "unsupported_operation"


def test_get_extracts_text_and_ids_and_url_forms():
    doc = {
        "documentId": "DOC1",
        "title": "My Doc",
        "revisionId": "rev-9",
        "body": {"content": [
            {"endIndex": 1, "sectionBreak": {}},
            {"endIndex": 20, "paragraph": {"elements": [
                {"textRun": {"content": "Hello "}},
                {"textRun": {"content": "world\n"}},
            ]}},
        ]},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/documents/DOC1"
        assert request.url.params["includeTabsContent"] == "true"
        assert request.headers["Authorization"] == "Bearer tok-123"
        return _json_response(request, doc)

    # accepts a full Docs URL and normalizes to the id
    out = _run("get", {"document_ref": "https://docs.google.com/document/d/DOC1/edit"}, handler)
    assert out["ok"] is True
    assert out["ret"]["document_id"] == "DOC1"
    assert out["ret"]["text"] == "Hello world\n"
    assert out["ret"]["end_index"] == 19


def test_get_includes_nested_tab_and_table_cell_text_for_document_edits():
    def _paragraph(text: str) -> dict[str, Any]:
        return {"paragraph": {"elements": [{"textRun": {"content": text}}]}}

    document = {
        "documentId": "INVOICE",
        "title": "26_006",
        "tabs": [
            {
                "tabProperties": {"tabId": "tab-main", "title": "Invoice"},
                "documentTab": {
                    "body": {
                        "content": [
                            {
                                "startIndex": 1,
                                "endIndex": 40,
                                "table": {
                                    "tableRows": [
                                        {
                                            "tableCells": [
                                                {"content": [_paragraph("276/006\n")]},
                                                {"content": [_paragraph("5000.00\n")]},
                                            ]
                                        },
                                        {
                                            "tableCells": [
                                                {"content": [_paragraph("01.07.2026\n")]},
                                                {"content": [_paragraph("Total\n")]},
                                            ]
                                        },
                                    ]
                                },
                            }
                        ]
                    }
                },
                "childTabs": [
                    {
                        "tabProperties": {
                            "tabId": "tab-notes",
                            "title": "Notes",
                        },
                        "documentTab": {
                            "body": {"content": [_paragraph("Internal note\n")]}
                        },
                    }
                ],
            }
        ],
    }

    out = _run(
        "get",
        {"document_ref": "INVOICE", "include_text": True},
        lambda request: _json_response(request, document),
    )

    assert out["ok"] is True
    text = out["ret"]["text"]
    assert "[tab: Invoice]" in text
    assert "276/006" in text
    assert "5000.00" in text
    assert "01.07.2026" in text
    assert "[tab: Notes]" in text
    assert "Internal note" in text
    assert out["ret"]["tabs"] == [
        {"tab_id": "tab-main", "title": "Invoice", "parent_tab_id": ""},
        {
            "tab_id": "tab-notes",
            "title": "Notes",
            "parent_tab_id": "tab-main",
        },
    ]
    assert out["ret"]["end_index"] == 39


def test_search_returns_exact_title_before_prefix_matches_and_supports_shared_drives():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        query = request.url.params["q"]
        assert request.url.params["spaces"] == "drive"
        assert request.url.params["corpora"] == "user"
        assert request.url.params["includeItemsFromAllDrives"] == "true"
        assert request.url.params["supportsAllDrives"] == "true"
        if "name = '26_006'" in query:
            return _json_response(
                request,
                {
                    "files": [
                        {
                            "id": "exact",
                            "name": "26_006",
                            "mimeType": docs_proxy.DOCS_MIME_TYPE,
                            "parents": ["shared-folder"],
                            "capabilities": {"canCopy": True},
                        }
                    ]
                },
            )
        assert "name contains '26_006'" in query
        return _json_response(
            request,
            {
                "files": [
                    {
                        "id": "prefix",
                        "name": "26_006 Archive",
                        "mimeType": docs_proxy.DOCS_MIME_TYPE,
                        "capabilities": {"canCopy": False},
                    },
                    # The exact query and prefix query may return the same row.
                    {"id": "exact", "name": "26_006"},
                ],
                "nextPageToken": "next-page",
                "incompleteSearch": True,
            },
        )

    out = _run("search", {"query": "26_006", "limit": 5}, handler)

    assert out["ok"] is True
    assert len(requests) == 2
    assert [item["document_id"] for item in out["ret"]["items"]] == [
        "exact",
        "prefix",
    ]
    assert out["ret"]["items"][0]["exact_title_match"] is True
    assert out["ret"]["items"][0]["copyable"] is True
    assert out["ret"]["items"][0]["parent_ids"] == ["shared-folder"]
    assert out["ret"]["exact_match_count"] == 1
    assert out["ret"]["next_cursor"] == "next-page"
    assert out["ret"]["incomplete_search"] is True
    assert out["ret"]["match_mode"] == "exact_then_title_prefix"


def test_search_cursor_continues_prefix_query_without_repeating_exact_lookup():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert "name contains '26_006'" in request.url.params["q"]
        assert "name = '26_006'" not in request.url.params["q"]
        assert request.url.params["pageToken"] == "page-2"
        return _json_response(request, {"files": []})

    out = _run(
        "search",
        {"query": "26_006", "limit": 5, "cursor": "page-2"},
        handler,
    )

    assert out["ok"] is True
    assert len(requests) == 1
    assert out["ret"]["match_mode"] == "title_prefix_page"


def test_copy_preserves_provider_document_and_returns_the_new_identity():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/drive/v3/files/SOURCE/copy"
        assert request.url.params["supportsAllDrives"] == "true"
        assert json.loads(request.content) == {
            "name": "26_007",
            "parents": ["invoices-folder"],
        }
        return _json_response(
            request,
            {
                "id": "CLONE",
                "name": "26_007",
                "mimeType": docs_proxy.DOCS_MIME_TYPE,
                "parents": ["invoices-folder"],
                "webViewLink": "https://docs.google.com/document/d/CLONE/edit",
            },
        )

    out = _run(
        "copy",
        {
            "document_ref": "SOURCE",
            "title": "26_007",
            "parent_id": "invoices-folder",
            "idempotency_key": "invoice-26-007",
        },
        handler,
    )

    assert out["ok"] is True
    assert out["ret"] == {
        "source_document_id": "SOURCE",
        "document_id": "CLONE",
        "title": "26_007",
        "web_url": "https://docs.google.com/document/d/CLONE/edit",
        "mime_type": docs_proxy.DOCS_MIME_TYPE,
        "parent_ids": ["invoices-folder"],
        "copied": True,
        "idempotency_key": "invoice-26-007",
    }


def test_copy_5xx_marks_outcome_unknown_to_prevent_blind_retry():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(
            request,
            {"error": {"message": "upstream timeout"}},
            status=503,
        )

    out = _run(
        "copy",
        {"document_ref": "SOURCE", "title": "26_007"},
        handler,
    )

    assert out["ok"] is False
    assert out["ret"]["outcome_unknown"] is True
    assert out["ret"]["retryable"] is True


def test_append_text_inserts_at_body_end():
    doc = {"documentId": "DOC1", "body": {"content": [{"endIndex": 42, "paragraph": {}}]}}
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return _json_response(request, doc)
        captured["body"] = json.loads(request.content)
        return _json_response(request, {"replies": []})

    out = _run("append_text", {"document_ref": "DOC1", "text": "more"}, handler)
    assert out["ok"] is True and out["ret"]["appended_chars"] == 4
    loc = captured["body"]["requests"][0]["insertText"]["location"]["index"]
    assert loc == 41  # endIndex - 1


def test_replace_text_counts_occurrences():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(request, {"replies": [
            {"replaceAllText": {"occurrencesChanged": 3}},
        ]})

    out = _run("replace_text", {"document_ref": "DOC1", "replacements": [
        {"find": "foo", "replace": "bar"}]}, handler)
    assert out["ok"] is True and out["ret"]["occurrences_changed"] == 3


def test_embed_image_requires_public_uri():
    out = _run("embed_image", {"document_ref": "DOC1", "image_uri": "file:///tmp/x.png"},
               lambda r: _json_response(r, {}))
    assert out["ok"] is False and out["error"]["code"] == "invalid_image_uri"


def test_export_returns_base64_and_bounds_size():
    payload_bytes = b"%PDF-1.4 fake"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/drive/v3/files/DOC1/export"
        assert request.url.params["mimeType"] == "application/pdf"
        return httpx.Response(200, content=payload_bytes, request=request)

    out = _run("export", {"document_ref": "DOC1", "format": "pdf"}, handler)
    assert out["ok"] is True
    assert base64.b64decode(out["ret"]["content_base64"]) == payload_bytes
    assert out["ret"]["extension"] == "pdf"


def test_import_uploads_multipart_and_converts():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/upload/drive/v3/files"
        assert request.url.params["uploadType"] == "multipart"
        assert b"vnd.google-apps.document" in request.content  # convert target
        return _json_response(request, {"id": "NEW", "name": "T", "webViewLink": "http://x"})

    out = _run("import", {"title": "T", "content": "# hi", "source_format": "markdown"}, handler)
    assert out["ok"] is True and out["ret"]["document_id"] == "NEW"


def test_list_comments_filters_resolved_by_default():
    body = {"comments": [
        {"id": "c1", "content": "open one", "resolved": False, "author": {"displayName": "A"}},
        {"id": "c2", "content": "done", "resolved": True, "author": {"displayName": "B"}},
    ]}
    out = _run("list_comments", {"document_ref": "DOC1"}, lambda r: _json_response(r, body))
    assert out["ok"] is True
    ids = [c["comment_id"] for c in out["ret"]["comments"]]
    assert ids == ["c1"]  # resolved hidden unless include_resolved


def test_create_comment_roundtrips_content():
    def handler(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content)
        return _json_response(request, {"id": "c9", "content": sent["content"],
                                        "author": {"displayName": "Me"}})

    out = _run("create_comment", {"document_ref": "DOC1", "content": "please fix"}, handler)
    assert out["ok"] is True and out["ret"]["comment"]["content"] == "please fix"


def test_resolve_comment_posts_resolve_action():
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _json_response(request, {"id": "r1", "action": "resolve"})

    out = _run("resolve_comment", {"document_ref": "DOC1", "comment_id": "c1"}, handler)
    assert out["ok"] is True
    assert captured["body"]["action"] == "resolve"


def test_provider_401_normalizes_to_authorization_failed_and_redacts():
    def handler(request: httpx.Request) -> httpx.Response:
        return _json_response(request, {"error": {"status": "UNAUTHENTICATED",
                                                  "message": "bad Bearer tok-123 token"}}, status=401)

    out = _run("get", {"document_ref": "DOC1"}, handler)
    assert out["ok"] is False
    assert out["error"]["code"] == "google_docs_authorization_failed"
    assert out["ret"]["category"] == "authorization_failed"
    assert "tok-123" not in json.dumps(out)  # token never leaks


def test_mutating_5xx_marks_outcome_unknown():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return _json_response(request, {"body": {"content": [{"endIndex": 2}]}})
        return _json_response(request, {"error": {"message": "boom"}}, status=503)

    out = _run("append_text", {"document_ref": "DOC1", "text": "x"}, handler)
    assert out["ok"] is False
    assert out["ret"]["outcome_unknown"] is True
    assert out["ret"]["retryable"] is True


def test_invalid_document_ref():
    out = _run("get", {"document_ref": "!!not an id!!"}, lambda r: _json_response(r, {}))
    assert out["ok"] is False and out["error"]["code"] == "invalid_document_ref"
