from __future__ import annotations

import json
from typing import Any

import pytest

from kdcube_ai_app.apps.chat.sdk.integrations.docs.named_service import (
    ACTION_CREATE_COMMENT,
    ACTION_DELETE_COMMENT,
    ACTION_EXPORT,
    ACTION_APPEND_TEXT,
    DOCS_COMMENT_CLAIM,
    DOCS_CONNECTED_ACCOUNT_REQUIREMENTS,
    DOCS_DOCUMENT_KIND,
    DOCS_GRANT_HINTS,
    DOCS_READ_CLAIM,
    DOCS_SNAPSHOT_MEDIA_TYPE,
    DOCS_SNAPSHOT_SCHEMA,
    DOCS_WRITE_CLAIM,
    DocsNamedServiceProvider,
    document_ref,
    docs_named_service_spec,
    parse_docs_ref,
)
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers import (
    NamedServiceContext,
    NamedServiceRequest,
    NamedServiceStreamResult,
)
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.types import (
    BLOCK_PRODUCE,
    EVENT_RESOLVE,
    OBJECT_ACTION,
    OBJECT_DELETE,
    OBJECT_GET,
    OBJECT_SCHEMA,
    OBJECT_SEARCH,
    OBJECT_UPSERT,
    PROVIDER_ABOUT,
)


def _ctx() -> NamedServiceContext:
    return NamedServiceContext(tenant="demo", project="project", user_id="user-1")


def test_spec_publishes_discoverable_ref_and_object_metadata() -> None:
    metadata = docs_named_service_spec().metadata

    assert metadata["canonical_refs"]["document"].startswith("docs:<provider>")
    assert metadata["canonical_refs"]["document"].endswith(":document:<document_id>")
    assert metadata["object_kinds"][DOCS_DOCUMENT_KIND]
    assert metadata["actions"][ACTION_APPEND_TEXT]


class _FakeDocs:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error: dict[str, Any] | None = None

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        if self.error is not None:
            return dict(self.error)
        operation = kwargs["operation"]
        account_id = kwargs.get("account_id") or "account-1"
        if operation == "search":
            return {
                "ok": True,
                "ret": {
                    "account_id": account_id,
                    "items": [
                        {
                            "document_id": "doc-1",
                            "title": "Launch plan",
                            "web_url": "https://docs.google.com/document/d/doc-1/edit",
                        }
                    ],
                    "next_cursor": "next-1",
                },
            }
        if operation == "get":
            return {
                "ok": True,
                "ret": {
                    "account_id": account_id,
                    "document_id": "doc-1",
                    "title": "Launch plan",
                    "revision_id": "rev-9",
                    "web_url": "https://docs.google.com/document/d/doc-1/edit",
                    "text": "Intro paragraph.\nSecond paragraph body.",
                    "tabs": [{"title": "Main"}],
                    "end_index": 41,
                },
            }
        if operation == "list_comments":
            return {
                "ok": True,
                "ret": {
                    "account_id": account_id,
                    "document_id": "doc-1",
                    "comments": [
                        {"comment_id": "c-1", "content": "Please expand.", "resolved": False}
                    ],
                    "count": 1,
                    "next_cursor": "",
                },
            }
        if operation == "create":
            return {
                "ok": True,
                "ret": {
                    "account_id": account_id,
                    "document_id": "created-1",
                    "title": kwargs["payload"]["title"],
                    "web_url": "https://docs.google.com/document/d/created-1/edit",
                    "revision_id": "rev-1",
                },
            }
        if operation == "export":
            return {
                "ok": True,
                "ret": {
                    "account_id": account_id,
                    "document_id": "doc-1",
                    "format": kwargs["payload"].get("format"),
                    "content_base64": "AAA=",
                },
            }
        if operation == "create_comment":
            return {
                "ok": True,
                "ret": {
                    "account_id": account_id,
                    "document_id": "doc-1",
                    "comment": {"comment_id": "c-9", "content": "New note."},
                },
            }
        if operation == "delete_comment":
            return {
                "ok": True,
                "ret": {
                    "document_id": "doc-1",
                    "deleted_comment_id": kwargs["payload"].get("comment_id"),
                },
            }
        return {
            "ok": True,
            "ret": {
                "account_id": account_id,
                "document_id": "doc-1",
                "operation": operation,
            },
        }


def _provider(
    fake: _FakeDocs,
    *,
    file_url_factory: Any = None,
) -> DocsNamedServiceProvider:
    return DocsNamedServiceProvider(
        execute_operation=fake.execute,
        bundle_id="kdcube-services@1-0",
        file_url_factory=file_url_factory,
    )


def test_docs_refs_are_provider_neutral_and_stable() -> None:
    ref = document_ref("account-1", "doc-1")
    assert ref == "docs:google:account-1:document:doc-1"
    assert parse_docs_ref(ref) == {
        "ref": ref,
        "provider": "google",
        "account_id": "account-1",
        "document_id": "doc-1",
        "object_kind": DOCS_DOCUMENT_KIND,
    }


@pytest.mark.parametrize(
    "bad_ref",
    [
        "",
        "docs:google:account-1:document",
        "docs:google:account-1:spreadsheet:doc-1",
        "docs:google::document:doc-1",
        "mail:google:account-1:document:doc-1",
        "docs:google:account-1:document:doc-1:tab:7",
    ],
)
def test_parse_docs_ref_rejects_malformed_input(bad_ref: str) -> None:
    with pytest.raises(ValueError):
        parse_docs_ref(bad_ref)


def test_write_grant_and_connected_account_claims_are_separate_boundaries() -> None:
    assert DOCS_GRANT_HINTS["object.upsert"] == [DOCS_WRITE_CLAIM]
    assert DOCS_GRANT_HINTS["object.action.export"] == [DOCS_READ_CLAIM]
    assert DOCS_GRANT_HINTS["object.action.create_comment"] == [DOCS_COMMENT_CLAIM]
    claims_by_operation = DOCS_CONNECTED_ACCOUNT_REQUIREMENTS[0][
        "claims_by_operation"
    ]
    assert claims_by_operation["object.upsert"] == [
        DOCS_READ_CLAIM,
        DOCS_WRITE_CLAIM,
    ]
    assert claims_by_operation["object.action.create_comment"] == [
        DOCS_READ_CLAIM,
        DOCS_COMMENT_CLAIM,
    ]
    assert claims_by_operation["object.action.export"] == [DOCS_READ_CLAIM]


@pytest.mark.anyio
async def test_about_and_schema_return_expected_shape() -> None:
    fake = _FakeDocs()
    provider = _provider(fake)

    about = await provider.dispatch(
        _ctx(),
        NamedServiceRequest(operation=PROVIDER_ABOUT, namespace="docs"),
    )
    assert about.ok is True
    assert about.extra["schema"]["namespace"] == "docs"
    assert "google" in about.extra["providers"]

    schema = await provider.dispatch(
        _ctx(),
        NamedServiceRequest(operation=OBJECT_SCHEMA, namespace="docs"),
    )
    assert schema.ok is True
    assert schema.extra["schema"]["refs"]["document"].startswith("docs:<provider>")
    assert DOCS_DOCUMENT_KIND in schema.extra["schema"]["object_kinds"]
    assert fake.calls == []


@pytest.mark.anyio
async def test_search_returns_named_service_objects() -> None:
    fake = _FakeDocs()
    response = await _provider(fake).dispatch(
        _ctx(),
        NamedServiceRequest(
            operation=OBJECT_SEARCH,
            namespace="docs",
            query="launch",
            limit=5,
            filters={"account_id": "account-1"},
        ),
    )
    assert response.ok is True
    assert response.next_cursor == "next-1"
    assert response.items[0]["ref"] == "docs:google:account-1:document:doc-1"
    assert response.items[0]["object_kind"] == DOCS_DOCUMENT_KIND
    assert fake.calls == [
        {
            "operation": "search",
            "claim": DOCS_READ_CLAIM,
            "tool_name": "named_services.docs.object.search",
            "payload": {"query": "launch", "limit": 5, "cursor": ""},
            "account_id": "account-1",
        }
    ]


@pytest.mark.anyio
async def test_get_reads_document_metadata_and_text() -> None:
    fake = _FakeDocs()
    ref = "docs:google:account-1:document:doc-1"
    response = await _provider(fake).dispatch(
        _ctx(),
        NamedServiceRequest(operation=OBJECT_GET, namespace="docs", object_ref=ref),
    )
    assert response.ok is True
    assert response.object["ref"] == ref
    assert response.object["object_kind"] == DOCS_DOCUMENT_KIND
    assert response.object["text"].startswith("Intro paragraph")
    assert fake.calls[0]["operation"] == "get"
    assert fake.calls[0]["claim"] == DOCS_READ_CLAIM
    assert fake.calls[0]["payload"]["document_ref"] == "doc-1"


@pytest.mark.anyio
async def test_turnless_metadata_get_offers_complete_snapshot_download() -> None:
    fake = _FakeDocs()
    minted: list[dict[str, Any]] = []

    async def _file_url(ctx: NamedServiceContext, info: Any) -> dict[str, str]:
        minted.append({"ctx": ctx, "info": dict(info or {})})
        return {
            "url": "https://runtime.test/signed-doc",
            "expires_at": "2026-07-27T12:00:00Z",
        }

    ref = "docs:google:account-1:document:doc-1"
    response = await _provider(fake, file_url_factory=_file_url).dispatch(
        _ctx(),
        NamedServiceRequest(operation=OBJECT_GET, namespace="docs", object_ref=ref),
    )

    assert response.ok is True
    assert next(iter(response.object)) == "snapshot"
    assert response.object["snapshot"] == {
        "schema": DOCS_SNAPSHOT_SCHEMA,
        "media_type": DOCS_SNAPSHOT_MEDIA_TYPE,
        "filename": "doc-1.docs.json",
        "download": {
            "encoding": "url",
            "url": "https://runtime.test/signed-doc",
            "expires_at": "2026-07-27T12:00:00Z",
        },
    }
    assert minted[0]["info"] == {"ref": ref}


@pytest.mark.anyio
async def test_turn_scoped_get_does_not_mint_out_of_band_snapshot_url() -> None:
    fake = _FakeDocs()
    calls = 0

    async def _file_url(ctx: NamedServiceContext, info: Any) -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"url": "https://runtime.test/unused"}

    response = await _provider(fake, file_url_factory=_file_url).dispatch(
        NamedServiceContext(
            tenant="demo",
            project="project",
            user_id="user-1",
            turn_id="turn-1",
        ),
        NamedServiceRequest(
            operation=OBJECT_GET,
            namespace="docs",
            object_ref="docs:google:account-1:document:doc-1",
        ),
    )

    assert response.ok is True
    assert "snapshot" not in response.object
    assert calls == 0


@pytest.mark.anyio
async def test_get_stream_materializes_complete_doc_snapshot_for_react_pull() -> None:
    fake = _FakeDocs()
    ref = "docs:google:account-1:document:doc-1"

    result = await _provider(fake).dispatch(
        _ctx(),
        NamedServiceRequest(
            operation=OBJECT_GET,
            namespace="docs",
            object_ref=ref,
            response_mode="stream",
            context={"source": "react.pull", "materialize": True},
        ),
    )

    assert isinstance(result, NamedServiceStreamResult)
    assert result.response.ok is True
    assert result.response.object_ref == ref
    assert result.media_type == DOCS_SNAPSHOT_MEDIA_TYPE
    assert result.filename == "doc-1.docs.json"
    snapshot = json.loads(b"".join([chunk async for chunk in result.chunks]))
    assert snapshot["schema"] == DOCS_SNAPSHOT_SCHEMA
    assert snapshot["object_ref"] == ref
    assert snapshot["object"]["text"].startswith("Intro paragraph")
    assert snapshot["comments"][0]["comment_id"] == "c-1"
    assert snapshot["materialization"]["complete_text"] is True
    assert [call["operation"] for call in fake.calls] == ["get", "list_comments"]
    assert fake.calls[0]["claim"] == DOCS_READ_CLAIM
    assert fake.calls[1]["claim"] == DOCS_READ_CLAIM


@pytest.mark.anyio
async def test_event_resolve_maps_doc_refs_without_calling_google() -> None:
    fake = _FakeDocs()
    ref = "docs:google:account-1:document:doc-1"

    response = await _provider(fake).dispatch(
        _ctx(),
        NamedServiceRequest(operation=EVENT_RESOLVE, namespace="docs", object_ref=ref),
    )

    assert response.ok is True
    assert response.object_ref == ref
    assert response.extra["event_source_id"] == "named_services.docs"
    assert response.extra["target_surface"] == "sdk.docs.snapshot"
    assert fake.calls == []


@pytest.mark.anyio
async def test_block_produce_projects_document_inventory_and_preview() -> None:
    fake = _FakeDocs()
    ref = "docs:google:account-1:document:doc-1"
    snapshot = {
        "schema": DOCS_SNAPSHOT_SCHEMA,
        "object_ref": ref,
        "object_kind": DOCS_DOCUMENT_KIND,
        "object": {
            "ref": ref,
            "object_kind": DOCS_DOCUMENT_KIND,
            "document_id": "doc-1",
            "title": "Launch plan",
            "web_url": "https://docs.google.com/document/d/doc-1/edit",
            "revision_id": "rev-9",
            "text": "First heading line.\nBody content here.",
            "tabs": [{"title": "Main"}],
            "end_index": 41,
        },
        "comments": [{"comment_id": "c-1", "content": "note"}],
        "materialization": {"text_materialized": True, "complete_text": True},
    }
    logical_path = "conv:fi:conv_1.turn_1.named_services/docs/doc-1.docs.json"
    physical_path = "turn_1/attachments/named_services/docs/doc-1.docs.json"

    response = await _provider(fake).dispatch(
        _ctx(),
        NamedServiceRequest(
            operation=BLOCK_PRODUCE,
            namespace="docs",
            object_ref=ref,
            payload={
                "target": {
                    "turn_id": "turn_1",
                    "tool_call_id": "tc_1",
                    "tool_id": "react.read",
                    "logical_path": logical_path,
                    "raw": {
                        "text": json.dumps(snapshot),
                        "mime": DOCS_SNAPSHOT_MEDIA_TYPE,
                    },
                    "meta": {"physical_path": physical_path},
                }
            },
        ),
    )

    assert response.ok is True
    assert len(response.extra["blocks"]) == 1
    block = response.extra["blocks"][0]
    assert block["path"] == ref
    assert "[DOCS SNAPSHOT]" in block["text"]
    assert "Launch plan" in block["text"]
    assert "First heading line." in block["text"]
    assert "comment_count: 1" in block["text"]
    assert logical_path in block["text"]
    assert physical_path in block["text"]
    assert fake.calls == []


@pytest.mark.anyio
async def test_upsert_create_and_edit_route_to_google_operations() -> None:
    fake = _FakeDocs()
    provider = _provider(fake)

    created = await provider.dispatch(
        _ctx(),
        NamedServiceRequest(
            operation=OBJECT_UPSERT,
            namespace="docs",
            object={"title": "Created from named services"},
        ),
    )
    assert created.ok is True
    assert created.object_ref == "docs:google:account-1:document:created-1"
    assert fake.calls[0]["operation"] == "create"
    assert fake.calls[0]["claim"] == (DOCS_READ_CLAIM, DOCS_WRITE_CLAIM)

    appended = await provider.dispatch(
        _ctx(),
        NamedServiceRequest(
            operation=OBJECT_UPSERT,
            namespace="docs",
            object_ref="docs:google:account-1:document:doc-1",
            object={"text": "More body."},
        ),
    )
    assert appended.ok is True
    assert fake.calls[-1]["operation"] == "append_text"
    assert fake.calls[-1]["payload"]["text"] == "More body."

    replaced = await provider.dispatch(
        _ctx(),
        NamedServiceRequest(
            operation=OBJECT_UPSERT,
            namespace="docs",
            object_ref="docs:google:account-1:document:doc-1",
            object={"replacements": [{"find": "a", "replace": "b"}]},
        ),
    )
    assert replaced.ok is True
    assert fake.calls[-1]["operation"] == "replace_text"


@pytest.mark.anyio
async def test_actions_gate_on_the_right_claim_per_verb() -> None:
    fake = _FakeDocs()
    provider = _provider(fake)
    ref = "docs:google:account-1:document:doc-1"

    exported = await provider.dispatch(
        _ctx(),
        NamedServiceRequest(
            operation=OBJECT_ACTION,
            namespace="docs",
            object_ref=ref,
            action=ACTION_EXPORT,
            payload={"format": "pdf"},
        ),
    )
    assert exported.ok is True
    assert fake.calls[-1]["operation"] == ACTION_EXPORT
    assert fake.calls[-1]["claim"] == DOCS_READ_CLAIM
    assert fake.calls[-1]["payload"]["document_ref"] == "doc-1"

    commented = await provider.dispatch(
        _ctx(),
        NamedServiceRequest(
            operation=OBJECT_ACTION,
            namespace="docs",
            object_ref=ref,
            action=ACTION_CREATE_COMMENT,
            payload={"content": "New note."},
        ),
    )
    assert commented.ok is True
    assert fake.calls[-1]["operation"] == ACTION_CREATE_COMMENT
    assert fake.calls[-1]["claim"] == (DOCS_READ_CLAIM, DOCS_COMMENT_CLAIM)


@pytest.mark.anyio
async def test_unknown_action_is_rejected_before_google_call() -> None:
    fake = _FakeDocs()
    response = await _provider(fake).dispatch(
        _ctx(),
        NamedServiceRequest(
            operation=OBJECT_ACTION,
            namespace="docs",
            object_ref="docs:google:account-1:document:doc-1",
            action="drop_database",
            payload={},
        ),
    )
    assert response.ok is False
    assert response.error.code == "docs_action_not_supported"
    assert fake.calls == []


@pytest.mark.anyio
async def test_delete_removes_a_comment_and_refuses_document_deletion() -> None:
    fake = _FakeDocs()
    provider = _provider(fake)
    ref = "docs:google:account-1:document:doc-1"

    deleted = await provider.dispatch(
        _ctx(),
        NamedServiceRequest(
            operation=OBJECT_DELETE,
            namespace="docs",
            object_ref=ref,
            payload={"comment_id": "c-1"},
        ),
    )
    assert deleted.ok is True
    assert fake.calls[-1]["operation"] == ACTION_DELETE_COMMENT
    assert fake.calls[-1]["claim"] == (DOCS_READ_CLAIM, DOCS_COMMENT_CLAIM)
    assert fake.calls[-1]["payload"]["comment_id"] == "c-1"

    refused = await provider.dispatch(
        _ctx(),
        NamedServiceRequest(operation=OBJECT_DELETE, namespace="docs", object_ref=ref),
    )
    assert refused.ok is False
    assert refused.error.code == "docs_document_delete_not_supported"


@pytest.mark.anyio
async def test_connected_account_consent_error_is_preserved() -> None:
    fake = _FakeDocs()
    fake.error = {
        "ok": False,
        "error": {
            "code": "needs_connected_account_consent",
            "message": "Approve document access.",
        },
        "consent": {
            "reason": "claim_upgrade_required",
            "provider_id": "google",
            "claims": [DOCS_READ_CLAIM],
            "url": "https://runtime.test/connections",
            "retry_hint": True,
        },
    }
    response = await _provider(fake).dispatch(
        _ctx(),
        NamedServiceRequest(operation=OBJECT_SEARCH, namespace="docs", query="plan"),
    )
    assert response.ok is False
    assert response.status == 403
    assert response.error.code == "needs_connected_account_consent"
    assert response.error.details["reason"] == "claim_upgrade_required"
    assert response.error.details["claims"] == [DOCS_READ_CLAIM]
    assert response.error.details["connection_hub_url"] == (
        "https://runtime.test/connections"
    )


@pytest.mark.anyio
async def test_canonical_fields_and_ref_boundaries_cannot_be_overridden() -> None:
    fake = _FakeDocs()

    async def _malicious_search(**kwargs: Any) -> dict[str, Any]:
        fake.calls.append(dict(kwargs))
        return {
            "ok": True,
            "ret": {
                "account_id": "account-1",
                "items": [
                    {
                        "document_id": "doc-1",
                        "ref": "mail:message:not-a-doc",
                        "object_kind": "wrong.kind",
                        "account_id": "wrong-account",
                        "provider": "excel",
                    }
                ],
            },
        }

    provider = DocsNamedServiceProvider(execute_operation=_malicious_search)
    response = await provider.dispatch(
        _ctx(),
        NamedServiceRequest(
            operation=OBJECT_SEARCH,
            namespace="docs",
            query="",
            filters={"account_id": "account-1"},
        ),
    )
    assert response.ok is True
    assert response.items[0]["ref"] == "docs:google:account-1:document:doc-1"
    assert response.items[0]["object_kind"] == DOCS_DOCUMENT_KIND
    assert response.items[0]["account_id"] == "account-1"
    assert response.items[0]["provider"] == "google"


@pytest.mark.anyio
async def test_unknown_provider_fails_before_google_call() -> None:
    fake = _FakeDocs()
    response = await _provider(fake).dispatch(
        _ctx(),
        NamedServiceRequest(
            operation=OBJECT_ACTION,
            namespace="docs",
            object_ref="docs:excel:account-1:document:doc-1",
            action=ACTION_EXPORT,
            payload={"format": "pdf"},
        ),
    )
    assert response.ok is False
    assert response.error.code == "docs_provider_not_implemented"
    assert fake.calls == []
