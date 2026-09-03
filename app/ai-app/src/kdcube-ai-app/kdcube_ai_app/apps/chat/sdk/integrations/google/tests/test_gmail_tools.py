from __future__ import annotations

import base64

import httpx

from kdcube_ai_app.apps.chat.sdk.integrations.google import gmail_tools
from kdcube_ai_app.apps.chat.sdk.integrations.connected_accounts import ConnectedAccountCredential
from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace import artifact_outdir_for, resolve_artifact_path
from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.references import (
    ARTIFACT_NAMESPACE_FILES,
    build_physical_artifact_path,
    physical_path_to_logical_path,
)


def _gmail_b64(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")


def test_extract_message_content_reads_body_and_attachment_metadata():
    message = {
        "id": "m1",
        "threadId": "t1",
        "snippet": "receipt",
        "payload": {
            "headers": [
                {"name": "Subject", "value": "Receipt"},
                {"name": "From", "value": "billing@example.com"},
                {"name": "To", "value": "user@example.com"},
                {"name": "Date", "value": "Mon, 6 Jul 2026 10:00:00 +0000"},
            ],
            "parts": [
                {
                    "mimeType": "multipart/alternative",
                    "parts": [
                        {"mimeType": "text/plain", "body": {"data": _gmail_b64("plain body")}},
                        {"mimeType": "text/html", "body": {"data": _gmail_b64("<p>html body</p>")}},
                    ],
                },
                {
                    "partId": "2",
                    "mimeType": "application/pdf",
                    "filename": "invoice.pdf",
                    "body": {"attachmentId": "att1", "size": 123},
                    "headers": [{"name": "Content-Disposition", "value": "attachment; filename=invoice.pdf"}],
                },
            ],
        },
    }

    out = gmail_tools._extract_message_content(message)

    assert out["body_text"] == "plain body"
    assert out["body_html"] == "<p>html body</p>"
    assert out["headers"]["subject"] == "Receipt"
    assert out["attachments"] == [
        {
            "attachment_id": "att1",
            "part_id": "2",
            "filename": "invoice.pdf",
            "mime_type": "application/pdf",
            "size_bytes": 123,
            "inline": False,
            "content_id": "",
        }
    ]


def test_load_local_attachments_resolves_kdcube_logical_paths(monkeypatch, tmp_path):
    artifact_root = artifact_outdir_for(tmp_path / "runtime")
    physical = build_physical_artifact_path(
        turn_id="turn_test",
        namespace=ARTIFACT_NAMESPACE_FILES,
        relpath="reports/report.txt",
    )
    target = resolve_artifact_path(artifact_root, physical, prefer_existing=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("hello", encoding="utf-8")
    logical = physical_path_to_logical_path(physical)

    monkeypatch.setattr(gmail_tools, "_current_artifact_context", lambda: (artifact_root, "turn_test"))

    attachments, errors = gmail_tools._load_local_attachments(logical)

    assert errors == []
    assert attachments == [
        {
            "filename": "report.txt",
            "mime_type": "text/plain",
            "data": b"hello",
            "source_path": logical,
        }
    ]


def test_load_local_attachments_rejects_absolute_paths_outside_artifacts(monkeypatch, tmp_path):
    artifact_root = artifact_outdir_for(tmp_path / "runtime")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    monkeypatch.setattr(gmail_tools, "_current_artifact_context", lambda: (artifact_root, "turn_test"))

    attachments, errors = gmail_tools._load_local_attachments(str(outside))

    assert attachments == []
    assert errors and errors[0]["code"] == "attachment_not_found"


def test_connected_account_provider_auth_failure_returns_consent_envelope():
    credential = ConnectedAccountCredential(
        ok=True,
        account_id="acct-1",
        provider_id="google",
        connector_app_id="gmail",
        claim="gmail:read",
        tool_name="gmail.read_gmail_message",
        tenant="demo-tenant",
        project="demo-project",
    )

    envelope = credential.consent_required_envelope(
        where="gmail.read_gmail_message",
        message="Gmail rejected the stored authorization.",
    )

    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "needs_connected_account_consent"
    assert envelope["consent"]["provider_id"] == "google"
    assert envelope["consent"]["connector_app_id"] == "gmail"
    assert envelope["consent"]["claims"] == ["gmail:read"]
    assert envelope["action_label"] == "Open Connection Hub"
    assert envelope["error"]["action_url"] == envelope["consent"]["url"]
    assert envelope["ret"]["action_url"] == envelope["consent"]["url"]
    assert envelope["consent"]["url"].endswith(
        "/connection-hub%401-0/widgets/connections_settings"
        "?tab=delegated_to_kdcube&provider_id=google&connector_app_id=gmail"
        "&claims=gmail%3Aread&tool_name=gmail.read_gmail_message&account_id=acct-1"
    )
    assert envelope["consent"]["account_id"] == "acct-1"


def test_disabled_gmail_api_preserves_provider_reason(caplog):
    response = httpx.Response(
        403,
        json={
            "error": {
                "message": "Gmail API is disabled for this project.",
                "status": "PERMISSION_DENIED",
                "details": [{"reason": "SERVICE_DISABLED"}],
            }
        },
        request=httpx.Request("GET", "https://gmail.googleapis.com/gmail/v1/users/me/messages"),
    )

    failure = gmail_tools._gmail_failure(
        response,
        operation="search",
        fallback="Gmail search failed.",
        where="gmail.search_gmail",
    )
    result = failure.error_result(where="gmail.search_gmail")

    assert failure.credential_failure is False
    assert result["error"]["code"] == "gmail_provider_configuration_error"
    assert result["ret"]["provider_status"] == 403
    assert result["ret"]["provider_reason"] == "SERVICE_DISABLED"
    assert "provider_reason=SERVICE_DISABLED" in caplog.text


def test_load_inline_attachments_accepts_base64_and_bounds_size():
    """The automation lane for drafts: no KDCube workspace exists there, so
    attachment bytes ride the call itself and must be validated in place."""
    payload = [
        {
            "filename": "letter.zip",
            "content_base64": base64.b64encode(b"zip bytes").decode("ascii"),
            "mime_type": "application/zip",
        },
        {"filename": "broken.pdf", "content_base64": "not-base64!!"},
        {"filename": "empty.pdf"},
        "not-an-object",
    ]
    attachments, errors = gmail_tools._load_inline_attachments(payload)
    assert [item["filename"] for item in attachments] == ["letter.zip"]
    assert attachments[0]["data"] == b"zip bytes"
    assert attachments[0]["mime_type"] == "application/zip"
    assert attachments[0]["source_path"] == "inline:0"
    assert sorted(err["code"] for err in errors) == [
        "attachment_content_required",
        "attachment_invalid",
        "attachment_invalid_base64",
    ]

    oversized = [{
        "filename": "big.bin",
        "content_base64": base64.b64encode(
            b"x" * (gmail_tools.MAX_ATTACHMENT_BYTES + 1)
        ).decode("ascii"),
    }]
    attachments, errors = gmail_tools._load_inline_attachments(oversized)
    assert attachments == []
    assert errors[0]["code"] == "attachment_too_large"

    # JSON-string form (how an MCP string parameter arrives) parses the same.
    import json as _json

    attachments, errors = gmail_tools._load_inline_attachments(
        _json.dumps([{"filename": "a.txt", "content_base64": base64.b64encode(b"hi").decode("ascii")}])
    )
    assert not errors and attachments[0]["data"] == b"hi"


def test_create_gmail_draft_is_registered_with_compose_claim():
    """Drafts must exist as their own tool (send authority is exactly what the
    human-in-the-loop flow may not hold), gated by gmail:compose alone."""
    assert gmail_tools.GMAIL_COMPOSE_CLAIM == "gmail:compose"
    tool = getattr(gmail_tools.tools, "create_gmail_draft", None)
    assert tool is not None
    assert callable(tool)
    description = getattr(tool, "__kernel_function_description__", "")
    assert "without sending" in description or "DRAFT" in description
