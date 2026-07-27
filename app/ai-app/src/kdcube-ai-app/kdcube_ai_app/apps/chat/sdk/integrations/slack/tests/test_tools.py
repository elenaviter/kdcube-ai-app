from __future__ import annotations

import httpx
import pytest

from kdcube_ai_app.apps.chat.sdk.integrations.connected_accounts import (
    ConnectedAccountCredential,
)
from kdcube_ai_app.apps.chat.sdk.integrations.slack import tools
from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace import artifact_outdir_for, resolve_artifact_path
from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.references import (
    ARTIFACT_NAMESPACE_FILES,
    build_physical_artifact_path,
    physical_path_to_logical_path,
)


def test_compact_message_includes_file_metadata():
    out = tools._compact_message(
        {
            "type": "message",
            "user": "U1",
            "text": "see file",
            "ts": "123.456",
            "files": [
                {
                    "id": "F1",
                    "name": "report.pdf",
                    "title": "Report",
                    "mimetype": "application/pdf",
                    "size": 12,
                    "url_private_download": "https://files.slack.com/private",
                }
            ],
        }
    )

    assert out["timestamp"] == "123.456"
    assert out["file_count"] == 1
    assert out["files"][0]["id"] == "F1"
    assert out["files"][0]["name"] == "report.pdf"


def test_load_upload_file_resolves_kdcube_artifacts(monkeypatch, tmp_path):
    artifact_root = artifact_outdir_for(tmp_path / "runtime")
    physical = build_physical_artifact_path(
        turn_id="turn_test",
        namespace=ARTIFACT_NAMESPACE_FILES,
        relpath="reports/report.txt",
    )
    target = resolve_artifact_path(artifact_root, physical, prefer_existing=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("hello slack", encoding="utf-8")
    logical = physical_path_to_logical_path(physical)

    monkeypatch.setattr(tools, "_current_artifact_context", lambda: (artifact_root, "turn_test"))

    upload_file, error = tools._load_upload_file(logical)

    assert error is None
    assert upload_file is not None
    assert upload_file["filename"] == "report.txt"
    assert upload_file["mime_type"] == "text/plain"
    assert upload_file["data"] == b"hello slack"


def test_load_upload_file_rejects_absolute_paths_outside_artifacts(monkeypatch, tmp_path):
    artifact_root = artifact_outdir_for(tmp_path / "runtime")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    monkeypatch.setattr(tools, "_current_artifact_context", lambda: (artifact_root, "turn_test"))

    upload_file, error = tools._load_upload_file(str(outside))

    assert upload_file is None
    assert error is not None
    assert error["code"] == "file_not_found"


@pytest.mark.asyncio
async def test_common_slack_call_returns_auth_retry_marker(monkeypatch):
    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        async def get(self, *args, **kwargs):
            return httpx.Response(
                200,
                json={"ok": False, "error": "invalid_auth"},
                request=httpx.Request("GET", "https://slack.com/api/conversations.list"),
            )

    monkeypatch.setattr(tools.httpx, "AsyncClient", lambda **kwargs: FakeClient())
    credential = ConnectedAccountCredential(
        ok=True,
        access_token="secret",
        account_id="acct-1",
        provider_id="slack",
        connector_app_id="slack-demo",
        claim="slack:channels",
        tool_name="slack.list_slack_channels",
        tenant="demo-tenant",
        project="demo-project",
    )

    data, error = await tools.SlackTools()._call_json(
        credential=credential,
        method="conversations.list",
        where="slack.list_slack_channels",
    )

    assert data is None
    assert error is not None
    assert set(error) == {"__connected_account_auth_failure__"}


def test_generic_slack_403_is_access_denied_not_auth_failure(caplog):
    response = httpx.Response(
        403,
        json={"ok": False, "error": "access_denied"},
        request=httpx.Request("GET", "https://slack.com/api/files.info"),
    )

    failure = tools._slack_failure(
        response,
        operation="files.info",
        fallback="Slack file lookup failed.",
        where="slack.download_slack_file",
    )

    assert failure.category == "access_denied"
    assert failure.credential_failure is False
    assert failure.error_result(where="slack.download_slack_file")["ret"][
        "provider_status"
    ] == 403
    assert "provider_reason=access_denied" in caplog.text
