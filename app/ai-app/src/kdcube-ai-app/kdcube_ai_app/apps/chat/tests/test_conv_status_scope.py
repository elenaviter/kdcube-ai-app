# SPDX-License-Identifier: MIT

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kdcube_ai_app.apps.chat.ingress import ingress_core


class _FakeIdx:
    def __init__(self, row):
        self.row = row
        self.calls = []

    async def get_conversation_state_row(self, **kwargs):
        self.calls.append(kwargs)
        return self.row


class _FakeChatComm:
    def __init__(self):
        self.calls = []

    async def emit_conv_status(self, *args, **kwargs):
        self.calls.append((args, kwargs))


class _FakeConversationBrowser:
    def __init__(self, row, *, exists: bool):
        self.idx = _FakeIdx(row)
        self.exists = exists
        self.exists_calls = []

    async def conversation_exists(self, **kwargs):
        self.exists_calls.append(kwargs)
        return self.exists


@pytest.mark.asyncio
async def test_get_conversation_status_returns_404_when_conversation_not_found_for_user(monkeypatch):
    async def _load_registry(app, tenant, project):
        del app
        assert tenant == "tenant-a"
        assert project == "project-a"
        return SimpleNamespace(bundles={"bundle.scoped": object()}, default_bundle_id="bundle.scoped")

    monkeypatch.setattr(ingress_core, "_load_active_registry", _load_registry)

    browser = _FakeConversationBrowser(None, exists=False)
    app = SimpleNamespace(state=SimpleNamespace(conversation_browser=browser))
    session = SimpleNamespace(session_id="sess-1", user_id="user-1")

    with pytest.raises(ingress_core.HTTPException) as exc:
        await ingress_core.get_conversation_status(
            app=app,
            chat_comm=_FakeChatComm(),
            session=session,
            tenant="tenant-a",
            project="project-a",
            bundle_id=None,
            conversation_id="conv-foreign",
            stream_id="stream-1",
        )

    assert exc.value.status_code == 404
    assert exc.value.detail == "Conversation not found"
    assert browser.idx.calls == [
        {
            "user_id": "user-1",
            "conversation_id": "conv-foreign",
            "bundle_ids": ["bundle.scoped"],
        }
    ]
    assert browser.exists_calls == [
        {
            "user_id": "user-1",
            "conversation_id": "conv-foreign",
            "bundle_ids": ["bundle.scoped"],
        }
    ]


@pytest.mark.asyncio
async def test_get_conversation_status_returns_404_when_bundle_id_not_in_tenant_project_scope(monkeypatch):
    async def _load_registry(app, tenant, project):
        del app
        return SimpleNamespace(bundles={"bundle.scoped": object()}, default_bundle_id="bundle.scoped")

    monkeypatch.setattr(ingress_core, "_load_active_registry", _load_registry)

    idx = _FakeIdx(
        {
            "id": 1,
            "tags": ["artifact:conversation.state", "conv.state:idle"],
            "ts": None,
            "bundle_id": "bundle.scoped",
        }
    )
    app = SimpleNamespace(state=SimpleNamespace(conversation_browser=SimpleNamespace(idx=idx)))
    session = SimpleNamespace(session_id="sess-1", user_id="user-1")

    with pytest.raises(ingress_core.HTTPException) as exc:
        await ingress_core.get_conversation_status(
            app=app,
            chat_comm=_FakeChatComm(),
            session=session,
            tenant="tenant-a",
            project="project-a",
            bundle_id="bundle.foreign",
            conversation_id="conv-1",
            stream_id="stream-1",
        )

    assert exc.value.status_code == 404
    assert exc.value.detail == "Conversation not found"
    assert idx.calls == []


@pytest.mark.asyncio
async def test_get_conversation_status_returns_idle_when_state_row_missing_but_conversation_exists(monkeypatch):
    async def _load_registry(app, tenant, project):
        del app
        return SimpleNamespace(bundles={"bundle.scoped": object()}, default_bundle_id="bundle.scoped")

    monkeypatch.setattr(ingress_core, "_load_active_registry", _load_registry)

    browser = _FakeConversationBrowser(None, exists=True)
    app = SimpleNamespace(state=SimpleNamespace(conversation_browser=browser))
    session = SimpleNamespace(session_id="sess-1", user_id="user-1", user_type=SimpleNamespace(value="registered"))

    result = await ingress_core.get_conversation_status(
        app=app,
        chat_comm=_FakeChatComm(),
        session=session,
        tenant="tenant-a",
        project="project-a",
        bundle_id=None,
        conversation_id="conv-owned",
        stream_id="stream-1",
        publish=False,
    )

    assert result["conversation_id"] == "conv-owned"
    assert result["state"] == "idle"
    assert result["current_turn_id"] is None
    assert browser.idx.calls == [
        {
            "user_id": "user-1",
            "conversation_id": "conv-owned",
            "bundle_ids": ["bundle.scoped"],
        }
    ]
    assert browser.exists_calls == [
        {
            "user_id": "user-1",
            "conversation_id": "conv-owned",
            "bundle_ids": ["bundle.scoped"],
        }
    ]
