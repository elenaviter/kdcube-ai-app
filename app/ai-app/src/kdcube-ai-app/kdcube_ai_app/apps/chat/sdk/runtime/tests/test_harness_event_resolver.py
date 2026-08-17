# SPDX-License-Identifier: MIT
from __future__ import annotations

"""Conversation-file resolution through the agent harness event scope."""

import pytest

from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.references import (
    build_logical_artifact_path,
)
from kdcube_ai_app.apps.chat.sdk.runtime.harness.events.resolver import (
    CONVERSATION_FILE_EVENT_RESOLVER_ID,
    canonicalize_event_ref_for_context,
    read_event_ref_bytes,
    resolve_event_ref_action,
)
from kdcube_ai_app.apps.chat.sdk.storage.conversation_store import ConversationStore


@pytest.mark.asyncio
async def test_resolve_event_ref_action_downloads_canonical_fi_artifact(tmp_path):
    store = ConversationStore(storage_uri=tmp_path.as_uri())
    await store.put_artifact_file(
        tenant="tenant",
        project="project",
        user="user",
        fingerprint=None,
        conversation_id="conversation",
        turn_id="turn_1",
        relpath="turn_1/files/problem.md",
        data=b"# Problem\n",
        mime="text/markdown",
    )
    ref = build_logical_artifact_path(
        turn_id="turn_1",
        namespace="files",
        relpath="problem.md",
        conversation_id="conversation",
    )

    result = await resolve_event_ref_action(
        {"object_ref": ref, "action": "download"},
        tenant="tenant",
        project="project",
        user_id="user",
        storage_path=tmp_path.as_uri(),
    )

    assert result["ok"] is True
    assert result["resolver"] == CONVERSATION_FILE_EVENT_RESOLVER_ID
    assert result["object_ref"] == ref
    assert result["filename"] == "problem.md"
    assert result["mime"] == "text/markdown"
    assert "content_base64" not in result
    assert result["download_url"] == (
        "/api/cb/resources/tenant/project/conv/user/conversation/turn/turn_1/attachment/turn_1/files/problem.md/download"
    )


@pytest.mark.asyncio
async def test_read_event_ref_bytes_is_scoped_to_current_user(tmp_path):
    store = ConversationStore(storage_uri=tmp_path.as_uri())
    await store.put_artifact_file(
        tenant="tenant",
        project="project",
        user="owner-a",
        fingerprint=None,
        conversation_id="conversation",
        turn_id="turn_1",
        relpath="turn_1/files/secret.txt",
        data=b"secret",
        mime="text/plain",
    )
    ref = build_logical_artifact_path(
        turn_id="turn_1",
        namespace="files",
        relpath="secret.txt",
        conversation_id="conversation",
    )

    data, _meta = await read_event_ref_bytes(
        ref=ref,
        tenant="tenant",
        project="project",
        user_id="owner-a",
        storage_path=tmp_path.as_uri(),
    )
    assert data == b"secret"

    with pytest.raises(FileNotFoundError):
        await read_event_ref_bytes(
            ref=ref,
            tenant="tenant",
            project="project",
            user_id="owner-b",
            storage_path=tmp_path.as_uri(),
        )


@pytest.mark.asyncio
async def test_resolve_event_ref_action_reports_unknown_namespace():
    result = await resolve_event_ref_action(
        {"object_ref": "mem:mem_1", "action": "download"},
        tenant="tenant",
        project="project",
        user_id="user",
    )

    assert result["ok"] is False
    assert result["namespace"] == "mem"
    assert result["error"] == "event_ref_resolver_not_registered"


@pytest.mark.asyncio
async def test_resolve_event_ref_action_treats_bare_fi_refs_as_unregistered():
    """`conv:fi:` is the only conversation-file ref form; a bare `fi:` ref is an
    unregistered namespace and reports the explicit unsupported result."""
    result = await resolve_event_ref_action(
        {"object_ref": "fi:conv_abc.turn_1.files/report.md", "action": "capabilities"},
        tenant="tenant",
        project="project",
        user_id="user",
        require_embedded_conversation=True,
    )

    assert result["ok"] is False
    assert result["namespace"] == "fi"
    assert result["resolver_status"] == "unsupported_namespace"
    assert result["error"] == "event_ref_resolver_not_registered"
    assert result["status"] == 404


@pytest.mark.asyncio
async def test_resolve_event_ref_action_can_require_cross_conversation_fi_refs():
    ref = "conv:fi:turn_2026-06-07.files/problem.md"
    result = await resolve_event_ref_action(
        {"object_ref": ref, "action": "download"},
        tenant="tenant",
        project="project",
        user_id="user",
        require_embedded_conversation=True,
    )

    assert result["ok"] is False
    assert result["object_ref"] == ref
    assert result["error"] == "fi_ref_requires_embedded_conversation"


@pytest.mark.asyncio
async def test_resolve_event_ref_action_allows_current_turn_fi_refs_when_not_required():
    ref = "conv:fi:turn_2026-06-07.files/problem.md"
    result = await resolve_event_ref_action(
        {"object_ref": ref, "action": "capabilities"},
        tenant="tenant",
        project="project",
        user_id="user",
    )

    assert result["ok"] is True
    assert result["object_ref"] == ref
    assert result["namespace"] == "conv:fi"
    assert result["capabilities"]["download"] is True
    assert result["default_open_effect_action"] == "download"


def test_canonicalize_event_ref_for_context_adds_conversation_to_current_turn_fi_refs():
    assert canonicalize_event_ref_for_context(
        "conv:fi:turn_2026-06-07.files/problem.md",
        conversation_id="abc",
    ) == "conv:fi:conv_abc.turn_2026-06-07.files/problem.md"


def test_canonicalize_event_ref_for_context_leaves_other_namespaces_unchanged():
    ref = "fi:turn_2026-06-07.files/problem.md"
    assert canonicalize_event_ref_for_context(ref, conversation_id="abc") == ref


def test_canonicalize_event_ref_for_context_preserves_existing_cross_conversation_fi_refs():
    ref = "conv:fi:conv_other.turn_2026-06-07.files/problem.md"
    assert canonicalize_event_ref_for_context(ref, conversation_id="abc") == ref


@pytest.mark.asyncio
async def test_uploaded_attachment_resolves_by_its_plain_filename_ref(tmp_path):
    """User uploads are STORED under `{ts}-{filename}` while their conv:fi:
    refs (turn recorder / Files tab / pull) carry the PLAIN filename — the
    resolver must bridge the two, for download AND for pull byte reads."""
    store = ConversationStore(storage_uri=tmp_path.as_uri())
    await store.put_attachment(
        tenant="tenant",
        project="project",
        user="user",
        fingerprint=None,
        conversation_id="conversation",
        turn_id="turn_1",
        role="user",
        filename="report.docx",
        data=b"DOCX-BYTES",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        origin="user",
    )
    # The literal shape the turn recorder / Files tab publish for uploads.
    ref = "conv:fi:conv_conversation.turn_1.user.attachments/report.docx"

    data, meta = await read_event_ref_bytes(
        ref=ref,
        tenant="tenant",
        project="project",
        user_id="user",
        storage_path=tmp_path.as_uri(),
    )
    assert data == b"DOCX-BYTES"
    assert meta["turn_id"] == "turn_1"
    assert meta["storage_relpath"].endswith("-report.docx")

    result = await resolve_event_ref_action(
        {"object_ref": ref, "action": "download"},
        tenant="tenant",
        project="project",
        user_id="user",
        storage_path=tmp_path.as_uri(),
    )
    assert result["ok"] is True
    assert result["filename"] == "report.docx"
    assert result["size"] == len(b"DOCX-BYTES")
