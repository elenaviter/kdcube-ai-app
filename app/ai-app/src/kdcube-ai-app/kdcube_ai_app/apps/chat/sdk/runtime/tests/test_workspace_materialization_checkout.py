# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import kdcube_ai_app.apps.chat.sdk.runtime.harness.events.resolver as resolver
from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.checkout import (
    WorkspaceCheckoutError,
    checkout_workspace_items,
)
from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.materialization import (
    MaterializedWorkspaceSource,
    pull_refs_into_workspace,
)


def _fake_bytes(payload: dict[str, tuple[bytes, str]]):
    async def read_event_ref_bytes(
        *, ref, tenant, project, user_id, storage_path=None, conversation_id=""
    ):
        if ref not in payload:
            raise FileNotFoundError(ref)
        data, relpath = payload[ref]
        source_conversation = "c2" if "conv_c2" in ref else (conversation_id or "c1")
        return data, {
            "conversation_id": source_conversation,
            "turn_id": "turn_1",
            "namespace": "files",
            "relpath": relpath,
        }

    return read_event_ref_bytes


def test_pull_preserves_source_identity_and_avoids_basename_collisions(tmp_path, monkeypatch):
    refs = {
        "conv:fi:conv_c1.turn_1.files/a/source.pdf": (b"first", "a/source.pdf"),
        "conv:fi:conv_c2.turn_1.files/b/source.pdf": (b"second", "b/source.pdf"),
    }
    monkeypatch.setattr(resolver, "read_event_ref_bytes", _fake_bytes(refs))

    reports = asyncio.run(pull_refs_into_workspace(
        refs=list(refs),
        artifact_root=tmp_path,
        tenant="tenant",
        project="project",
        user_id="user",
        conversation_id="c1",
    ))

    assert [row["ok"] for row in reports] == [True, True]
    assert (tmp_path / "turn_1/files/a/source.pdf").read_bytes() == b"first"
    assert (tmp_path / "conv_c2/turn_1/files/b/source.pdf").read_bytes() == b"second"
    assert reports[0]["physical_path"] == "turn_1/files/a/source.pdf"
    assert reports[1]["physical_path"] == "conv_c2/turn_1/files/b/source.pdf"


def test_pull_rejects_event_record_and_names_object_ref_transition(tmp_path):
    reports = asyncio.run(pull_refs_into_workspace(
        refs=["conv:ev:conv_c1.turn_1.events/canvas/evt_1"],
        artifact_root=tmp_path,
        tenant="tenant",
        project="project",
        user_id="user",
        conversation_id="c1",
    ))

    assert reports[0]["ok"] is False
    assert reports[0]["error_code"] == "record_ref_not_materializable"
    assert "object_ref" in reports[0]["error"]


def test_pull_rejects_symlinked_materialization_target(tmp_path, monkeypatch):
    ref = "conv:fi:conv_c1.turn_1.files/source.txt"
    monkeypatch.setattr(resolver, "read_event_ref_bytes", _fake_bytes({ref: (b"safe", "source.txt")}))
    root = tmp_path / "workspace"
    outside = tmp_path / "outside"
    outside.mkdir()
    root.mkdir()
    (root / "turn_1").symlink_to(outside, target_is_directory=True)

    reports = asyncio.run(pull_refs_into_workspace(
        refs=[ref],
        artifact_root=root,
        tenant="tenant",
        project="project",
        user_id="user",
        conversation_id="c1",
    ))

    assert reports[0]["ok"] is False
    assert reports[0]["error_code"] == "materialization_target_symlink_not_allowed"
    assert not (outside / "files/source.txt").exists()


def test_pull_uses_trusted_owner_resolver_and_returns_pinned_ref(tmp_path):
    source = tmp_path / "provider" / "board.md"
    source.parent.mkdir()
    source.write_text("board", encoding="utf-8")

    async def owner_resolver(*, ref, staging_dir):
        assert ref == "cnv:boards/card-7"
        return MaterializedWorkspaceSource(
            requested_ref=ref,
            object_ref=ref,
            resolved_ref="conv:fi:conv_c1.turn_2.git/snapshots/cnv/card-7/board.md",
            local_path=source,
            mime="text/markdown",
        )

    reports = asyncio.run(pull_refs_into_workspace(
        refs=["cnv:boards/card-7"],
        artifact_root=tmp_path / "workspace",
        tenant="tenant",
        project="project",
        user_id="user",
        conversation_id="c1",
        source_resolver=owner_resolver,
    ))

    assert reports[0]["ok"] is True
    assert reports[0]["object_ref"] == "cnv:boards/card-7"
    assert reports[0]["logical_path"].startswith("conv:fi:conv_c1.")
    assert (tmp_path / "workspace/turn_2/git/snapshots/cnv/card-7/board.md").read_text() == "board"


def test_checkout_file_replace_is_repeatable_reset(tmp_path):
    source = tmp_path / "remote" / "source.pdf"
    source.parent.mkdir()
    source.write_bytes(b"original")

    async def source_resolver(*, ref, staging_dir):
        return MaterializedWorkspaceSource(
            requested_ref=ref,
            resolved_ref="conv:fi:conv_c1.turn_1.files/review/source.pdf",
            local_path=source,
            mime="application/pdf",
        )

    kwargs = dict(
        items=[{
            "from": "conv:fi:conv_c1.turn_1.files/review/source.pdf",
            "to": "files/pdf-review/working.pdf",
            "strategy": "replace",
        }],
        artifact_root=tmp_path / "workspace",
        current_turn_id="turn_9",
        tenant="tenant",
        project="project",
        user_id="user",
        conversation_id="c1",
        source_resolver=source_resolver,
    )
    first = asyncio.run(checkout_workspace_items(**kwargs))
    editable = tmp_path / "workspace/turn_9/files/pdf-review/working.pdf"
    editable.write_bytes(b"bad edit")
    second = asyncio.run(checkout_workspace_items(**kwargs))

    assert first["ok"] and second["ok"]
    assert editable.read_bytes() == b"original"
    assert second["items"][0]["logical_path"].endswith("turn_9.files/pdf-review/working.pdf")


def test_checkout_directory_replace_and_overlay_have_distinct_semantics(tmp_path):
    replace_source = tmp_path / "replace"
    (replace_source / "src").mkdir(parents=True)
    (replace_source / "src/app.py").write_text("v1", encoding="utf-8")
    overlay_source = tmp_path / "overlay"
    (overlay_source / "src").mkdir(parents=True)
    (overlay_source / "src/app.py").write_text("v2", encoding="utf-8")
    (overlay_source / "docs").mkdir()
    (overlay_source / "docs/readme.md").write_text("docs", encoding="utf-8")

    async def source_resolver(*, ref, staging_dir):
        source = overlay_source if ref.endswith("overlay") else replace_source
        return MaterializedWorkspaceSource(
            requested_ref=ref,
            resolved_ref="conv:fi:conv_c1.turn_1.git/projects/app",
            local_path=source,
            kind="directory",
        )

    root = tmp_path / "workspace"
    asyncio.run(checkout_workspace_items(
        items=[{
            "from": "project:replace",
            "to": "git/projects/app",
            "strategy": "replace",
        }],
        artifact_root=root,
        current_turn_id="turn_9",
        tenant="tenant",
        project="project",
        user_id="user",
        conversation_id="c1",
        source_resolver=source_resolver,
    ))
    target = root / "turn_9/git/projects/app"
    (target / "local.txt").write_text("keep", encoding="utf-8")
    asyncio.run(checkout_workspace_items(
        items=[{
            "from": "project:overlay",
            "to": "git/projects/app",
            "strategy": "overlay",
        }],
        artifact_root=root,
        current_turn_id="turn_9",
        tenant="tenant",
        project="project",
        user_id="user",
        conversation_id="c1",
        source_resolver=source_resolver,
    ))

    assert (target / "src/app.py").read_text() == "v2"
    assert (target / "docs/readme.md").read_text() == "docs"
    assert (target / "local.txt").read_text() == "keep"


def test_checkout_overlay_rejects_symlinks_already_in_target(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "safe.txt").write_text("safe", encoding="utf-8")

    async def source_resolver(*, ref, staging_dir):
        return MaterializedWorkspaceSource(
            requested_ref=ref,
            resolved_ref="conv:fi:conv_c1.turn_1.git/projects/app",
            local_path=source,
            kind="directory",
        )

    root = tmp_path / "workspace"
    target = root / "turn_9/git/projects/app"
    target.mkdir(parents=True)
    secret = tmp_path / "secret.txt"
    secret.write_text("secret", encoding="utf-8")
    (target / "secret-link").symlink_to(secret)

    with pytest.raises(WorkspaceCheckoutError) as error:
        asyncio.run(checkout_workspace_items(
            items=[{"from": "project:source", "to": "git/projects/app", "strategy": "overlay"}],
            artifact_root=root,
            current_turn_id="turn_9",
            tenant="tenant",
            project="project",
            user_id="user",
            conversation_id="c1",
            source_resolver=source_resolver,
        ))
    assert error.value.code == "checkout_target_symlink_not_allowed"


@pytest.mark.parametrize("target", ["../escape", "/tmp/file", "turn_9/files/file", "attachments/file"])
def test_checkout_rejects_targets_outside_current_editable_areas(tmp_path, target):
    with pytest.raises(WorkspaceCheckoutError):
        asyncio.run(checkout_workspace_items(
            items=[{"from": "conv:fi:conv_c1.turn_1.files/a", "to": target, "strategy": "replace"}],
            artifact_root=tmp_path,
            current_turn_id="turn_9",
            tenant="tenant",
            project="project",
            user_id="user",
            conversation_id="c1",
        ))


def test_checkout_rejects_overlapping_batch_before_resolving(tmp_path):
    with pytest.raises(WorkspaceCheckoutError) as error:
        asyncio.run(checkout_workspace_items(
            items=[
                {"from": "a:1", "to": "files/review", "strategy": "replace"},
                {"from": "a:2", "to": "files/review/source.pdf", "strategy": "replace"},
            ],
            artifact_root=tmp_path,
            current_turn_id="turn_9",
            tenant="tenant",
            project="project",
            user_id="user",
            conversation_id="c1",
        ))
    assert error.value.code == "checkout_targets_overlap"
