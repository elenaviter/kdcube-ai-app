# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.materialization import (
    MaterializedWorkspaceSource,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.workspace_broker import (
    WorkspaceBrokerError,
    broker_source_resolver,
    request_workspace_broker,
    start_workspace_broker,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.workspace_tools import (
    WorkspacePublishError,
    _bound_communicator,
    publish_workspace_files,
    pull_into_workspace,
    workspace_mcp_server,
)


def test_broker_materializes_owner_ref_and_rejects_wrong_token(tmp_path):
    async def scenario():
        async def resolve(*, ref, staging_dir):
            source = Path(staging_dir) / "source.txt"
            source.write_text("owner bytes", encoding="utf-8")
            return MaterializedWorkspaceSource(
                requested_ref=ref,
                resolved_ref="conv:fi:conv_c1.turn_old.files/source.txt",
                local_path=source,
                object_ref=ref,
                mime="text/plain",
            )

        async def publish(*, paths):
            return [{"logical_path": "conv:fi:conv_c1.turn_9.files/result.txt", "paths": paths}]

        async with await start_workspace_broker(
            source_resolver=resolve,
            publisher=publish,
        ) as broker:
            resolver = broker_source_resolver(
                socket_path=str(broker.socket_path),
                token=broker.token,
            )
            rows = await pull_into_workspace(
                ["cnv:main@7"],
                workspace=tmp_path,
                tenant="tenant",
                project="project",
                user_id="user",
                conversation_id="c1",
                source_resolver=resolver,
            )
            assert rows[0]["ok"] is True
            assert rows[0]["object_ref"] == "cnv:main@7"
            assert rows[0]["logical_path"] == "conv:fi:conv_c1.turn_old.files/source.txt"
            assert Path(rows[0]["path"]).read_text(encoding="utf-8") == "owner bytes"

            published = await request_workspace_broker(
                socket_path=str(broker.socket_path),
                token=broker.token,
                operation="publish",
                payload={"paths": ["files/result.txt"]},
            )
            assert published == [{
                "logical_path": "conv:fi:conv_c1.turn_9.files/result.txt",
                "paths": ["files/result.txt"],
            }]

            with pytest.raises(WorkspaceBrokerError) as caught:
                await request_workspace_broker(
                    socket_path=str(broker.socket_path),
                    token="wrong",
                    operation="materialize",
                    payload={"ref": "cnv:main@7"},
                )
            assert caught.value.code == "unauthorized"

    asyncio.run(scenario())


def test_bound_communicator_falls_back_outside_request_task(monkeypatch):
    from kdcube_ai_app.apps.chat.sdk.runtime import comm_ctx

    expected = object()
    monkeypatch.setattr(comm_ctx, "get_comm", lambda: expected)

    class _Entrypoint:
        @property
        def comm(self):
            raise RuntimeError("no active request task")

    assert _bound_communicator(_Entrypoint()) is expected


class _Hosting:
    def __init__(self):
        self.hosted = []
        self.emitted = []

    async def host_files_to_conversation(self, **kwargs):
        outdir = Path(kwargs["outdir"])
        rows = []
        for artifact in kwargs["files"]:
            physical = artifact["output"]["path"]
            source = outdir / "workdir" / physical
            assert source.is_file()
            rows.append({
                "filename": source.name,
                "mime": artifact["mime"],
                "physical_path": physical,
                "logical_path": f"conv:fi:conv_{kwargs['conversation_id']}.{physical.replace('/', '.', 1)}",
                "hosted_uri": f"hosted://{source.name}",
            })
        self.hosted.extend(rows)
        return rows

    async def emit_solver_artifacts(self, *, files, citations):
        assert citations == []
        self.emitted.extend(files)


def test_publish_is_files_only_and_enters_turn_state(tmp_path):
    async def scenario():
        source = tmp_path / ".kdcube" / "turn-workspace" / "turn_9" / "files" / "review" / "report.txt"
        source.parent.mkdir(parents=True)
        source.write_text("report", encoding="utf-8")
        hosting = _Hosting()
        state = {}
        rows = await publish_workspace_files(
            ["files/review/report.txt"],
            workspace=tmp_path,
            turn_id="turn_9",
            hosting_service=hosting,
            tenant="tenant",
            project="project",
            user_id="user",
            user_type="registered",
            conversation_id="c1",
            state=state,
        )
        assert rows == hosting.emitted
        assert state["hosted_files"] == rows
        assert rows[0]["physical_path"] == "turn_9/files/review/report.txt"

        with pytest.raises(WorkspacePublishError) as outside:
            await publish_workspace_files(
                ["git/projects/private.txt"],
                workspace=tmp_path,
                turn_id="turn_9",
                hosting_service=hosting,
                tenant="tenant",
                project="project",
                user_id="user",
                user_type="registered",
                conversation_id="c1",
            )
        assert outside.value.code == "publish_path_outside_files"

        secret = tmp_path / "secret.txt"
        secret.write_text("secret", encoding="utf-8")
        link = source.parent / "leak.txt"
        link.symlink_to(secret)
        with pytest.raises(WorkspacePublishError) as symlink:
            await publish_workspace_files(
                ["files/review/leak.txt"],
                workspace=tmp_path,
                turn_id="turn_9",
                hosting_service=hosting,
                tenant="tenant",
                project="project",
                user_id="user",
                user_type="registered",
                conversation_id="c1",
            )
        assert symlink.value.code == "publish_symlink_not_allowed"

        outside = tmp_path / "outside-files"
        outside.mkdir()
        turn_root = tmp_path / ".kdcube" / "turn-workspace" / "turn_link"
        turn_root.mkdir(parents=True)
        (turn_root / "files").symlink_to(outside, target_is_directory=True)
        (outside / "leak.txt").write_text("secret", encoding="utf-8")
        with pytest.raises(WorkspacePublishError) as root_symlink:
            await publish_workspace_files(
                ["files/leak.txt"],
                workspace=tmp_path,
                turn_id="turn_link",
                hosting_service=hosting,
                tenant="tenant",
                project="project",
                user_id="user",
                user_type="registered",
                conversation_id="c1",
            )
        assert root_symlink.value.code == "publish_symlink_not_allowed"

    asyncio.run(scenario())


def test_workspace_mcp_server_carries_broker_only_in_trusted_env(tmp_path):
    spec = workspace_mcp_server(
        workspace=tmp_path,
        tenant="tenant",
        project="project",
        user_id="user",
        conversation_id="c1",
        turn_id="turn_9",
        broker_socket="/tmp/broker.sock",
        broker_token="secret-token",
    )
    assert spec["env"]["KDCUBE_WS_BROKER_SOCKET"] == "/tmp/broker.sock"
    assert spec["env"]["KDCUBE_WS_BROKER_TOKEN"] == "secret-token"
    assert "broker" not in spec["args"]
