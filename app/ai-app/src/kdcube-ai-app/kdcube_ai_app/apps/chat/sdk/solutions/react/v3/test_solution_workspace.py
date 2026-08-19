# SPDX-License-Identifier: MIT

import pytest


@pytest.mark.asyncio
async def test_host_files_rejects_non_image_bytes_declared_as_png(tmp_path):
    from kdcube_ai_app.apps.chat.sdk.solutions.react.solution_workspace import ApplicationHostingService

    class _FakeStore:
        def __init__(self):
            self.calls = []

        async def put_artifact_file(self, **kwargs):
            self.calls.append(kwargs)
            return "file:///unexpected", "unexpected", "rn:unexpected"

    class _FakeLogger:
        def __init__(self):
            self.messages = []

        def log(self, message, *, level):
            self.messages.append((level, message))

    invalid_path = tmp_path / "extracted_image.png"
    invalid_path.write_bytes(b"x" * 32)
    store = _FakeStore()
    logger = _FakeLogger()
    hosting = ApplicationHostingService(store=store, logger=logger)

    hosted = await hosting.host_files_to_conversation(
        rid="rid_1",
        files=[{
            "type": "file",
            "output": {
                "path": str(invalid_path),
                "mime": "image/png",
            },
            "resource_id": "artifact:extracted_image",
        }],
        outdir=tmp_path,
        tenant="tenant",
        project="project",
        user="user",
        conversation_id="conv_1",
        user_type="user",
        turn_id="turn_1",
    )

    assert hosted == []
    assert store.calls == []
    assert logger.messages == [(
        "ERROR",
        f"[host_files] Rejected invalid image {invalid_path}: invalid_image_data",
    )]


@pytest.mark.asyncio
async def test_emit_solver_artifacts_preserves_transport_fields():
    from kdcube_ai_app.apps.chat.sdk.solutions.react.solution_workspace import ApplicationHostingService

    class _FakeComm:
        service = {"conversation_id": "conv_1"}

        def __init__(self):
            self.events = []

        async def event(self, **kwargs):
            self.events.append(kwargs)

    comm = _FakeComm()
    hosting = ApplicationHostingService(store=None, comm=comm)

    await hosting.emit_solver_artifacts(
        files=[
            {
                "filename": "diagram-scene-hub.svg",
                "mime": "image/svg+xml",
                "visibility": "external",
                "logical_path": "conv:fi:conv_1.turn_1.user.attachments/named_services/task/digest/diagram-scene-hub.svg",
                "physical_path": "turn_1/attachments/named_services/task/digest/diagram-scene-hub.svg",
                "hosted_uri": "s3://bucket/cb/tenants/t/projects/p/attachments/u/conv_1/turn_1/diagram-scene-hub.svg",
                "key": "cb/tenants/t/projects/p/attachments/u/conv_1/turn_1/diagram-scene-hub.svg",
                "rn": "rn:file",
                "content_sha256": "a" * 64,
            }
        ],
        citations=[],
    )

    assert len(comm.events) == 1
    event = comm.events[0]
    assert event["type"] == "chat.files"
    item = event["data"]["items"][0]
    assert item["logical_path"] == "conv:fi:conv_1.turn_1.user.attachments/named_services/task/digest/diagram-scene-hub.svg"
    assert item["physical_path"] == "turn_1/attachments/named_services/task/digest/diagram-scene-hub.svg"
    assert item["hosted_uri"].startswith("s3://bucket/")
    assert item["key"].startswith("cb/tenants/")
    assert item["rn"] == "rn:file"
    assert item["content_sha256"] == "a" * 64
    assert item["object_ref"] == item["logical_path"]
    assert item["ref"] == item["logical_path"]


@pytest.mark.asyncio
async def test_host_artifact_file_overwrites_stale_content_metadata(tmp_path):
    from types import SimpleNamespace

    from kdcube_ai_app.apps.chat.sdk.solutions.react.tools.common import host_artifact_file

    authoritative_sha = "b" * 64

    class _FakeHosting:
        async def host_files_to_conversation(self, **_kwargs):
            return [
                {
                    "hosted_uri": "file:///store/turn_1/files/report.md",
                    "key": "store/turn_1/files/report.md",
                    "rn": "rn:file",
                    "physical_path": "turn_1/files/report.md",
                    "size": 23,
                    "content_sha256": authoritative_sha,
                }
            ]

    comm = SimpleNamespace(
        service={
            "tenant": "tenant",
            "project": "project",
            "user": "user",
            "conversation_id": "conv_1",
        },
        user_id="user",
        user_type="user",
    )
    artifact = {
        "path": "turn_1/files/report.md",
        "content_sha256": "stale-top-level",
        "size_bytes": 1,
        "value": {
            "path": "turn_1/files/report.md",
            "content_sha256": "stale-nested",
            "size_bytes": 1,
        },
    }

    hosted = await host_artifact_file(
        hosting_service=_FakeHosting(),
        comm=comm,
        runtime_ctx=SimpleNamespace(conversation_id="conv_1", turn_id="turn_1"),
        artifact=artifact,
        outdir=tmp_path,
    )

    assert hosted[0]["content_sha256"] == authoritative_sha
    assert artifact["content_sha256"] == authoritative_sha
    assert artifact["value"]["content_sha256"] == authoritative_sha
    assert artifact["size_bytes"] == 23
    assert artifact["value"]["size_bytes"] == 23


def test_artifact_content_fingerprint_is_transport_only():
    import json

    from kdcube_ai_app.apps.chat.sdk.solutions.react.artifacts import (
        artifact_transport_metadata,
        build_artifact_meta_block,
    )

    sha = "c" * 64
    block = build_artifact_meta_block(
        turn_id="turn_1",
        tool_call_id="tc_1",
        artifact={
            "value": {"content_sha256": sha, "size_bytes": 12},
            "visibility": "external",
        },
        artifact_path="conv:fi:conv_1.turn_1.files/report.md",
        physical_path="turn_1/files/report.md",
    )

    assert "content_sha256" not in json.loads(block["text"])
    assert block["meta"]["content_sha256"] == sha
    assert artifact_transport_metadata(block) == {"content_sha256": sha}
