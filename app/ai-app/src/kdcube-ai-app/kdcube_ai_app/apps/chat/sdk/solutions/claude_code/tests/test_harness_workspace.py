# SPDX-License-Identifier: MIT

from __future__ import annotations

import asyncio
from pathlib import Path

from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.materialization import (
    MaterializedWorkspaceSource,
)
from kdcube_ai_app.apps.chat.sdk.solutions.claude_code.harness_workspace import (
    CLAUDE_CODE_TURN_WORKSPACE_PERMISSION,
    bind_claude_code_turn_workspace,
)
from kdcube_ai_app.apps.chat.sdk.solutions.claude_code.types import (
    ClaudeCodeWorkspaceConfig,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.publication import (
    WorkspacePublicationDecision,
    WorkspacePublicationPolicy,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.workspace_broker import (
    request_workspace_broker,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.workspace_tools import (
    WORKSPACE_MCP_SERVER_ID,
    WORKSPACE_TURN_SUMMARY_TOOL,
)


class _Hosting:
    def __init__(self):
        self.comm = None
        self.hosted = []
        self.emitted = []

    async def host_files_to_conversation(self, **kwargs):
        rows = []
        for artifact in kwargs["files"]:
            physical = artifact["output"]["path"]
            source = Path(kwargs["outdir"]) / "workdir" / physical
            assert source.is_file()
            rows.append({
                "filename": source.name,
                "logical_path": f"conv:fi:conv_{kwargs['conversation_id']}.{physical.replace('/', '.', 1)}",
                "mime": artifact["mime"],
            })
        self.hosted.extend(rows)
        return rows

    async def emit_solver_artifacts(self, *, files, citations):
        assert citations == []
        self.emitted.extend(files)


def test_reusable_binding_owns_broker_server_config_and_publication(tmp_path):
    async def scenario():
        async def resolve(*, ref, staging_dir):
            source = Path(staging_dir) / "source.txt"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("source", encoding="utf-8")
            return MaterializedWorkspaceSource(
                requested_ref=ref,
                resolved_ref="conv:fi:conv_c1.turn_old.files/source.txt",
                local_path=source,
                object_ref=ref,
                mime="text/plain",
            )

        approvals = []

        async def approve(request):
            approvals.append(request)
            return WorkspacePublicationDecision.approved()

        hosting = _Hosting()
        turn_state = {"turn_id": "turn_9"}
        binding = await bind_claude_code_turn_workspace(
            workspace=tmp_path,
            tenant="tenant",
            project="project",
            user_id="user",
            conversation_id="c1",
            turn_id="turn_9",
            source_resolver=resolve,
            hosting_service=hosting,
            publication_policy=WorkspacePublicationPolicy(approver=approve),
            state=turn_state,
            turn_summary_enabled=True,
            user_type="registered",
            request_id="request-9",
        )
        socket_path = binding.broker.socket_path

        servers = binding.merge_mcp_servers({
            "provider": {"type": "http", "url": "https://example.invalid/mcp"},
        })
        config = binding.apply_workspace_config(
            ClaudeCodeWorkspaceConfig(
                mcp_servers=servers,
                allowed_tools=("Read", CLAUDE_CODE_TURN_WORKSPACE_PERMISSION),
                instructions_markdown="# Product instructions\n",
            )
        )
        assert set(config.mcp_servers) == {"provider", WORKSPACE_MCP_SERVER_ID}
        assert config.enabled_mcp_servers == ("provider", WORKSPACE_MCP_SERVER_ID)
        assert config.allowed_tools.count(CLAUDE_CODE_TURN_WORKSPACE_PERMISSION) == 1
        assert "KDCube turn workspace" in config.instructions_markdown
        assert ".kdcube/turn-workspace/turn_9" in config.instructions_markdown
        assert config.instructions_markdown.count("kdcube-agent-harness-workspace") == 1
        assert "record_turn_summary" in config.instructions_markdown
        assert WORKSPACE_TURN_SUMMARY_TOOL not in config.denied_tools

        resolved = await request_workspace_broker(
            socket_path=str(socket_path),
            token=binding.broker.token,
            operation="materialize",
            payload={"ref": "cnv:main@7"},
        )
        assert resolved["object_ref"] == "cnv:main@7"

        output = tmp_path / ".kdcube" / "turn-workspace" / "turn_9" / "files" / "result.md"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("result", encoding="utf-8")
        published = await request_workspace_broker(
            socket_path=str(socket_path),
            token=binding.broker.token,
            operation="publish",
            payload={"paths": ["files/result.md"]},
        )
        assert published == hosting.emitted
        assert approvals[0].request_id == "request-9"
        assert approvals[0].files[0].relative_path == "result.md"

        contribution = await request_workspace_broker(
            socket_path=str(socket_path),
            token=binding.broker.token,
            operation="record_turn_summary",
            payload={
                "summary": "Prepared the reviewed publication.",
                "refs": [published[0]["logical_path"]],
                "phrases": ["reviewed publication"],
                "entities": ["result.md"],
            },
        )
        assert contribution["status"] == "staged"
        assert turn_state["_kdcube_turn_summary_contribution"]["contributor"] == "claude_code"
        assert turn_state["_kdcube_turn_summary_contribution"]["refs"] == [
            published[0]["logical_path"]
        ]

        await binding.close()
        assert binding.closed
        assert not socket_path.exists()

    asyncio.run(scenario())


def test_reusable_binding_denies_publish_tool_when_hosting_is_unavailable(tmp_path):
    async def scenario():
        binding = await bind_claude_code_turn_workspace(
            workspace=tmp_path,
            tenant="tenant",
            project="project",
            user_id="user",
            conversation_id="c1",
            turn_id="turn_9",
            hosting_service=None,
        )
        config = binding.apply_workspace_config(ClaudeCodeWorkspaceConfig())
        assert "mcp__turn_workspace__publish" in config.denied_tools
        assert WORKSPACE_TURN_SUMMARY_TOOL in config.denied_tools
        assert "publication is unavailable" in config.instructions_markdown
        assert "record_turn_summary" not in config.instructions_markdown
        await binding.close()

    asyncio.run(scenario())
