# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""Composition root for the hosted Claude Code harness demonstration."""

from __future__ import annotations

from typing import Any, Dict

from kdcube_ai_app.apps.chat.sdk.protocol import ExternalEventPayload
from kdcube_ai_app.apps.chat.sdk.solutions.chatbot.entrypoint_with_economic import (
    BaseEntrypointWithEconomics,
)
from kdcube_ai_app.apps.chat.sdk.solutions.foreign_runtime.turn_record import (
    persist_turn_artifacts,
)
from kdcube_ai_app.infra.plugin.bundle_loader import bundle_entrypoint, bundle_id
from kdcube_ai_app.infra.service_hub.inventory import Config

from .services.turn import run_claude_demo_turn


BUNDLE_ID = "harness-claude-demo@1-0"


@bundle_entrypoint(name="harness-claude-demo", version="1.0.0", priority=10)
@bundle_id(id=BUNDLE_ID)
class HarnessClaudeDemoEntrypoint(BaseEntrypointWithEconomics):
    """Serve one Claude Code agent through the normal KDCube turn boundary."""

    def __init__(
        self,
        config: Config,
        pg_pool: Any = None,
        redis: Any = None,
        comm_context: ExternalEventPayload = None,
    ) -> None:
        super().__init__(
            config=config,
            pg_pool=pg_pool,
            redis=redis,
            comm_context=comm_context,
        )

    def configuration_defaults(self) -> Dict[str, Any]:
        defaults = {
            "role_models": {
                "harness.claude.title": {
                    "provider": "anthropic",
                    "model": "claude-haiku-4-5-20251001",
                },
            },
            "surfaces": {
                "as_provider": {
                    "bundle": {
                        "default_chat": True,
                        "visibility": {"allowed_roles": []},
                    },
                },
                "as_consumer": {
                    "default_agent": "claude",
                    "agents": {
                        "claude": {
                            "capabilities": {
                                "conversation": {
                                    "accepts_steer": True,
                                    "accepts_followup": True,
                                },
                            },
                        },
                    },
                },
            },
            "agent": {
                "command": "claude",
                "model": "claude-sonnet-4-6",
                "credential_ref": "b:agent.claude_code_key",
                "credential_env": "CLAUDE_CODE_KEY",
                "live_control": True,
                "timeout_seconds": 900,
                "allowed_tools": [
                    "Read",
                    "Grep",
                    "Glob",
                    "Edit",
                    "Write",
                    "Bash",
                    "WebSearch",
                    "WebFetch",
                ],
            },
        }
        return self._deep_merge_props(super().configuration_defaults(), defaults)

    async def post_run_hook(
        self,
        *,
        state: Dict[str, Any],
        result: Dict[str, Any],
        econ_ctx: Dict[str, Any] | None = None,
    ) -> None:
        await super().post_run_hook(
            state=state,
            result=result,
            econ_ctx=econ_ctx or {},
        )
        await persist_turn_artifacts(self, state, result)

    async def execute_core(
        self,
        *,
        state: Dict[str, Any],
        thread_id: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        return await run_claude_demo_turn(
            self,
            state=state,
            thread_id=thread_id,
            params=params,
        )
