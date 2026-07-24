# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

from __future__ import annotations

from typing import Any

from kdcube_ai_app.apps.chat.sdk.solutions.react.tools.write_path_policy import (
    WRITE_PATH_ALREADY_EXISTS,
    duplicate_write_message,
    react_write_target_is_visible,
    resolve_react_write_target,
)
from kdcube_ai_app.apps.chat.sdk.streaming.stream_policy import StreamPolicyViolation


class ReactWriteStreamGuard:
    """Interrupt a duplicate current-turn react.write before content streams."""

    def __init__(
        self,
        *,
        ctx_browser: Any,
        action_gate: Any,
        answer_gate: Any,
        action_index: int,
    ) -> None:
        self._ctx_browser = ctx_browser
        self._action_gate = action_gate
        self._answer_gate = answer_gate
        self._action_index = int(action_index or 0)

    async def observe_path(self, path: str) -> None:
        runtime_ctx = getattr(self._ctx_browser, "runtime_ctx", None)
        turn_id = str(getattr(runtime_ctx, "turn_id", "") or "").strip()
        conversation_id = str(getattr(runtime_ctx, "conversation_id", "") or "").strip()
        target = resolve_react_write_target(
            path=path,
            turn_id=turn_id,
            conversation_id=conversation_id,
        )
        if target is None or not react_write_target_is_visible(
            ctx_browser=self._ctx_browser,
            target=target,
        ):
            return

        await self._action_gate.deny()
        await self._answer_gate.deny()
        raise StreamPolicyViolation(
            code=WRITE_PATH_ALREADY_EXISTS,
            message=duplicate_write_message(target),
            extra={"index": self._action_index, **target.violation_extra()},
        )


__all__ = ["ReactWriteStreamGuard"]
