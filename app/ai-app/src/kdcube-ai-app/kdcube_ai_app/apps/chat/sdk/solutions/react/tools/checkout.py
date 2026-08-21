# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict

from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace import (
    WorkspaceCheckoutError,
    artifact_outdir_for,
    checkout_workspace_items,
    resolve_context_workspace_source,
)
from kdcube_ai_app.apps.chat.sdk.solutions.react.tools.common import (
    add_block,
    notice_block,
    tc_result_path,
    tool_call_block,
)


TOOL_SPEC = {
    "id": "react.checkout",
    "purpose": (
        "Create or reset editable current-turn workspace state from durable object locators. "
        "Each item maps one materializable source to an exact target under git/projects or files. "
        "Use git/projects for a versioned working project and files for an individual derivative. "
        "replace makes the target exactly match the source; repeating it resets the target. "
        "overlay is available only for directories and retains destination-only entries. "
        "Sources are resolved and pinned by checkout itself; a separate react.pull call is not required."
    ),
    "args": {
        "items": (
            "ordered list[object], each with exactly: "
            "from (durable conv:fi or registered owner object ref), "
            "to (current-workspace-relative path under git/projects/... or files/...), and "
            "strategy (replace|overlay; overlay is directory-only). "
            "Do not include a conversation id or turn id in to."
        ),
    },
    "returns": (
        "JSON object {ok, turn_id, items, checked_out_from, editable_roots}. "
        "Each result item reports the pinned resolved_from ref and the new current-turn "
        "logical_path/physical_path. The whole batch is validated and prepared before any "
        "destination changes."
    ),
}


async def handle_react_checkout(
    *,
    ctx_browser: Any,
    state: Dict[str, Any],
    tool_call_id: str,
) -> Dict[str, Any]:
    last_decision = state.get("last_decision") or {}
    tool_call = last_decision.get("tool_call") or {}
    tool_id = "react.checkout"
    raw_params = tool_call.get("params") or {}
    params = raw_params if isinstance(raw_params, dict) else {}
    items = params.get("items")

    runtime_ctx = getattr(ctx_browser, "runtime_ctx", None)
    turn_id = str(getattr(runtime_ctx, "turn_id", "") or "").strip()
    conversation_id = str(
        getattr(runtime_ctx, "conversation_id", "") or ""
    ).strip()

    tool_call_block(
        ctx_browser=ctx_browser,
        tool_call_id=tool_call_id,
        tool_id=tool_id,
        payload={
            "tool_id": tool_id,
            "tool_call_id": tool_call_id,
            "params": tool_call.get("params") or {},
        },
    )

    def _fail(
        error_code: str,
        message: str,
        *,
        extra: Dict[str, Any] | None = None,
        retry: bool = True,
    ) -> Dict[str, Any]:
        payload = {
            "ok": False,
            "error": error_code,
            "message": message,
            **(extra or {}),
        }
        notice_block(
            ctx_browser=ctx_browser,
            tool_call_id=tool_call_id,
            code=error_code,
            message=message,
            extra={"tool_id": tool_id, **(extra or {})},
        )
        add_block(ctx_browser, {
            "turn": turn_id,
            "type": "react.tool.result",
            "call_id": tool_call_id,
            "mime": "application/json",
            "path": tc_result_path(turn_id=turn_id, call_id=tool_call_id),
            "text": json.dumps(payload, ensure_ascii=False, indent=2),
            "meta": {
                "tool_call_id": tool_call_id,
                "tool_id": tool_id,
            },
        })
        state["last_tool_result"] = payload
        if retry:
            state["retry_decision"] = True
        return state

    if not isinstance(items, list):
        return _fail(
            "protocol_violation.checkout_items_missing",
            "react.checkout requires params.items with {from, to, strategy} objects.",
        )

    outdir_raw = str(
        state.get("outdir") or getattr(runtime_ctx, "outdir", "") or ""
    ).strip()
    if not outdir_raw:
        return _fail(
            "react.checkout.workspace_unavailable",
            "The current turn workspace is unavailable because no output directory is bound.",
        )
    outdir = pathlib.Path(outdir_raw)

    async def _resolve_source(*, ref: str, staging_dir: pathlib.Path):
        return await resolve_context_workspace_source(
            ref=ref,
            staging_dir=staging_dir,
            ctx_browser=ctx_browser,
            outdir=outdir,
            state=state,
            tool_id=tool_id,
            tool_call_id=tool_call_id,
        )

    try:
        result = await checkout_workspace_items(
            items=items,
            artifact_root=artifact_outdir_for(outdir),
            current_turn_id=turn_id,
            tenant=str(getattr(runtime_ctx, "tenant", "") or ""),
            project=str(getattr(runtime_ctx, "project", "") or ""),
            user_id=str(getattr(runtime_ctx, "user_id", "") or ""),
            conversation_id=conversation_id,
            storage_path=str(getattr(runtime_ctx, "storage_path", "") or "") or None,
            source_resolver=_resolve_source,
        )
    except WorkspaceCheckoutError as error:
        return _fail(
            f"react.checkout.{error.code}",
            str(error),
            extra={"details": error.details} if error.details else None,
        )
    except Exception as error:
        return _fail(
            "react.checkout.failed",
            f"The requested checkout could not be completed: {error}",
        )

    add_block(ctx_browser, {
        "turn": turn_id,
        "type": "react.tool.result",
        "call_id": tool_call_id,
        "mime": "application/json",
        "path": tc_result_path(turn_id=turn_id, call_id=tool_call_id),
        "text": json.dumps(result, ensure_ascii=False, indent=2),
        "meta": {
            "tool_call_id": tool_call_id,
            "tool_id": tool_id,
        },
    })
    add_block(ctx_browser, {
        "turn": turn_id,
        "turn_id": turn_id,
        "type": "react.workspace.checkout",
        "mime": "application/json",
        "path": f"conv:ar:{turn_id}.react.workspace.checkout.{tool_call_id}",
        "text": json.dumps(result, ensure_ascii=False, indent=2),
        "meta": {
            "tool_call_id": tool_call_id,
            "tool_id": tool_id,
        },
    })
    state["last_tool_result"] = result
    return state
