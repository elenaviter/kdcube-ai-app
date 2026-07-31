# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

from __future__ import annotations

import json
import time
import uuid
import datetime as _dt
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import kdcube_ai_app.apps.chat.sdk.solutions.react.call as react_tools
from kdcube_ai_app.apps.chat.sdk.solutions.react.tools.common import add_block
from kdcube_ai_app.apps.chat.sdk.util import isoz


def _path_prefixes(value: Any) -> list[str]:
    items = value if isinstance(value, list) else [value]
    prefixes: set[str] = set()
    for item in items:
        path = ""
        if isinstance(item, dict):
            path = str(item.get("path") or "").strip()
        elif isinstance(item, str):
            path = item.strip()
        if not path:
            continue
        parts = path.split(":")
        if len(parts) >= 3 and parts[0] == "conv" and parts[1]:
            prefixes.add(f"conv:{parts[1]}:")
        elif ":" in path:
            prefixes.add(parts[0] + ":")
        else:
            prefixes.add("plain")
    return sorted(prefixes)


def _tool_params_summary(params: Any) -> Dict[str, Any]:
    if isinstance(params, dict):
        keys = sorted(str(key) for key in params.keys())
        summary: Dict[str, Any] = {
            "shape": "object",
            "keys": keys[:40],
            "key_count": len(keys),
            "redacted": True,
        }
        paths = params.get("paths")
        if isinstance(paths, list):
            summary["paths_count"] = len(paths)
            summary["path_prefixes"] = _path_prefixes(paths)
        items = params.get("items")
        if isinstance(items, list):
            summary["items_count"] = len(items)
            item_prefixes = _path_prefixes(items)
            if item_prefixes:
                summary["item_path_prefixes"] = item_prefixes
        for key in ("query", "prompt", "message", "text", "content", "code"):
            value = params.get(key)
            if isinstance(value, str):
                summary[f"{key}_len"] = len(value)
        for key in ("limit", "top_k", "max_hits", "max_results", "line_start", "line_count"):
            value = params.get(key)
            if isinstance(value, (int, float, bool)):
                summary[key] = value
        return summary
    if isinstance(params, list):
        return {
            "shape": "array",
            "items_count": len(params),
            "path_prefixes": _path_prefixes(params),
            "redacted": True,
        }
    if params is None:
        return {"shape": "none", "redacted": True}
    return {"shape": type(params).__name__, "redacted": True}


def _tool_result_status(result_state: Any) -> tuple[str, int, str]:
    if not isinstance(result_state, dict):
        return "completed", 0, ""
    status = "completed"
    error_count = 0
    error_code = ""
    if result_state.get("exit_reason") == "error" or result_state.get("error"):
        status = "error"
        err = result_state.get("error")
        if isinstance(err, dict):
            error_code = str(err.get("code") or err.get("error") or "").strip()
    last_tool_result = result_state.get("last_tool_result")
    if isinstance(last_tool_result, list):
        for item in last_tool_result:
            if not isinstance(item, dict) or not item.get("error"):
                continue
            error_count += 1
            if not error_code:
                err = item.get("error")
                if isinstance(err, dict):
                    error_code = str(err.get("code") or err.get("error") or "").strip()
                else:
                    error_code = str(err or "").strip()
    if error_count:
        status = "error"
    return status, error_count, error_code


def _step_suffix(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return uuid.uuid4().hex[:12]
    out = []
    for ch in raw[:96]:
        out.append(ch if ch.isalnum() or ch in {"_", "-", "."} else "_")
    return "".join(out) or uuid.uuid4().hex[:12]


def _json_preview(value: Any, *, max_text: int = 6000, max_items: int = 40, max_depth: int = 5) -> Tuple[Any, bool]:
    truncated = False

    def _walk(item: Any, depth: int) -> Any:
        nonlocal truncated
        if item is None or isinstance(item, (bool, int, float)):
            return item
        if isinstance(item, str):
            if len(item) > max_text:
                truncated = True
                return item[:max_text] + f"... [truncated {len(item) - max_text} chars]"
            return item
        if isinstance(item, (bytes, bytearray, memoryview)):
            truncated = True
            try:
                size = len(item)
            except Exception:
                size = 0
            return {"__bytes__": True, "size_bytes": size}
        # Depth limits containers, not their scalar leaves. Checking depth
        # before scalar handling erased useful fields at the boundary (for
        # example every title/ref in a named-service list result).
        if depth >= max_depth:
            truncated = True
            return {"__truncated__": True, "type": type(item).__name__}
        if isinstance(item, dict):
            out: Dict[str, Any] = {}
            for index, (key, val) in enumerate(item.items()):
                if index >= max_items:
                    truncated = True
                    out["__truncated_keys__"] = max(0, len(item) - max_items)
                    break
                out[str(key)] = _walk(val, depth + 1)
            return out
        if isinstance(item, (list, tuple, set)):
            seq = list(item)
            out = [_walk(val, depth + 1) for val in seq[:max_items]]
            if len(seq) > max_items:
                truncated = True
                out.append({"__truncated_items__": len(seq) - max_items})
            return out
        truncated = True
        return repr(item)

    return _walk(value, 0), truncated


def _tool_error_payload(result_state: Any = None, exception: Optional[BaseException] = None) -> Optional[Dict[str, Any]]:
    if exception is not None:
        return {
            "code": exception.__class__.__name__,
            "message": str(exception),
            "exception_type": exception.__class__.__name__,
        }
    if not isinstance(result_state, dict):
        return None
    err = result_state.get("error")
    if err:
        if isinstance(err, dict):
            preview, truncated = _json_preview(err, max_text=2000, max_items=20, max_depth=4)
            out = preview if isinstance(preview, dict) else {"message": str(preview)}
            if truncated:
                out["truncated"] = True
            return out
        return {"message": str(err)}
    last_tool_result = result_state.get("last_tool_result")
    if isinstance(last_tool_result, list):
        errors = []
        for item in last_tool_result:
            if isinstance(item, dict) and item.get("error"):
                errors.append(item.get("error"))
        if errors:
            preview, truncated = _json_preview(errors, max_text=2000, max_items=10, max_depth=4)
            return {
                "items": preview,
                **({"truncated": True} if truncated else {}),
            }
    return None


def _tool_result_payload(result_state: Any = None, exception: Optional[BaseException] = None) -> Dict[str, Any]:
    if exception is not None:
        return {
            "result": None,
            "result_truncated": False,
            "error": _tool_error_payload(exception=exception),
        }
    payload: Dict[str, Any] = {}
    if isinstance(result_state, dict):
        last_tool_result = result_state.get("last_tool_result")
        if last_tool_result is not None:
            result_preview, result_truncated = _json_preview(
                last_tool_result,
                max_depth=8,
            )
            payload["result"] = result_preview
            payload["result_truncated"] = bool(result_truncated)
        else:
            state_preview, state_truncated = _json_preview({
                key: result_state.get(key)
                for key in ("exit_reason", "error", "final_answer", "suggested_followups")
                if key in result_state
            })
            payload["result"] = state_preview
            payload["result_truncated"] = bool(state_truncated)
        error_payload = _tool_error_payload(result_state=result_state)
        if error_payload is not None:
            payload["error"] = error_payload
    else:
        result_preview, result_truncated = _json_preview(result_state)
        payload["result"] = result_preview
        payload["result_truncated"] = bool(result_truncated)
    return payload


def _tool_event_markdown(
    *,
    tool_id: str,
    tool_call_id: str,
    phase: str,
    status: str,
    data: Dict[str, Any],
) -> str:
    title = "Tool call generated" if phase == "call" else "Tool result"
    lines = [
        f"**{title}:** `{tool_id or 'unknown'}`",
        f"- call: `{tool_call_id}`",
        f"- status: `{status}`",
    ]
    if phase == "result":
        if data.get("error"):
            lines.append("- error:")
            try:
                err_text = json.dumps(data.get("error"), ensure_ascii=False, indent=2)
            except Exception:
                err_text = str(data.get("error"))
            lines.append(f"```json\n{err_text[:2000]}\n```")
        if "result" in data:
            try:
                result_text = json.dumps(data.get("result"), ensure_ascii=False, indent=2)
            except Exception:
                result_text = str(data.get("result"))
            if len(result_text) > 4000:
                result_text = result_text[:4000] + "\n... [truncated]"
            lines.append("- result:")
            lines.append(f"```json\n{result_text}\n```")
    return "\n".join(lines)


async def _emit_react_tool_event(
    *,
    react: Any,
    event_type: str,
    phase: str,
    tool_id: str,
    tool_call_id: str,
    params: Any,
    iteration: Optional[int],
    status: str,
    duration_ms: int,
    result_state: Any = None,
    exception: Optional[BaseException] = None,
) -> None:
    comm = getattr(react, "comm", None)
    service_event = getattr(comm, "service_event", None) if comm is not None else None
    if not callable(service_event):
        return
    result_status, error_count, error_code = _tool_result_status(result_state)
    if phase == "result" and status == "completed" and result_status == "error":
        status = "error"
    step = f"{event_type}.{_step_suffix(tool_call_id)}"
    data: Dict[str, Any] = {
        "tool_id": tool_id,
        "tool_call_id": tool_call_id,
        "tool_family": "react" if tool_id.startswith("react.") else "external",
        "phase": phase,
        "executed": phase == "result" and status != "rejected",
        "params": _tool_params_summary(params),
        "duration_ms": max(0, int(duration_ms)),
    }
    if iteration is not None:
        data["iteration"] = int(iteration)
    if error_count:
        data["error_count"] = error_count
    if error_code:
        data["error_code"] = error_code
    if exception is not None:
        data["exception_type"] = exception.__class__.__name__
    if phase == "result":
        data.update(_tool_result_payload(result_state=result_state, exception=exception))
    try:
        result = service_event(
            type=event_type,
            step=step,
            status=status,
            title="ReAct Tool Call" if phase == "call" else "ReAct Tool Result",
            agent="react.tool",
            data=data,
            markdown=_tool_event_markdown(
                tool_id=tool_id,
                tool_call_id=tool_call_id,
                phase=phase,
                status=status,
                data=data,
            ),
            auto_markdown=False,
        )
        if hasattr(result, "__await__"):
            await result
    except Exception:
        return


async def emit_react_tool_rejected_event(
    *,
    react: Any,
    tool_id: str = "",
    tool_call_id: str = "",
    params: Any = None,
    iteration: Optional[int] = None,
    code: str,
    message: str = "",
    index: Optional[int] = None,
    parent_tool_call_id: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    comm = getattr(react, "comm", None)
    service_event = getattr(comm, "service_event", None) if comm is not None else None
    if not callable(service_event):
        return
    call_id = tool_call_id or parent_tool_call_id or uuid.uuid4().hex[:12]
    data: Dict[str, Any] = {
        "tool_id": tool_id or "",
        "tool_call_id": call_id,
        "parent_tool_call_id": parent_tool_call_id or "",
        "tool_family": "react" if str(tool_id or "").startswith("react.") else "external",
        "phase": "rejected",
        "executed": False,
        "params": _tool_params_summary(params),
        "error": {
            "code": code,
            "message": message or code,
        },
    }
    if iteration is not None:
        data["iteration"] = int(iteration)
    if index is not None:
        data["action_index"] = int(index)
    if extra:
        data["extra"] = _json_preview(extra, max_text=2000, max_items=20, max_depth=4)[0]
    try:
        result = service_event(
            type="react.tool.rejected",
            step=f"react.tool.rejected.{_step_suffix(call_id)}{'.' + str(index) if index is not None else ''}",
            status="error",
            title="ReAct Tool Rejected",
            agent="react.tool",
            data=data,
            markdown=(
                f"**Tool action rejected:** `{tool_id or 'unknown'}`\n"
                f"- call: `{call_id}`\n"
                f"- code: `{code}`\n"
                f"- executed: `false`\n"
                f"- message: {message or code}"
            ),
            auto_markdown=False,
        )
        if hasattr(result, "__await__"):
            await result
    except Exception:
        return


@dataclass
class ReactRound:
    tool_id: str = ""
    tool_call_id: str = ""

    @classmethod
    def start(
        cls,
        *,
        ctx_browser: Any,
        tool_call_id: str,
        iteration: int,
    ) -> None:
        if not ctx_browser or not tool_call_id:
            return
        turn_id = (ctx_browser.runtime_ctx.turn_id or "")
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        add_block(ctx_browser, {
            "type": "react.round.start",
            "author": "react",
            "turn_id": turn_id,
            "ts": ts,
            "mime": "text/plain",
            "path": f"conv:ar:{turn_id}.react.round.start.{tool_call_id}" if turn_id else "",
            "text": "thinking",
            "meta": {
                "tool_call_id": tool_call_id,
                "iteration": iteration,
                "phase": "decision",
            },
            "call_id": tool_call_id,
        })

    @classmethod
    def thinking(
        cls,
        *,
        ctx_browser: Any,
        decision: Optional[Dict[str, Any]] = None,
        text: Optional[str] = None,
        title: str,
        iteration: int,
        tool_call_id: Optional[str] = None,
    ) -> None:
        if not ctx_browser:
            return
        thinking_info: Dict[str, Any] = {}
        if isinstance(decision, dict):
            channels = decision.get("channels") if isinstance(decision.get("channels"), dict) else {}
            thinking_info = channels.get("thinking") if isinstance(channels.get("thinking"), dict) else {}
            if text is None:
                text = thinking_info.get("text") or decision.get("internal_thinking")
        if not isinstance(text, str) or not text.strip():
            return
        def _to_iso(val: Any) -> str:
            if isinstance(val, (int, float)):
                ts_sec = val / 1000.0 if val > 1e12 else float(val)
                return _dt.datetime.fromtimestamp(ts_sec, tz=_dt.timezone.utc).isoformat().replace("+00:00", "Z")
            if isinstance(val, str):
                return isoz(val)
            return ""
        started_at = _to_iso(thinking_info.get("started_at"))
        finished_at = _to_iso(thinking_info.get("finished_at"))
        turn_id = (ctx_browser.runtime_ctx.turn_id or "")
        ts = started_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        meta: Dict[str, Any] = {
            "channel": "thinking",
            "title": title,
            "iteration": iteration,
        }
        call_id = str(tool_call_id or "").strip()
        if call_id:
            meta["tool_call_id"] = call_id
        if started_at:
            meta["started_at"] = started_at
        if finished_at:
            meta["finished_at"] = finished_at
        add_block(ctx_browser, {
            "type": "react.thinking",
            "author": "react",
            "turn_id": turn_id,
            "ts": ts,
            "mime": "text/markdown",
            "path": f"conv:ar:{turn_id}.react.thinking.{iteration}" if turn_id else "",
            "text": text.strip(),
            "meta": meta,
            "call_id": call_id,
        })

    @classmethod
    def note(
        cls,
        *,
        ctx_browser: Any,
        notes: str,
        tool_call_id: str,
        tool_id: str,
        action: str,
        iteration: int,
        ts: Optional[str] = None,
    ) -> None:
        if not ctx_browser or not isinstance(notes, str) or not notes.strip():
            return
        turn_id = (ctx_browser.runtime_ctx.turn_id or "")
        ts = str(ts or "").strip() or (_dt.datetime.utcnow().isoformat() + "Z")
        add_block(ctx_browser, {
            "type": "react.notes",
            "author": "react",
            "turn_id": turn_id,
            "ts": ts,
            "mime": "text/markdown",
            "path": f"conv:ar:{turn_id}.react.notes.{tool_call_id}" if turn_id else "",
            "text": notes.strip(),
            "meta": {
                "channel": "timeline_text",
                "tool_id": tool_id,
                "tool_call_id": tool_call_id,
                "action": action,
                "iteration": iteration,
            },
        })

    @classmethod
    def decision_raw(
        cls,
        *,
        ctx_browser: Any,
        decision: Optional[Dict[str, Any]] = None,
        iteration: int,
        reason: Optional[str] = None,
        tool_call_id: Optional[str] = None,
    ) -> None:
        if not ctx_browser or not isinstance(decision, dict):
            return
        raw_text = (decision.get("raw") or "").strip()
        if not raw_text:
            raw_text = ((decision.get("log") or {}).get("raw_data") or "").strip()
        if not raw_text:
            return
        if not reason:
            channels = decision.get("channels") if isinstance(decision.get("channels"), dict) else {}
            json_chan = channels.get("action") if isinstance(channels.get("action"), dict) else {}
            if not isinstance(json_chan, dict) or not (json_chan.get("text") or "").strip():
                reason = "missing_channel.action"
        turn_id = (ctx_browser.runtime_ctx.turn_id or "")
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        meta: Dict[str, Any] = {
            "channel": "raw",
            "iteration": iteration,
        }
        if reason:
            meta["reason"] = reason
        if tool_call_id:
            meta["tool_call_id"] = tool_call_id
        add_block(ctx_browser, {
            "type": "react.decision.raw",
            "author": "react",
            "turn_id": turn_id,
            "ts": ts,
            "mime": "application/json",
            "path": f"conv:ar:{turn_id}.react.decision.raw.{iteration}" if turn_id else "",
            "text": raw_text,
            "meta": meta,
            **({"call_id": tool_call_id} if tool_call_id else {}),
        })

    @classmethod
    async def execute(cls,
                      react,
                      state: Dict[str, Any]) -> Dict[str, Any]:
        decision = state.get("last_decision") or {}
        tool_call = decision.get("tool_call") or {}
        tool_id = (tool_call.get("tool_id") or "").strip()
        tool_call_id = state.pop("pending_tool_call_id", None) or tool_call.get("tool_call_id") or uuid.uuid4().hex[:12]
        if not tool_id:
            state["exit_reason"] = "error"
            state["error"] = {"where": "tool_execution", "error": "missing_tool_id", "managed": True}
            await emit_react_tool_rejected_event(
                react=react,
                tool_id="",
                tool_call_id=tool_call_id,
                params=tool_call.get("params") if isinstance(tool_call, dict) else None,
                iteration=state.get("pending_tool_origin_iteration"),
                code="missing_tool_id",
                message="tool_call.tool_id is missing for action=call_tool.",
            )
            return state
        ctx_browser = react.ctx_browser
        runtime_ctx = getattr(ctx_browser, "runtime_ctx", None)
        sentinel = object()
        previous_iteration = sentinel
        tool_iteration: Optional[int] = None
        if runtime_ctx is not None:
            previous_iteration = getattr(runtime_ctx, "_current_react_iteration", sentinel)
            try:
                raw_origin_iteration = state.get("pending_tool_origin_iteration")
                if raw_origin_iteration is None:
                    raw_state_iteration = int(state.get("iteration") or 0)
                    raw_origin_iteration = max(0, raw_state_iteration - 1)
                tool_iteration = int(raw_origin_iteration)
                setattr(runtime_ctx, "_current_react_iteration", tool_iteration)
            except Exception:
                pass

        async def _dispatch_tool_call() -> Dict[str, Any]:
            if tool_id == "react.read":
                return await react_tools.handle_react_read(react=react, ctx_browser=ctx_browser, state=state, tool_call_id=tool_call_id)
            if tool_id == "react.pull":
                return await react_tools.handle_react_pull(react=react, ctx_browser=ctx_browser, state=state, tool_call_id=tool_call_id)
            if tool_id == "react.checkout":
                return await react_tools.handle_react_checkout(ctx_browser=ctx_browser, state=state, tool_call_id=tool_call_id)
            if tool_id == "react.patch":
                return await react_tools.handle_react_patch(react=react, ctx_browser=ctx_browser, state=state, tool_call_id=tool_call_id)
            if tool_id == "react.memsearch":
                return await react_tools.handle_react_memsearch(ctx_browser=ctx_browser, state=state, tool_call_id=tool_call_id)
            if tool_id == "react.hide":
                return await react_tools.handle_react_hide(ctx_browser=ctx_browser, state=state, tool_call_id=tool_call_id)
            if tool_id == "react.rg":
                return await react_tools.handle_react_rg(ctx_browser=ctx_browser, state=state, tool_call_id=tool_call_id)
            if tool_id == "react.plan":
                return await react_tools.handle_react_plan(react=react, ctx_browser=ctx_browser, state=state, tool_call_id=tool_call_id)
            if tool_id == "react.write":
                return await react_tools.handle_react_write(react=react, ctx_browser=ctx_browser, state=state, tool_call_id=tool_call_id)
            if tool_id == "react.delegate":
                return await react_tools.handle_react_delegate(react=react, ctx_browser=ctx_browser, state=state, tool_call_id=tool_call_id)
            if tool_id == "react.contribute":
                return await react_tools.handle_react_contribute(react=react, ctx_browser=ctx_browser, state=state, tool_call_id=tool_call_id)
            return await react_tools.handle_external_tool(react=react, ctx_browser=ctx_browser, state=state, tool_call_id=tool_call_id)

        started_ms = int(time.time() * 1000)
        await _emit_react_tool_event(
            react=react,
            event_type="react.tool.call",
            phase="call",
            tool_id=tool_id,
            tool_call_id=tool_call_id,
            params=tool_call.get("params"),
            iteration=tool_iteration,
            status="started",
            duration_ms=0,
        )
        try:
            result_state = await _dispatch_tool_call()
        except Exception as exc:
            await _emit_react_tool_event(
                react=react,
                event_type="react.tool.result",
                phase="result",
                tool_id=tool_id,
                tool_call_id=tool_call_id,
                params=tool_call.get("params"),
                iteration=tool_iteration,
                status="error",
                duration_ms=int(time.time() * 1000) - started_ms,
                exception=exc,
            )
            raise
        else:
            status, _, _ = _tool_result_status(result_state)
            await _emit_react_tool_event(
                react=react,
                event_type="react.tool.result",
                phase="result",
                tool_id=tool_id,
                tool_call_id=tool_call_id,
                params=tool_call.get("params"),
                iteration=tool_iteration,
                status=status,
                duration_ms=int(time.time() * 1000) - started_ms,
                result_state=result_state,
            )
            return result_state
        finally:
            if runtime_ctx is not None:
                try:
                    if previous_iteration is sentinel:
                        delattr(runtime_ctx, "_current_react_iteration")
                    else:
                        setattr(runtime_ctx, "_current_react_iteration", previous_iteration)
                except Exception:
                    pass


ToolCallView = ReactRound
