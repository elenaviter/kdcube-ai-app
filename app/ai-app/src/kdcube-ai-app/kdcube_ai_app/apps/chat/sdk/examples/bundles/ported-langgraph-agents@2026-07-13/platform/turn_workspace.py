# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
#
# ── turn_workspace.py ── the turn's distributed workspace, model-facing ──
#
# KDCube gives every turn a DISTRIBUTED WORKSPACE — per-turn `work/` + `out/`
# directories on the shared exec-workspace volume (the same concept the React
# agent stands on: `get_exec_workspace_root`, `OUTDIR_CV`/`WORKDIR_CV`, hosting
# through `ApplicationHostingService`). The workspace follows ONE rule with no
# exceptions: it starts EMPTY every turn, and a file enters it only through an
# EXPLICIT PULL — including files arriving with the current message. The agent
# pulls what it needs by conversation link (`conv:fi:`) or an event's separate
# owner `object_ref`, reads/processes it with code execution, and everything
# its code writes is hosted back into the conversation. Durable sources can be
# checked out directly into editable `git/projects/...` or `files/...` state.
#
# The model learns all of this IN-BAND, the way React's timeline frames its
# input (`turn.header` / `user.prompt` blocks): each turn's text is framed as
#
#     [Turn start turn_<id>]      <- the boundary + the empty-workspace rule
#     [User message]              <- the user's words, verbatim
#     [Files arriving this turn]  <- METADATA ONLY: filename, mime, size, LINK
#
# NOTHING is read for the model automatically — not text, not images. The
# frame carries metadata + the link; the model decides: `read_file` to view a
# file (text or visual, exactly like react.read), `pull_files` + `run_python`
# to process it, or `checkout` before editing it. These are LangGraph bindings
# over the same harness workspace operations used by the native ReAct Agent. Without
# the in-band frame the model trusts stale history ("I pulled that file
# before, it is still here").
#
# This module is the bundle's model-facing door to that workspace:
#
#   * `prepare_turn_workspace` — account for EVERY file arriving this turn:
#     metadata as received (filename, mime, size) + its durable conversation
#     link, and — when the workspace is not available — the honest reason its
#     contents cannot be examined. A file is never silently dropped.
#   * `frame_turn_input` — the framed turn text above.
#   * `build_read_file_tool` — the view door: one conversation file into
#     visible context by its link (text bounded; images/PDF as visual
#     payloads, downscaled under a byte cap — react.read semantics).
#   * `build_pull_files_tool` — the readonly materialize door for a `conv:fi:`
#     link or authorized owner locator. Byte resolution uses the shared harness
#     source resolver and collision-safe workspace materializer.
#   * `build_checkout_tool` — direct editable import/reset into an explicit
#     current-turn `git/projects/...` or `files/...` target.
#
# Fail-open per file, never per turn: a link that cannot be pulled is reported
# by the pull result, and the turn proceeds.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping

from langchain_core.tools import tool

from kdcube_ai_app.apps.chat.sdk.protocol import hosted_external_event_attachments

LOGGER = logging.getLogger("kdcube.ported_langgraph_agents.turn_workspace")


def _human_size(size: int) -> str:
    value = float(max(0, int(size or 0)))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{int(value)} B"


@dataclass
class TurnFile:
    """One arriving file's account this turn — the model hears exactly this."""
    filename: str
    mime: str
    size: int
    # The file's durable conversation link (`conv:fi:<turn>.user.attachments/<name>`,
    # the same shape the turn recorder / Files tab carry) — THE pull handle, this
    # turn and in every later one. It rides the turn frame into checkpointed history.
    ref: str = ""
    # Set when the workspace is unavailable: why the contents cannot be examined.
    reason: str = ""


@dataclass
class TurnObject:
    """One accepted non-file event and the object locator it carries."""

    event_type: str
    event_ref: str
    object_ref: str = ""
    label: str = ""
    summary: str = ""


@dataclass
class TurnWorkspace:
    """This turn's workspace state and the files that arrived with the turn."""
    live: bool
    files: List[TurnFile] = field(default_factory=list)
    objects: List[TurnObject] = field(default_factory=list)
    # The current turn's id — stamped into the turn frame so the model anchors
    # "now" against the turn segments inside conv:fi: links (its history spans
    # many turns; the working directory belongs to exactly this one).
    turn_id: str = ""


def _event_mappings(event: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    out: List[Mapping[str, Any]] = [event]
    payload = event.get("payload")
    if isinstance(payload, Mapping):
        out.append(payload)
        nested = payload.get("event")
        if isinstance(nested, Mapping):
            out.append(nested)
    data = event.get("data")
    if isinstance(data, Mapping):
        out.append(data)
    return out


def _first_text(rows: List[Mapping[str, Any]], *keys: str) -> str:
    for row in rows:
        for key in keys:
            value = str(row.get(key) or "").strip()
            if value:
                return value
    return ""


def _turn_object(
    event: Any,
    *,
    turn_id: str,
    conversation_id: str,
    occurrence_index: int,
) -> TurnObject | None:
    if not isinstance(event, Mapping):
        return None
    event_type = str(event.get("type") or "").strip()
    if not event_type.startswith("event."):
        return None
    if event_type.startswith((
        "event.user.prompt",
        "event.user.followup",
        "event.user.steer",
        "event.user.attachment",
    )):
        return None

    rows = _event_mappings(event)
    event_id = _first_text(rows, "event_id", "id", "message_id") or f"event_{occurrence_index + 1}"
    logical_path = _first_text(rows, "logical_path", "logicalPath", "path")
    if not logical_path.startswith("conv:ev:"):
        logical_path = ""
    from kdcube_ai_app.apps.chat.sdk.runtime.harness.timeline import (
        event_record_ref,
        object_ref_from_event,
    )

    event_ref = event_record_ref(
        turn_id=turn_id,
        event_id=event_id,
        conversation_id=conversation_id,
        logical_path=logical_path,
    )
    object_ref = object_ref_from_event(event)
    return TurnObject(
        event_type=event_type,
        event_ref=event_ref,
        object_ref=object_ref,
        label=_first_text(rows, "label", "title", "name"),
        summary=_first_text(rows, "summary", "description"),
    )


async def prepare_turn_workspace(
    ctx: Any,
    events: Any,
    *,
    exec_tool_bound: bool,
) -> TurnWorkspace:
    """Account for every hosted attachment arriving this turn.

    No bytes move here — the workspace starts empty and STAYS empty until the
    model pulls (one rule, no current-turn exception). ``ctx`` is the code-exec
    context (`build_code_exec_context`); `exec_tool_bound` reflects whether
    `run_python` is actually bound this turn (admin-declared and not
    user-disabled) — with no workspace tools the files are honestly reported
    as not examinable this turn."""
    hosted = hosted_external_event_attachments(events or [])
    live = bool(ctx is not None and getattr(ctx, "enabled", False) and exec_tool_bound)
    workspace = TurnWorkspace(live=live, turn_id=str(getattr(ctx, "turn_id", "") or "").strip())
    conversation_id = str(getattr(ctx, "conversation_id", "") or "").strip()

    for event_index, event in enumerate(events or []):
        item = _turn_object(
            event,
            turn_id=workspace.turn_id,
            conversation_id=conversation_id,
            occurrence_index=event_index,
        )
        if item is not None:
            workspace.objects.append(item)

    for item in hosted:
        if not isinstance(item, dict):
            continue
        mime = str(item.get("mime") or "application/octet-stream").strip().lower()
        raw_filename = str(item.get("filename") or "").strip()
        entry = TurnFile(
            filename=raw_filename or "attachment",
            mime=mime,
            size=int(item.get("size") or 0),
            ref=(
                f"conv:fi:{workspace.turn_id}.user.attachments/{raw_filename}"
                if workspace.turn_id and raw_filename
                else ""
            ),
        )
        if not live:
            entry.reason = "no workspace tools are active this turn (code execution is not enabled)"
        workspace.files.append(entry)
        LOGGER.info(
            "[ported-langgraph] turn workspace: arriving file %s (%s, %d bytes) ref=%s live=%s",
            entry.filename, entry.mime, entry.size, entry.ref or "-", live,
        )
    return workspace


def frame_turn_input(question: str, workspace: TurnWorkspace) -> str:
    """The model's turn text, framed like React frames its timeline input:
    turn-start header (boundary + the empty-workspace rule), the user's
    message verbatim, and the arriving-files block with a pull link per file.

    Emitted every turn the workspace is live (files or not) — the boundary
    must be in-band. Without a workspace and without files the user's text
    passes through unframed (there is nothing to explain)."""
    question = question or ""
    if not workspace.live and not workspace.files and not workspace.objects:
        return question
    header = f"[Turn start {workspace.turn_id}]" if workspace.turn_id else "[Turn start]"
    parts: List[str] = []
    if workspace.live:
        parts.append(
            header + "\n"
            "Your working directory is EMPTY — it starts fresh every turn. Files are "
            "given to you as LINKS only; nothing is read for you automatically. To VIEW "
            "a file, call read_file with its conversation link. To PROCESS it with code, "
            "call pull_files with its object ref and use the exact OUTPUT_DIR-relative "
            "path it reports. To modify or reset durable source data, call checkout with "
            "an explicit editable target under files/ or git/projects/."
        )
    else:
        parts.append(header)
    parts.append("[User message]\n" + question)
    if workspace.files:
        lines = ["[Files arriving this turn]"]
        for f in workspace.files:
            head = f"- {f.filename} ({f.mime}, {_human_size(f.size)})"
            if workspace.live and f.ref:
                lines.append(f"{head} — link: {f.ref}")
            elif f.reason:
                lines.append(
                    f"{head} — received and stored with the conversation, but {f.reason}; its "
                    f"contents are not available to you right now — tell the user plainly when "
                    f"it matters." + (f" Conversation link: {f.ref}" if f.ref else "")
                )
            else:
                lines.append(f"{head} — link: {f.ref}" if f.ref else f"{head} — received.")
        parts.append("\n".join(lines))
    if workspace.objects:
        lines = ["[Objects and events arriving this turn]"]
        for item in workspace.objects:
            title = item.label or item.event_type
            lines.append(f"- {title} ({item.event_type})")
            if item.summary:
                lines.append(f"  summary: {item.summary}")
            if item.event_ref:
                lines.append(f"  event_ref: {item.event_ref}")
            if item.object_ref:
                lines.append(f"  object_ref: {item.object_ref}")
                lines.append(
                    "  use object_ref with pull_files for read-only local bytes, "
                    "or checkout when an editable derivative is needed"
                )
            else:
                lines.append("  no materializable object_ref was supplied")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def build_pull_files_tool() -> Any:
    """Return the `pull_files` LangChain tool (a fresh object per call, so each
    agent binds its own instance). Bound beside `run_python` — it feeds the
    same workspace working directory the code reads."""

    @tool
    async def pull_files(refs: List[str]) -> str:
        """Materialize object refs into this turn's read-only workspace view.

        Pass a durable ``conv:fi:`` file link or an owner-provided ``object_ref``
        exactly as shown in the turn frame. ``conv:ev:`` identifies an event
        record and cannot be pulled; read the event and pass its ``object_ref``.
        The result reports a collision-safe path relative to ``OUTPUT_DIR``.
        Pulled bytes are read-only; use ``checkout`` before modifying them.

        The working directory starts EMPTY every turn — nothing from earlier
        turns is in it until you pull it again.
        """
        from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace import (
            artifact_outdir_for,
            pull_refs_into_workspace,
            resolve_context_workspace_source,
        )
        from .code_exec import current_code_exec_context

        ctx = current_code_exec_context()
        if ctx is None or not ctx.enabled:
            return "The code workspace is not available for this turn (code execution disabled or offline)."
        if ctx.outdir is None:
            return "The code workspace is not available for this turn (no working directory)."

        async def _source_resolver(*, ref: str, staging_dir: Any) -> Any:
            if ctx.ctx_browser is None:
                return None
            return await resolve_context_workspace_source(
                ref=ref,
                staging_dir=staging_dir,
                ctx_browser=ctx.ctx_browser,
                outdir=ctx.outdir,
                state=ctx.state,
                tool_id="workspace.pull",
            )

        LOGGER.info("[ported-langgraph] pull_files: %d ref(s) requested", len(refs or []))
        reports = await pull_refs_into_workspace(
            refs=list(refs or []),
            artifact_root=artifact_outdir_for(ctx.outdir),
            tenant=ctx.tenant,
            project=ctx.project,
            user_id=ctx.user_id,
            conversation_id=ctx.conversation_id,
            source_resolver=_source_resolver,
        )
        if not reports:
            return "No refs were pulled — pass one or more object refs exactly as shown in the conversation."
        lines: List[str] = []
        for report in reports:
            if report.get("ok"):
                lines.append(
                    f"- pulled {report['ref']} as read-only "
                    f"OUTPUT_DIR/{report['physical_path']} "
                    f"({report.get('mime')}, {_human_size(report.get('size') or 0)})"
                )
            else:
                lines.append(f"- FAILED {report.get('ref')}: {report.get('error')}")
        ok_count = sum(1 for r in reports if r.get("ok"))
        LOGGER.info("[ported-langgraph] pull_files: %d/%d ref(s) materialized", ok_count, len(reports))
        return "\n".join([f"Pulled {ok_count}/{len(reports)} object(s) into the read-only workspace view:"] + lines)

    return pull_files


def build_checkout_tool() -> Any:
    """Return the shared workspace checkout contract as a LangChain tool."""

    @tool
    async def checkout(items: List[Dict[str, str]]) -> str:
        """Create or reset editable workspace state from durable object refs.

        Each item has exactly ``from``, ``to``, and ``strategy``. ``from`` is a
        materializable object ref. ``to`` is a current-workspace-relative path
        below ``files/`` or ``git/projects/``; do not include conversation or
        turn ids. ``replace`` works for a file or directory and resets the
        target exactly. ``overlay`` is directory-only and keeps unrelated
        target files. The whole batch is validated before any target changes.
        """
        from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace import (
            artifact_outdir_for,
            checkout_workspace_items,
            resolve_context_workspace_source,
        )
        from .code_exec import current_code_exec_context

        ctx = current_code_exec_context()
        if ctx is None or not ctx.enabled:
            return "The code workspace is not available for this turn (code execution disabled or offline)."
        if ctx.outdir is None:
            return "The code workspace is not available for this turn (no working directory)."

        async def _source_resolver(*, ref: str, staging_dir: Any) -> Any:
            if ctx.ctx_browser is None:
                return None
            return await resolve_context_workspace_source(
                ref=ref,
                staging_dir=staging_dir,
                ctx_browser=ctx.ctx_browser,
                outdir=ctx.outdir,
                state=ctx.state,
                tool_id="workspace.checkout",
            )

        try:
            result = await checkout_workspace_items(
                items=list(items or []),
                artifact_root=artifact_outdir_for(ctx.outdir),
                current_turn_id=ctx.turn_id,
                tenant=ctx.tenant,
                project=ctx.project,
                user_id=ctx.user_id,
                conversation_id=ctx.conversation_id,
                source_resolver=_source_resolver,
            )
        except Exception as error:
            LOGGER.warning("[ported-langgraph] checkout failed", exc_info=True)
            details = getattr(error, "details", None)
            suffix = f" Details: {details}" if details else ""
            return f"Checkout failed: {error}.{suffix}"

        rows = [
            f"- {row['from']} -> OUTPUT_DIR/{row['physical_path']} "
            f"({row['strategy']}, editable, link={row['logical_path']})"
            for row in result.get("items", [])
        ]
        return "\n".join([f"Checked out {len(rows)} item(s):"] + rows)

    return checkout


# react.read-mirroring caps: bounded text view; images/PDF ride as visual
# payloads only under a byte cap (oversized images are downscaled first).
_READ_TEXT_CAP = 60_000
_READ_BLOB_CAP = 4 * 1024 * 1024

_TEXTUAL_MIME_PREFIXES = ("text/",)
_TEXTUAL_MIME_EXACT = {
    "application/json", "application/xml", "application/x-yaml",
    "application/yaml", "application/csv", "application/x-ndjson",
    "application/javascript", "application/sql",
}


def _is_textual_mime(mime: str) -> bool:
    mime = (mime or "").strip().lower()
    return mime.startswith(_TEXTUAL_MIME_PREFIXES) or mime in _TEXTUAL_MIME_EXACT


def build_read_file_tool() -> Any:
    """Return the `read_file` LangChain tool (fresh per call). The VIEW door of
    the workspace triad, mirroring react.read: text in, visuals in, binaries
    routed to pull+exec."""

    @tool
    async def read_file(path: str, max_text_symbols: int = 0) -> Any:
        """Read ONE conversation file into your visible context by its conv:fi: link.

        Text files return their text (bounded). Images and PDFs are returned to
        you as visual content (oversized images are downscaled). Other binary
        files are not viewable this way — use pull_files + run_python to
        process them. Links appear in `[Files arriving this turn]` and in
        run_python reports (``link=conv:fi:...``); pass the link exactly as
        shown. `max_text_symbols` optionally lowers the text bound.
        """
        from kdcube_ai_app.apps.chat.sdk.runtime.harness.events.resolver import read_event_ref_bytes
        from .attachments import _DOC_MIME, _IMAGE_MIME
        from .code_exec import current_code_exec_context

        ctx = current_code_exec_context()
        if ctx is None or not ctx.enabled:
            return "The workspace tools are not available for this turn (code execution disabled or offline)."
        ref = str(path or "").strip()
        if not ref:
            return "Pass one conv:fi: link exactly as shown in the conversation."
        try:
            data, meta = await read_event_ref_bytes(
                ref=ref,
                tenant=ctx.tenant,
                project=ctx.project,
                user_id=ctx.user_id,
                conversation_id=ctx.conversation_id,
            )
        except Exception as error:
            LOGGER.warning("[ported-langgraph] read_file failed ref=%r", ref, exc_info=True)
            return f"Could not read {ref}: {error}"

        import mimetypes as _mt
        from pathlib import PurePosixPath as _P

        filename = _P(str(meta.get("relpath") or ref)).name or "file"
        mime = (_mt.guess_type(filename)[0] or "application/octet-stream").lower()
        head = f"read {filename} ({mime}, {_human_size(len(data))}) from {ref}"
        LOGGER.info("[ported-langgraph] read_file: %s", head)

        if _is_textual_mime(mime):
            cap = max(1, int(max_text_symbols)) if max_text_symbols else _READ_TEXT_CAP
            text = data.decode("utf-8", errors="replace")
            clipped = text[:cap] + ("\n...[truncated]" if len(text) > cap else "")
            return f"[{head}]\n{clipped}"

        if mime in _IMAGE_MIME:
            import base64 as _b64
            from kdcube_ai_app.infra.service_hub.multimodality import normalize_image_base64_for_model

            b64 = _b64.b64encode(data).decode("ascii")
            try:
                normalized = normalize_image_base64_for_model(b64, media_type=mime)
                b64 = normalized.get("base64") or b64
            except Exception:
                pass
            if len(b64) > _READ_BLOB_CAP:
                return (
                    f"[{head}] The image is too large for visible context — "
                    f"pull_files it and process it with run_python instead."
                )
            return [
                {"type": "text", "text": f"[{head}]"},
                {"type": "image", "data": b64, "media_type": mime},
            ]

        if mime in _DOC_MIME:
            import base64 as _b64

            if len(data) > _READ_BLOB_CAP:
                return (
                    f"[{head}] The PDF is too large for visible context — "
                    f"pull_files it and process it with run_python instead."
                )
            return [
                {"type": "text", "text": f"[{head}]"},
                {"type": "document", "data": _b64.b64encode(data).decode("ascii"), "media_type": mime},
            ]

        return (
            f"[{head}] This is a binary file and is not viewable directly — "
            f"pull_files it and examine it with run_python."
        )

    return read_file
