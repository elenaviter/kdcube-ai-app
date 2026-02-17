# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

from __future__ import annotations

from typing import Any, Dict, List

import pathlib

import json
import hashlib

from kdcube_ai_app.apps.chat.sdk.solutions.react.v2.artifacts import (
    build_artifact_meta_block,
)
from kdcube_ai_app.apps.chat.sdk.solutions.react.v2.solution_workspace import (
    read_artifact_for_react,
)
from kdcube_ai_app.apps.chat.sdk.solutions.react.v2.tools.common import (
    tool_call_block,
    notice_block,
    add_block,
    tc_result_path,
)

TOOL_SPEC = {
    "id": "react.read",
    "purpose": (
        "Read artifacts or skills into the visible context so you can use them. "
        "Paths must be context paths (fi:/ar:/so:/sk:), not physical paths. "
        "Each path you read becomes visible in the timeline; skills are shown with ACTIVE 💡 banner."
    ),
    "args": {
        "paths": "list[str] context paths to read (files via fi:<turn_id>.files/<filepath>, sources via so:sources_pool[...], skills via sk:<skill_id or num>)",
    },
    "returns": "ok",
}


async def handle_react_read(*, ctx_browser: Any, state: Dict[str, Any], tool_call_id: str) -> Dict[str, Any]:
    last_decision = state.get("last_decision") or {}
    tool_call = last_decision.get("tool_call") or {}
    tool_id = "react.read"
    params = tool_call.get("params") or {}
    raw_paths = params.get("paths")
    if raw_paths is None and isinstance(params, list):
        raw_paths = params
    raw_paths = raw_paths if isinstance(raw_paths, list) else []
    paths = [str(p).strip() for p in raw_paths if str(p).strip()]

    turn_id = (ctx_browser.runtime_ctx.turn_id or "")
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

    skill_paths = [p for p in paths if p.startswith("sk:") or p.startswith("SK") or p.startswith("skill:") or p.startswith("skills.")]
    artifact_paths = [p for p in paths if p not in skill_paths]
    pending_blocks: List[Dict[str, Any]] = []
    missing_skills: List[str] = []
    exists_paths: List[str] = []
    def _normalize_block_for_hash(block: Dict[str, Any]) -> Dict[str, Any]:
        data = dict(block or {})
        data.pop("replacement_text", None)
        data.pop("ts", None)
        data.pop("turn", None)
        data.pop("turn_id", None)
        data.pop("author", None)
        data.pop("call_id", None)
        meta = data.get("meta")
        if isinstance(meta, dict):
            meta = dict(meta)
            for key in ("turn_id", "tool_call_id", "tool_id", "call_id", "ts", "started_at", "finished_at"):
                meta.pop(key, None)
            meta.pop("replacement_text", None)
            data["meta"] = meta
        data["hidden"] = bool(data.get("hidden", False))
        return data

    def _block_hash(block: Dict[str, Any]) -> str:
        try:
            normalized = _normalize_block_for_hash(block)
            payload = json.dumps(normalized, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            return hashlib.sha256(payload).hexdigest()
        except Exception:
            return ""

    def _block_exists_in_timeline(block: Dict[str, Any]) -> bool:
        path = (block.get("path") or "").strip()
        if not path:
            return False
        target_hash = _block_hash(block)
        if not target_hash:
            return False
        try:
            blocks = ctx_browser.timeline._collect_blocks()  # type: ignore[attr-defined]
        except Exception:
            blocks = []
        for existing in blocks:
            if not isinstance(existing, dict):
                continue
            if (existing.get("path") or "").strip() != path:
                continue
            if _block_hash(existing) == target_hash:
                return True
        return False

    def _maybe_add_block(block: Dict[str, Any]) -> bool:
        # Ensure read output is visible even if the source was hidden.
        block["hidden"] = False
        if isinstance(block.get("meta"), dict):
            block["meta"]["hidden"] = False
            block["meta"].pop("replacement_text", None)
        block.pop("replacement_text", None)
        path = (block.get("path") or "").strip()
        if path.startswith(("fi:", "so:", "sk:")) and _block_exists_in_timeline(block):
            return False
        pending_blocks.append(block)
        return True

    if skill_paths:
        try:
            from kdcube_ai_app.apps.chat.sdk.skills.skills_registry import (
                import_skillset,
                build_skill_short_id_map,
                build_skills_instruction_block,
            )
            short_map = build_skill_short_id_map(consumer="solver.react.decision.v2")
            normalized_skills: List[str] = []
            for raw in skill_paths:
                s = str(raw or "").strip()
                if not s:
                    continue
                if s.startswith("sk:"):
                    s = s[len("sk:"):].strip()
                if s.startswith("skill:"):
                    s = s[len("skill:"):].strip()
                if s.startswith("skills."):
                    s = s[len("skills."):].strip()
                if s.isdigit():
                    s = f"SK{s}"
                normalized_skills.append(s)
            skill_ids = import_skillset(normalized_skills, short_id_map=short_map)
            if normalized_skills:
                missing_skills = [s for s in normalized_skills if s not in skill_ids]
            loaded = state.setdefault("loaded_skills", set())
            if not isinstance(loaded, set):
                loaded = set(loaded)
                state["loaded_skills"] = loaded
            for sid in skill_ids:
                if not sid:
                    continue
                loaded.add(sid)
                skill_text = build_skills_instruction_block([sid])
                skill_block = {
                    "turn": turn_id,
                    "type": "react.tool.result",
                    "call_id": tool_call_id,
                    "mime": "text/markdown",
                    "path": f"sk:{sid}",
                    "text": f"ACTIVE 💡{skill_text}",
                    "meta": {
                        "tool_call_id": tool_call_id,
                    },
                }
                if not _maybe_add_block(skill_block):
                    exists_paths.append(f"sk:{sid}")
        except Exception:
            pass

    missing_artifacts: List[str] = []
    items: List[Dict[str, Any]] = []
    fi_paths = [p for p in artifact_paths if isinstance(p, str) and p.startswith("fi:")]
    other_paths = [p for p in artifact_paths if p not in fi_paths]
    try:
        items = ctx_browser.timeline_artifacts(
            paths=other_paths,
        )
    except Exception:
        items = []
    try:
        paths_seen = {item.get("context_path") for item in (items or []) if item.get("context_path")}
        if fi_paths:
            paths_seen.update(fi_paths)
        if paths_seen:
            ctx_browser.unhide_paths(paths=list(paths_seen))
    except Exception:
        pass
    total_tokens = 0
    per_path: List[Dict[str, Any]] = []
    items_by_path = {item.get("context_path"): item for item in (items or []) if item.get("context_path")}

    async def _emit_fi_path(ctx_path: str) -> None:
        outdir = pathlib.Path(state.get("outdir") or "")
        res = {}
        if outdir and outdir.exists():
            try:
                res = await read_artifact_for_react(
                    ctx_browser=ctx_browser,
                    path=ctx_path,
                    outdir=outdir,
                )
            except Exception:
                res = {"missing": True}
        else:
            res = {"missing": True}

        if res.get("missing"):
            missing_artifacts.append(ctx_path)
            per_path.append({"path": ctx_path, "missing": True})
            return

        artifact = res.get("artifact") or {}
        physical_path = res.get("physical_path") or ""
        # Emit the original metadata block text (digest) when available.
        digest_text = artifact.get("digest") if isinstance(artifact.get("digest"), str) else ""
        if not digest_text:
            meta_block = build_artifact_meta_block(
                turn_id=turn_id,
                tool_call_id=tool_call_id,
                artifact=artifact,
                artifact_path=ctx_path,
                physical_path=physical_path,
            )
            pending_blocks.append(meta_block)
        else:
            pending_blocks.append({
                "turn": turn_id,
                "type": "react.tool.result",
                "call_id": tool_call_id,
                "mime": "application/json",
                "path": tc_result_path(turn_id=turn_id, call_id=tool_call_id),
                "text": digest_text,
                "meta": {
                    "tool_call_id": tool_call_id,
                },
            })

        meta_extra = {"tool_call_id": tool_call_id, "turn_id": turn_id, "tool_id": tool_id}
        for key in ("hosted_uri", "rn", "key", "physical_path", "digest"):
            val = artifact.get(key)
            if val:
                meta_extra[key] = val
        if not meta_extra.get("physical_path") and artifact.get("local_path"):
            meta_extra["physical_path"] = artifact.get("local_path")

        art_text = res.get("text")
        art_base64 = res.get("base64")
        art_mime = (res.get("mime") or "").strip() or "application/octet-stream"
        tokens = 0

        added_any = False
        if isinstance(art_text, str) and art_text.strip():
            try:
                from kdcube_ai_app.apps.chat.sdk.util import token_count
                tokens = token_count(art_text)
                total_tokens += tokens
            except Exception:
                pass
            blk = {
                "turn": turn_id,
                "type": "react.tool.result",
                "call_id": tool_call_id,
                "mime": art_mime if art_mime else "text/markdown",
                "path": ctx_path or tc_result_path(turn_id=turn_id, call_id=tool_call_id),
                "text": art_text,
                "meta": meta_extra,
            }
            if _maybe_add_block(blk):
                added_any = True
        elif isinstance(art_base64, str) and art_base64:
            blk = {
                "turn": turn_id,
                "type": "react.tool.result",
                "call_id": tool_call_id,
                "mime": art_mime or "application/octet-stream",
                "path": ctx_path or tc_result_path(turn_id=turn_id, call_id=tool_call_id),
                "base64": art_base64,
                "meta": meta_extra,
            }
            if _maybe_add_block(blk):
                added_any = True

        per_path_entry = {"path": ctx_path}
        if not added_any:
            per_path_entry["status"] = "exists_in_visible_context"
            exists_paths.append(ctx_path)
        if tokens:
            per_path_entry["tokens"] = tokens
        per_path.append(per_path_entry)
    for raw_path in artifact_paths:
        if isinstance(raw_path, str) and raw_path.startswith("fi:"):
            await _emit_fi_path(raw_path)
            continue

        if isinstance(raw_path, str) and raw_path.startswith("so:"):
            selector = raw_path[len("so:"):]
            if selector.startswith("sources_pool["):
                try:
                    rows = ctx_browser.timeline.resolve_sources_pool(selector)
                except Exception:
                    rows = []
                if rows:
                    file_rows = []
                    other_rows = []
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        ap = (row.get("artifact_path") or "").strip()
                        source_type = (row.get("source_type") or "").strip().lower()
                        if source_type in {"file", "attachment"} or (ap.startswith("fi:")):
                            file_rows.append(row)
                        else:
                            other_rows.append(row)
                    for row in file_rows:
                        ap = (row.get("artifact_path") or "").strip()
                        if not ap:
                            physical_path = (row.get("physical_path") or row.get("local_path") or "").strip()
                            if physical_path.startswith("turn_") and "/files/" in physical_path:
                                tid, rel = physical_path.split("/files/", 1)
                                ap = f"fi:{tid}.files/{rel}"
                            elif physical_path.startswith("turn_") and "/attachments/" in physical_path:
                                tid, rel = physical_path.split("/attachments/", 1)
                                ap = f"fi:{tid}.user.attachments/{rel}"
                        if ap:
                            await _emit_fi_path(ap)
                        else:
                            missing_artifacts.append(raw_path)
                    if other_rows:
                        try:
                            from kdcube_ai_app.apps.chat.sdk.solutions.react.v2.layout import build_sources_pool_text
                            art_text = build_sources_pool_text(sources_pool=other_rows)
                            tokens = 0
                            try:
                                from kdcube_ai_app.apps.chat.sdk.util import token_count
                                tokens = token_count(art_text)
                                total_tokens += tokens
                            except Exception:
                                pass
                            blk = {
                                "turn": turn_id,
                                "type": "react.tool.result",
                                "call_id": tool_call_id,
                                "mime": "text/markdown",
                                "path": raw_path or tc_result_path(turn_id=turn_id, call_id=tool_call_id),
                                "text": art_text,
                                "meta": {
                                    "tool_call_id": tool_call_id,
                                },
                            }
                            if _maybe_add_block(blk):
                                per_path_entry = {"path": raw_path}
                                if tokens:
                                    per_path_entry["tokens"] = tokens
                                per_path.append(per_path_entry)
                            else:
                                exists_paths.append(raw_path)
                                per_path.append({"path": raw_path, "status": "exists_in_visible_context"})
                            continue
                        except Exception:
                            pass
                    if file_rows:
                        continue

        path = raw_path
        display_path = raw_path
        if path.startswith("so:"):
            path = path[len("so:"):]
            display_path = raw_path
        item = items_by_path.get(path)
        if not item:
            missing_artifacts.append(display_path or path)
            per_path.append({"path": display_path, "missing": True})
            continue

        art = item.get("artifact") or {}
        ctx_path = item.get("context_path") or path
        if display_path.startswith("so:"):
            ctx_path = display_path
        meta_block = build_artifact_meta_block(
            turn_id=turn_id,
            tool_call_id=tool_call_id,
            artifact={"tool_id": tool_id, "tool_call_id": tool_call_id, "value": {}},
            artifact_path=ctx_path,
            physical_path="",
        )
        pending_blocks.append(meta_block)

        tokens = 0
        art_text = art.get("text") if isinstance(art, dict) else None
        art_base64 = art.get("base64") if isinstance(art, dict) else None
        art_fmt = (art.get("format") or "text").lower() if isinstance(art, dict) else "text"
        art_mime = art.get("mime") if isinstance(art, dict) else None
        if (not isinstance(art_text, str) or not art_text.strip()) and path.startswith("sources_pool["):
            try:
                from kdcube_ai_app.apps.chat.sdk.solutions.react.v2.layout import build_sources_pool_text
                sources = ctx_browser.timeline.resolve_sources_pool(path)
                if sources:
                    art_text = build_sources_pool_text(sources_pool=sources)
                    art_fmt = "text"
            except Exception:
                pass
        if isinstance(art_text, str) and art_text.strip():
            try:
                from kdcube_ai_app.apps.chat.sdk.util import token_count
                tokens = token_count(art_text)
                total_tokens += tokens
            except Exception:
                pass
            if art_fmt in {"json"}:
                mime = "application/json"
            elif art_fmt in {"html"}:
                mime = "text/html"
            else:
                mime = "text/markdown"
            blk = {
                "turn": turn_id,
                "type": "react.tool.result",
                "call_id": tool_call_id,
                "mime": mime,
                "path": ctx_path or tc_result_path(turn_id=turn_id, call_id=tool_call_id),
                "text": art_text,
                "meta": {
                    "tool_call_id": tool_call_id,
                },
            }
            if not _maybe_add_block(blk):
                exists_paths.append(ctx_path)
        elif isinstance(art_base64, str) and art_base64:
            blk = {
                "turn": turn_id,
                "type": "react.tool.result",
                "call_id": tool_call_id,
                "mime": art_mime or "application/octet-stream",
                "path": ctx_path or tc_result_path(turn_id=turn_id, call_id=tool_call_id),
                "base64": art_base64,
                "meta": {
                    "tool_call_id": tool_call_id,
                },
            }
            if not _maybe_add_block(blk):
                exists_paths.append(ctx_path)
        per_path_entry = {"path": ctx_path}
        if tokens:
            per_path_entry["tokens"] = tokens
        per_path.append(per_path_entry)

    if artifact_paths or skill_paths:
        summary = {"paths": per_path, "total_tokens": total_tokens}
        if missing_artifacts:
            summary["missing"] = missing_artifacts
        if missing_skills:
            summary["missing_skills"] = missing_skills
        if exists_paths:
            summary["exists_in_visible_context"] = sorted(set(exists_paths))
        add_block(ctx_browser, {
            "turn": turn_id,
            "type": "react.tool.result",
            "call_id": tool_call_id,
            "mime": "application/json",
            "path": tc_result_path(turn_id=turn_id, call_id=tool_call_id),
            "text": json.dumps(summary, ensure_ascii=False),
            "meta": {
                "tool_call_id": tool_call_id,
            },
        })
    # Emit results after the status block.
    for blk in pending_blocks:
        add_block(ctx_browser, blk)
    if missing_artifacts or missing_skills:
        notice_block(
            ctx_browser=ctx_browser,
            tool_call_id=tool_call_id,
            code="read_paths_missing",
            message=f"react.read requested non-existent paths: {missing_artifacts}" if missing_artifacts else "react.read requested missing skills.",
            extra={"missing": missing_artifacts, "missing_skills": missing_skills, "tool_id": tool_id},
            rel="result",
        )
    state["last_tool_result"] = []
    return state
