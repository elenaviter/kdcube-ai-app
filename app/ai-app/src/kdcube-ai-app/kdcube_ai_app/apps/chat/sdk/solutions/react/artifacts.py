# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

from __future__ import annotations

import json
import pathlib
import base64
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from kdcube_ai_app.apps.chat.sdk.util import _truncate, token_count
import kdcube_ai_app.apps.chat.sdk.tools.tools_insights as tools_insights
from kdcube_ai_app.apps.chat.sdk.solutions.react.tools.common import tc_result_path
from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.artifacts import (
    WorkspaceArtifact,
)
from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.references import (
    ARTIFACT_NAMESPACE_FILES,
    build_logical_artifact_path,
    build_physical_artifact_path,
    infer_artifact_namespace,
    normalize_physical_path,
    split_physical_artifact_path,
)


def detect_edit(*, timeline: Any, artifact_path: str, tool_call_id: str) -> bool:
    if not timeline or not artifact_path:
        return False
    try:
        existing = timeline.resolve_artifact(artifact_path)
        if not isinstance(existing, dict):
            return False
        prev_call_id = existing.get("tool_call_id")
        if prev_call_id and prev_call_id != tool_call_id:
            return True
        return True
    except Exception:
        return False


def build_artifact_meta_block(
    *,
    turn_id: str,
    tool_call_id: str,
    artifact: Dict[str, Any],
    artifact_path: str,
    physical_path: str,
    edited: bool = False,
    tokens: Optional[int] = None,
) -> Dict[str, Any]:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta_json = {
        "artifact_path": artifact_path,
        "physical_path": physical_path,
        "mime": (artifact.get("value") or {}).get("mime") or artifact.get("mime"),
        "kind": artifact.get("artifact_kind") or artifact.get("kind"),
        "visibility": artifact.get("visibility"),
        "channel": artifact.get("channel"),
        "tool_call_id": tool_call_id,
        "edited": bool(edited),
        "ts": ts,
    }
    size_bytes = (artifact.get("value") or {}).get("size_bytes") or artifact.get("size_bytes")
    if size_bytes is not None:
        meta_json["size_bytes"] = size_bytes
    # Delivery fingerprint stays in block metadata, outside the model-facing
    # artifact digest stored in this block's text.
    content_sha256 = (artifact.get("value") or {}).get("content_sha256") or artifact.get("content_sha256")
    text_symbols = (artifact.get("value") or {}).get("text_symbols") or artifact.get("text_symbols")
    if text_symbols is not None:
        meta_json["text_symbols"] = text_symbols
    line_count = (artifact.get("value") or {}).get("line_count") or artifact.get("line_count")
    if line_count is not None:
        meta_json["line_count"] = line_count
    description = (artifact.get("value") or {}).get("description") or artifact.get("description")
    if description:
        meta_json["description"] = description
    write_warning = (artifact.get("value") or {}).get("write_warning")
    if write_warning:
        meta_json["write_warning"] = write_warning
    sources_used = artifact.get("sources_used") or (artifact.get("value") or {}).get("sources_used")
    if sources_used:
        meta_json["sources_used"] = sources_used
    if artifact.get("error"):
        meta_json["status"] = "error"
        meta_json["error"] = artifact.get("error")
    if tokens is not None:
        try:
            meta_json["tokens"] = int(tokens)
        except Exception:
            meta_json["tokens"] = tokens
    value = artifact.get("value") if isinstance(artifact.get("value"), dict) else {}
    for key in ("hosted_uri", "key", "rn"):
        val = value.get(key) or artifact.get(key)
        if val:
            meta_json[key] = val
    # Drop empty or None attributes to avoid confusing metadata.
    meta_json = {
        k: v
        for k, v in meta_json.items()
        if v is not None and (not isinstance(v, str) or v.strip() != "")
    }
    block_meta = {
        "tool_call_id": tool_call_id,
    }
    if content_sha256:
        block_meta["content_sha256"] = str(content_sha256).strip()
    return {
        "turn": turn_id,
        "type": "react.tool.result",
        "call_id": tool_call_id,
        "mime": "application/json",
        "path": tc_result_path(turn_id=turn_id, call_id=tool_call_id),
        "text": json.dumps(meta_json, ensure_ascii=False, indent=2),
        "ts": ts,
        "meta": block_meta,
    }


def artifact_transport_metadata(meta_block: Dict[str, Any]) -> Dict[str, Any]:
    """Return non-model-facing metadata that must follow an artifact block."""
    meta = meta_block.get("meta") if isinstance(meta_block, dict) else None
    if not isinstance(meta, dict):
        return {}
    content_sha256 = str(meta.get("content_sha256") or "").strip()
    return {"content_sha256": content_sha256} if content_sha256 else {}


def build_artifact_binary_block(
    *,
    turn_id: str,
    tool_call_id: str,
    artifact_path: str,
    abs_path: pathlib.Path,
    mime: str,
    meta_extra: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    try:
        if not abs_path.exists() or not abs_path.is_file():
            return None
        data = abs_path.read_bytes()
        b64 = base64.b64encode(data).decode("utf-8")
    except Exception:
        return None
    meta = {
        "artifact_path": artifact_path,
        "tool_call_id": tool_call_id,
    }
    if isinstance(meta_extra, dict):
        for k, v in meta_extra.items():
            if v is not None:
                meta[k] = v
    return {
        "turn": turn_id,
        "type": "react.tool.result",
        "call_id": tool_call_id,
        "mime": mime,
        "path": artifact_path,
        "base64": b64,
        "meta": meta,
    }


def error_block_details(err: Any) -> Optional[Dict[str, Any]]:
    """Extract the `details` payload for a tool-result error block from an error
    envelope, WITHOUT duplicating the fields already carried at the top level
    (`code`/`message`). If the envelope has a real nested `details` (e.g. a code
    exit with `stderr_tail`), that is used verbatim; otherwise the remaining
    non-redundant keys (`where`, `description`, `retryable`, …) are returned."""
    if not isinstance(err, dict):
        return None
    nested = err.get("details")
    if isinstance(nested, dict) and nested:
        return nested
    extras = {
        k: v for k, v in err.items()
        if k not in ("code", "message", "error", "managed", "details")
    }
    return extras or None


def build_tool_result_error_block(
    *,
    turn_id: str,
    tool_call_id: str,
    code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    payload = {
        "tool_call_id": tool_call_id,
        "status": "error",
        "error": {
            "code": code,
            "message": message,
            **({"details": details} if details else {}),
        },
        "ts": ts,
    }
    return {
        "turn": turn_id,
        "type": "react.tool.result",
        "call_id": tool_call_id,
        "mime": "application/json",
        "path": tc_result_path(turn_id=turn_id, call_id=tool_call_id),
        "text": json.dumps(payload, ensure_ascii=False, indent=2),
        "ts": ts,
        "meta": {
            "tool_call_id": tool_call_id,
        },
    }


def materialize_inline_artifact_to_file(
    *,
    artifact: WorkspaceArtifact,
    outdir: pathlib.Path,
    turn_id: Optional[str],
    filename_hint: Optional[str] = None,
    mime_hint: Optional[str] = None,
    visibility: Optional[str] = None,
    scratchpad=None,
) -> None:
    # Normalize to the artifact root (out/workdir) so writes land where every
    # reader looks (the ANNOUNCE [WORKSPACE] scan, git-lineage publish staging,
    # react.rg/read). Without this, a runtime-root outdir writes files to
    # out/turn_<id>/... (missing the workdir layer) — invisible to the workspace
    # map and never committed to the git lineage. artifact_outdir_for is
    # idempotent, so passing an already-artifact-root outdir is a no-op.
    from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace import artifact_outdir_for
    workdir = artifact_outdir_for(outdir)
    save_hint = filename_hint
    if turn_id:
        hint = str(filename_hint or getattr(artifact, "path", "") or getattr(artifact, "filename", "") or "").strip()
        physical_hint, rel_hint, _ = normalize_physical_path(
            hint,
            turn_id=turn_id,
        )
        _, namespace, rel = split_physical_artifact_path(physical_hint)
        namespace = namespace or infer_artifact_namespace(hint, default=ARTIFACT_NAMESPACE_FILES)
        save_hint = rel or rel_hint or filename_hint
        workdir = workdir / turn_id / namespace
    updated, produced = artifact.save_inline(
        workdir=workdir,
        filename_hint=save_hint,
        mime_hint=mime_hint,
        visibility=visibility,
    )
    if produced and scratchpad is not None:
        try:
            scratchpad.add_produced_file(produced)
        except Exception:
            pass


def surrogate_from_writer_inputs(tool_id: str, inputs: Dict[str, Any]) -> tuple[str | None, str | None]:
    if not isinstance(inputs, dict):
        return None, None
    mime_hint = (inputs.get("mime") or None)
    if tool_id == "infra.write":
        content = inputs.get("content")
        if isinstance(content, (bytes, bytearray)):
            cd = inputs.get("content_description")
            return (cd if isinstance(cd, str) and cd.strip() else None), (mime_hint or "application/octet-stream")
        if isinstance(content, str):
            return content, (mime_hint or "text/plain")
    for key in ("content", "markdown", "html", "text"):
        if isinstance(inputs.get(key), str) and inputs.get(key).strip():
            return inputs.get(key), (mime_hint or None)
    return None, mime_hint


def build_artifact_view(
    *,
    turn_id: str,
    is_current: bool,
    artifact_id: str,
    tool_id: str,
    value: Any,
    summary: str,
    sources_used: List[Any] | None = None,
    inputs: Dict[str, Any] | None = None,
    call_record_rel: str | None = None,
    call_record_abs: str | None = None,
    artifact_kind: Optional[str] = None,
    visibility: Optional[str] = None,
    description: Optional[str] = None,
    channel: Optional[str] = None,
    error: Optional[Dict[str, Any]] = None,
    content_lineage: List[str] | None = None,
    tool_call_id: str | None = None,
    artifact_stats: Optional[Dict[str, Any]] = None,
) -> "ReactArtifactView":
    from kdcube_ai_app.apps.chat.sdk.runtime.harness.timeline.turn_view import (
        extract_source_sids,
    )
    value_norm = value
    if tools_insights.is_write_tool(tool_id):
        if isinstance(value, dict) and isinstance(value.get("path"), str) and value["path"].strip():
            file_path = value["path"].strip()
        elif isinstance(value, str) and value.strip():
            file_path = value.strip()
        else:
            file_path = ""
        surrogate_text, mime_hint = surrogate_from_writer_inputs(tool_id, inputs or {})
        value_norm = {
            "type": "file",
            "path": file_path,
            "text": (surrogate_text or ""),
            "mime": (mime_hint or tools_insights.default_mime_for_write_tool(tool_id)),
        }
        if isinstance(artifact_stats, dict) and artifact_stats:
            for k, v in artifact_stats.items():
                if k not in value_norm:
                    value_norm[k] = v
        if sources_used:
            value_norm["sources_used"] = extract_source_sids(sources_used)
        try:
            if file_path:
                from pathlib import Path
                value_norm["filename"] = Path(file_path).name
        except Exception:
            pass

    artifact = {
        "artifact_id": artifact_id,
        "tool_id": tool_id,
        "value": value_norm,
        "summary": str(summary or ""),
        "sources_used": extract_source_sids(sources_used) if sources_used else [],
        "timestamp": time.time(),
        "inputs": dict(inputs or {}),
        "call_record": {"rel": call_record_rel, "abs": call_record_abs},
        "artifact_kind": artifact_kind,
        "visibility": visibility,
        "description": description or "",
        "channel": channel or "",
        "tool_call_id": tool_call_id,
        "error": error,
        "content_lineage": content_lineage or [],
    }
    if isinstance(value_norm, dict):
        path_val = (value_norm.get("path") or "").strip()
        if path_val:
            artifact["path"] = path_val
    workspace_artifact = WorkspaceArtifact.from_artifact_dict(
        artifact,
        turn_id=turn_id,
    )
    return ReactArtifactView(
        **vars(workspace_artifact),
        is_current=is_current,
    )

@dataclass
class ReactArtifactView(WorkspaceArtifact):
    """ReAct rendering adapter over a harness workspace artifact."""

    is_current: bool = False

    @staticmethod
    def extract_files_from_contrib_log(contrib_log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build the latest assistant file record per path from contrib_log blocks.
        We only include external files (visibility=external, kind=file).
        """
        files: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for blk in reversed(contrib_log or []):
            if not isinstance(blk, dict):
                continue
            if (blk.get("type") or "") != "react.tool.result":
                continue
            if (blk.get("mime") or "").strip() != "application/json":
                continue
            text = blk.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            try:
                meta = json.loads(text)
            except Exception:
                continue
            if not isinstance(meta, dict):
                continue
            if meta.get("error"):
                continue
            if (meta.get("visibility") or "").strip() != "external":
                continue
            if (meta.get("kind") or "").strip() != "file":
                continue
            if not (meta.get("hosted_uri") or meta.get("rn") or meta.get("key") or meta.get("physical_path") or meta.get("local_path")):
                continue
            artifact_path = (meta.get("artifact_path") or "").strip()
            if not artifact_path or artifact_path in seen:
                continue
            seen.add(artifact_path)
            physical_path = (meta.get("physical_path") or meta.get("local_path") or "").strip()
            rec = {
                "artifact_path": artifact_path,
                "filename": physical_path.split("/")[-1] if physical_path else "",
                "mime": meta.get("mime") or "",
                "visibility": meta.get("visibility") or "external",
                "kind": meta.get("kind") or "file",
                "hosted_uri": meta.get("hosted_uri"),
                "rn": meta.get("rn"),
                "key": meta.get("key"),
                "path": physical_path,
                "tool_id": meta.get("tool_id") or "",
                "tool_call_id": meta.get("tool_call_id") or "",
            }
            files.append(rec)
        files.reverse()
        return files

    def to_historical_format(self, *, max_tokens: int = 200) -> List[str]:
        lines: List[str] = []
        name = self.path or self.filename or "artifact"
        lines.append(f"- {name}")
        namespace, rel = self.namespace_and_relpath()
        art_path = self.artifact_ref() or (
            build_logical_artifact_path(turn_id=self.turn_id, namespace=namespace, relpath=rel)
            if self.turn_id and namespace and rel
            else ""
        )
        if art_path:
            lines.append(f"    logical_path: {art_path}")
        phys = (
            build_physical_artifact_path(turn_id=self.turn_id, namespace=namespace, relpath=rel)
            if self.turn_id and namespace and rel
            else ""
        )
        if phys:
            lines.append("    physical_path: exists (derive)")
        meta = []
        if self.tool_id:
            meta.append(f"tool={self.tool_id}")
        if self.tool_call_id:
            meta.append(f"tool_call_id={self.tool_call_id}")
        if self.kind:
            meta.append(f"kind={self.kind}")
        if self.visibility:
            meta.append(f"visibility={self.visibility}")
        if self.channel:
            meta.append(f"channel={self.channel}")
        if self.mime:
            meta.append(f"mime={self.mime}")
        if self.size_bytes is not None:
            meta.append(f"size={self.size_bytes}B")
        if meta:
            lines.append("    meta: " + "; ".join(meta))
        used_sids: List[str] = []
        for s in self.sources_used or []:
            if isinstance(s, (int, float)):
                used_sids.append(f"S{int(s)}")
            elif isinstance(s, dict) and isinstance(s.get("sid"), (int, float)):
                used_sids.append(f"S{int(s.get('sid'))}")
        if used_sids:
            lines.append("    sources_used: " + ", ".join(used_sids))

        content = (self.text or "").strip()
        if content:
            if token_count(content) <= max_tokens:
                lines.append("    content:")
                lines.append("    ```text")
                lines.append(content)
                lines.append("    ```")
            elif self.summary:
                lines.append(f"    summary: {self.summary}")
            else:
                lines.append("    content:")
                lines.append("    ```text")
                lines.append(_truncate(content, 1000))
                lines.append("    ```")
        elif self.summary:
            lines.append(f"    summary: {self.summary}")
        return lines

    def to_current_format(self, *, max_tokens: int = 200, fallback_name: Optional[str] = None) -> List[str]:
        lines: List[str] = []
        name = self.path or self.filename or (fallback_name or "artifact")
        lines.append(f"- {name}")
        namespace, rel = self.namespace_and_relpath()
        art_path = self.artifact_ref() or (
            build_logical_artifact_path(turn_id=self.turn_id, namespace=namespace, relpath=rel)
            if self.turn_id and namespace and rel
            else ""
        )
        if art_path:
            lines.append(f"    logical_path: {art_path}")
        phys = (
            build_physical_artifact_path(turn_id=self.turn_id, namespace=namespace, relpath=rel)
            if self.turn_id and namespace and rel
            else ""
        )
        if phys:
            lines.append("    physical_path: exists (derive)")
        meta = []
        if self.tool_id:
            meta.append(f"tool={self.tool_id}")
        if self.tool_call_id:
            meta.append(f"tool_call_id={self.tool_call_id}")
        if self.kind:
            meta.append(f"kind={self.kind}")
        if self.visibility:
            meta.append(f"visibility={self.visibility}")
        if self.channel:
            meta.append(f"channel={self.channel}")
        if self.mime:
            meta.append(f"mime={self.mime}")
        if self.size_bytes is not None:
            meta.append(f"size={self.size_bytes}B")
        if meta:
            lines.append("    meta: " + "; ".join(meta))
        used_sids: List[str] = []
        for s in self.sources_used or []:
            if isinstance(s, (int, float)):
                used_sids.append(f"S{int(s)}")
            elif isinstance(s, dict) and isinstance(s.get("sid"), (int, float)):
                used_sids.append(f"S{int(s.get('sid'))}")
        if used_sids:
            lines.append("    sources_used: " + ", ".join(used_sids))

        content = (self.text or "").strip()
        if content:
            if token_count(content) <= max_tokens:
                lines.append("    content:")
                lines.append("    ```text")
                lines.append(content)
                lines.append("    ```")
            elif self.summary:
                lines.append(f"    summary: {self.summary}")
            else:
                lines.append("    content:")
                lines.append("    ```text")
                lines.append(_truncate(content, 1000))
                lines.append("    ```")
        elif self.summary:
            lines.append(f"    summary: {self.summary}")
        return lines
