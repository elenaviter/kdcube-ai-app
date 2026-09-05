# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""Framework-neutral conversation-file hosting for agent adapters."""

from __future__ import annotations

import logging
import pathlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from kdcube_ai_app.apps.chat.emitters import ChatCommunicator
from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.layout import (
    artifact_outdir_for,
    resolve_artifact_path,
    runtime_outdir_for_artifact_outdir,
)
from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.references import (
    build_logical_artifact_path,
    split_physical_artifact_ref,
)
from kdcube_ai_app.apps.chat.sdk.storage.conversation_store import ConversationStore
from kdcube_ai_app.infra.service_hub.inventory import AgentLogger
from kdcube_ai_app.infra.service_hub.multimodality import (
    MODALITY_IMAGE_MIME,
    validate_image_bytes,
)


def _is_hosted_path(path: str) -> bool:
    if not isinstance(path, str) or not path.strip():
        return False
    value = path.strip()
    return value.startswith("cb/") or "://" in value


class ApplicationHostingService:
    """
    Host local files into ConversationStore and emit chat events for hosted artifacts.
    """

    def __init__(
        self,
        *,
        store: ConversationStore,
        comm: Optional[ChatCommunicator] = None,
        logger: Optional[Union[logging.Logger, AgentLogger]] = None,
    ):
        self.store = store
        self.comm = comm
        self.log = logger or logging.getLogger(__name__)

    def _log_error(self, message: str) -> None:
        error = getattr(self.log, "error", None)
        if callable(error):
            error(message)
            return
        self.log.log(message, level="ERROR")

    def _extract_file_fields(self, a: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(a, dict):
            return None

        if a.get("type") == "file":
            output = a.get("output") or {}
            path = output.get("path") or a.get("path") or ""
            text = output.get("text") if isinstance(output, dict) else None
            return {
                "path": path,
                "mime": a.get("mime") or (output.get("mime") if isinstance(output, dict) else None),
                "visibility": a.get("visibility") or (output.get("visibility") if isinstance(output, dict) else None),
                "tool_id": a.get("tool_id") or "",
                "description": a.get("description") or "",
                "slot": a.get("resource_id") or a.get("slot") or a.get("artifact_id") or "",
                "text": text,
            }

        val = a.get("value") if isinstance(a.get("value"), dict) else None
        if isinstance(val, dict) and val.get("type") == "file":
            return {
                "path": val.get("path") or "",
                "mime": val.get("mime"),
                "visibility": a.get("visibility") or val.get("visibility"),
                "tool_id": a.get("tool_id") or "",
                "description": a.get("description") or "",
                "slot": a.get("resource_id") or a.get("slot") or a.get("artifact_id") or "",
                "text": val.get("text"),
            }

        return None

    async def host_files_to_conversation(
        self,
        *,
        rid: str,
        files: List[Dict[str, Any]],
        outdir: str | pathlib.Path | None,
        tenant: str,
        project: str,
        user: Optional[str] = None,
        conversation_id: str,
        user_type: str,
        turn_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Copy deliverable file artifacts from local outdir → ConversationStore.
        Returns rows: [{slot, key, hosted_uri, filename, mime, size, content_sha256, tool_id, description, owner_id, rn, physical_path}]

        The file OWNER (``user``) is the download-critical key component: it must be
        the same user the download resolver reconstructs. When a caller omits it, it
        is resolved from the bound communicator's service context (the one canonical
        source — see ``event_identity.resolve_request_identity``), so a file is always
        hosted under the turn's real user. Every existing caller passes ``user``
        explicitly, so this default only fills in for callers that don't.
        """
        import pathlib as _pathlib
        import hashlib as _hashlib

        if not str(user or "").strip():
            from kdcube_ai_app.apps.chat.sdk.event_identity import resolve_request_identity
            user = resolve_request_identity(self.comm).get("owner") or ""

        files_rehosted: List[Dict[str, Any]] = []
        base = artifact_outdir_for(_pathlib.Path(outdir), create=False) if outdir else None
        runtime_base = _pathlib.Path(outdir) if outdir else None
        for a in (files or []):
            info = self._extract_file_fields(a)
            if not info:
                continue
            rel_or_abs = (info.get("path") or "").strip()
            if not rel_or_abs:
                continue
            if _is_hosted_path(rel_or_abs):
                continue

            p = _pathlib.Path(rel_or_abs)
            if not p.is_absolute():
                if runtime_base is not None:
                    p = resolve_artifact_path(runtime_base, rel_or_abs).resolve()
                else:
                    p = (base / rel_or_abs).resolve() if base else p.resolve()
            try:
                data = p.read_bytes()
            except Exception as ex:
                self._log_error(f"[host_files] Failed to read file {p}: {ex}")
                continue
            declared_mime = str(info.get("mime") or "application/octet-stream").strip().lower()
            if declared_mime in MODALITY_IMAGE_MIME:
                image_validation = validate_image_bytes(data, media_type=declared_mime)
                if not image_validation.get("valid"):
                    self._log_error(
                        "[host_files] Rejected invalid image "
                        f"{p}: {image_validation.get('error') or 'invalid_image_data'}"
                    )
                    continue
            # Content fingerprint over the bytes already read (no extra I/O).
            content_sha256 = _hashlib.sha256(data).hexdigest()

            name = p.name
            physical_path = str(p)
            if base:
                try:
                    physical_path = str(p.relative_to(base))
                except Exception:
                    if runtime_base:
                        try:
                            physical_path = str(p.relative_to(runtime_base))
                        except Exception:
                            physical_path = str(p)
                    else:
                        physical_path = str(p)
            physical_path = physical_path.replace("\\", "/")
            if physical_path.startswith("/"):
                physical_path = name
            physical_path = physical_path.strip("/")
            try:
                pure_physical = pathlib.PurePosixPath(physical_path)
                if not physical_path or physical_path.startswith("/") or any(part in ("", ".", "..") for part in pure_physical.parts):
                    physical_path = name
            except Exception:
                physical_path = name
            logical_path = ""
            physical_conversation_id, physical_turn_id, physical_namespace, physical_rel = split_physical_artifact_ref(physical_path)
            if physical_turn_id and physical_namespace and physical_rel:
                logical_path = build_logical_artifact_path(
                    turn_id=physical_turn_id,
                    namespace=physical_namespace,
                    relpath=physical_rel,
                    conversation_id=physical_conversation_id or conversation_id,
                )
            uri, key, rn_f = await self.store.put_artifact_file(
                tenant=tenant,
                project=project,
                user=user,
                fingerprint=None,
                conversation_id=conversation_id,
                relpath=physical_path,
                data=data,
                mime=info.get("mime") or "application/octet-stream",
                turn_id=turn_id,
            )
            files_rehosted.append({
                "slot": info.get("slot") or "",
                "key": key,
                "filename": name,
                "mime": info.get("mime") or "application/octet-stream",
                "visibility": info.get("visibility") or "external",
                "size": len(data),
                "content_sha256": content_sha256,
                "tool_id": info.get("tool_id") or "",
                "description": info.get("description") or "",
                "owner_id": user,
                "rn": rn_f,
                "hosted_uri": uri,
                "physical_path": physical_path,
                "logical_path": logical_path,
            })
        return files_rehosted

    async def persist_workspace(
        self,
        *,
        outdir: Optional[str],
        workdir: Optional[str],
        tenant: str,
        project: str,
        user: Optional[str],
        conversation_id: str,
        user_type: str,
        turn_id: str,
        codegen_run_id: str,
        fingerprint: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Persist execution snapshot (out/work trees) into ConversationStore.
        Mirrors BaseWorkflow._snapshot_execution_tree.
        """
        if not self.store:
            return None
        if not (tenant and project and conversation_id and turn_id and codegen_run_id):
            return None
        try:
            if outdir:
                try:
                    outdir = str(runtime_outdir_for_artifact_outdir(pathlib.Path(outdir)))
                except Exception:
                    pass
            return await self.store.put_execution_snapshot(
                tenant=tenant,
                project=project,
                user=user,
                user_type=user_type,
                fingerprint=fingerprint,
                conversation_id=conversation_id,
                turn_id=turn_id,
                codegen_run_id=codegen_run_id,
                out_dir=outdir,
                pkg_dir=workdir,
            )
        except Exception as exc:
            try:
                self.log.log(f"[persist_workspace] failed: {exc}", level="ERROR")
            except Exception:
                pass
            return None

    async def emit_solver_artifacts(self, *, files: List[Dict[str, Any]], citations: List[Dict[str, Any]]) -> None:
        """
        Emits chat events for batch files + citations.
        """
        if not self.comm:
            return
        from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.references import normalize_file_payload
        service = getattr(self.comm, "service", None) or {}
        conversation_id = str(service.get("conversation_id") or "").strip()
        event_dt = datetime.now(timezone.utc)
        event_ts_ms = int(event_dt.timestamp() * 1000)
        event_ts_iso = event_dt.isoformat().replace("+00:00", "Z")
        cleaned_files: List[Dict[str, Any]] = []
        for item in files or []:
            if not isinstance(item, dict):
                continue
            payload = normalize_file_payload(dict(item))
            logical_path = str(payload.get("logical_path") or "").strip()
            if not logical_path:
                physical_path = str(payload.get("physical_path") or payload.get("path") or "").strip()
                physical_conversation_id, physical_turn_id, physical_namespace, physical_rel = split_physical_artifact_ref(physical_path)
                if physical_turn_id and physical_namespace and physical_rel:
                    logical_path = build_logical_artifact_path(
                        turn_id=physical_turn_id,
                        namespace=physical_namespace,
                        relpath=physical_rel,
                        conversation_id=physical_conversation_id or conversation_id,
                    )
            if logical_path:
                payload["logical_path"] = logical_path
                payload["object_ref"] = logical_path
                payload["ref"] = logical_path
            payload.setdefault("timestamp", event_ts_ms)
            payload.setdefault("timestamp_iso", event_ts_iso)
            payload.setdefault("ts", event_ts_ms)
            data = payload.get("data")
            if isinstance(data, dict):
                meta = data.get("meta")
                if isinstance(meta, dict):
                    meta = normalize_file_payload(meta)
                    data = dict(data)
                    data["meta"] = meta
                    payload["data"] = data
            cleaned_files.append(payload)
        cleaned_citations: List[Dict[str, Any]] = []
        for item in citations or []:
            if not isinstance(item, dict):
                continue
            payload = dict(item)
            payload.setdefault("timestamp", event_ts_ms)
            payload.setdefault("timestamp_iso", event_ts_iso)
            payload.setdefault("ts", event_ts_ms)
            cleaned_citations.append(payload)
        if cleaned_files:
            await self.comm.event(
                agent="tooling",
                type="chat.files",
                title=f"Files Ready ({len(cleaned_files)})",
                step="files",
                status="completed",
                data={"count": len(cleaned_files), "items": cleaned_files},
            )
        if cleaned_citations:
            await self.comm.event(
                agent="tooling",
                type="chat.citations",
                title=f"Citations ({len(cleaned_citations)})",
                step="citations",
                status="completed",
                data={"count": len(cleaned_citations), "items": cleaned_citations},
            )
