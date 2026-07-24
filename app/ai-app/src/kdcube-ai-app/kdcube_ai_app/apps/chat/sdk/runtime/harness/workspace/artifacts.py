# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""Framework-neutral artifacts in the agent-harness turn workspace."""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Any

from kdcube_ai_app.apps.chat.sdk.runtime.harness.workspace.references import (
    ARTIFACT_NAMESPACE_FILES,
    build_logical_artifact_path,
    infer_artifact_namespace,
    split_logical_artifact_path,
    split_physical_artifact_path,
)


def _artifact_fields_from_dict(
    artifact: dict[str, Any],
    *,
    turn_id: str | None,
) -> dict[str, Any]:
    value = (
        artifact.get("value")
        if isinstance(artifact.get("value"), dict)
        else {}
    )
    path_value = (artifact.get("path") or value.get("path") or "").strip()
    filename = (
        value.get("filename") or artifact.get("filename") or ""
    ).strip()
    if not filename and path_value:
        filename = pathlib.Path(path_value).name
    artifact_text = artifact.get("text")
    if isinstance(artifact_text, str):
        text = artifact_text
    else:
        text = value.get("text") or value.get("content") or ""
    sources_used = (
        artifact.get("sources_used") or artifact.get("used_sids") or []
    )
    return {
        "path": path_value,
        "filename": filename,
        "mime": (value.get("mime") or artifact.get("mime") or "").strip(),
        "kind": (
            artifact.get("artifact_kind") or artifact.get("kind") or ""
        ).strip(),
        "visibility": (artifact.get("visibility") or "").strip(),
        "channel": (artifact.get("channel") or "").strip(),
        "summary": (artifact.get("summary") or "").strip(),
        "text": text if isinstance(text, str) else "",
        "sources_used": (
            sources_used if isinstance(sources_used, list) else []
        ),
        "tool_id": (artifact.get("tool_id") or "").strip(),
        "tool_call_id": (artifact.get("tool_call_id") or "").strip(),
        "size_bytes": value.get("size_bytes"),
        "turn_id": turn_id or "",
        "raw": artifact,
    }


@dataclass
class WorkspaceArtifact:
    """A produced or materialized object in a harness turn workspace.

    The model is independent of any agent protocol or timeline block type.
    Timeline adapters may render it, while workspace code resolves its
    ``conv:fi`` identity and local file.
    """

    path: str = ""
    filename: str = ""
    mime: str = ""
    kind: str = ""
    visibility: str = ""
    channel: str = ""
    summary: str = ""
    text: str = ""
    sources_used: list[Any] = field(default_factory=list)
    tool_id: str = ""
    tool_call_id: str = ""
    size_bytes: int | None = None
    turn_id: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_output(cls, output: dict[str, Any]) -> "WorkspaceArtifact":
        if not isinstance(output, dict):
            return cls()
        path_value = (output.get("path") or "").strip()
        filename = (output.get("filename") or "").strip()
        if not filename and path_value:
            filename = pathlib.Path(path_value).name
        return cls(
            path=path_value,
            filename=filename,
            mime=(output.get("mime") or "").strip(),
        )

    @classmethod
    def from_artifact_dict(
        cls,
        artifact: dict[str, Any],
        *,
        turn_id: str | None = None,
    ) -> "WorkspaceArtifact":
        if not isinstance(artifact, dict):
            return cls(turn_id=turn_id or "")
        return cls(**_artifact_fields_from_dict(artifact, turn_id=turn_id))

    def save_inline(
        self,
        *,
        workdir: pathlib.Path,
        filename_hint: str | None = None,
        mime_hint: str | None = None,
        visibility: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Materialize an inline artifact as a workspace file."""
        artifact = dict(self.raw or {})
        value = artifact.get("value")
        artifact_id = (artifact.get("artifact_id") or "").strip()

        if isinstance(value, dict) and value.get("type") == "file":
            path = value.get("path")
            if isinstance(path, str) and path.strip():
                file_path = pathlib.Path(path)
                if not file_path.is_absolute():
                    file_path = workdir / file_path
                if not file_path.exists():
                    text = value.get("text")
                    if text is None:
                        text = value.get("content")
                    if text is None:
                        try:
                            text = json.dumps(
                                value,
                                ensure_ascii=False,
                                indent=2,
                            )
                        except Exception:
                            text = ""
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(str(text), encoding="utf-8")
            return artifact, None

        if (artifact.get("artifact_kind") or "").strip() == "file":
            return artifact, None

        text = None
        fmt = None
        if isinstance(value, dict):
            fmt = (
                value.get("format")
                if isinstance(value.get("format"), str)
                else None
            )
            if isinstance(value.get("content"), str):
                text = value.get("content")
            elif isinstance(value.get("text"), str):
                text = value.get("text")
        if text is None and isinstance(value, str):
            text = value
        if text is None:
            try:
                text = json.dumps(value, ensure_ascii=False, indent=2)
                fmt = fmt or "json"
            except Exception:
                return artifact, None

        extension = "txt"
        if isinstance(fmt, str):
            normalized_format = fmt.strip().lower()
            if normalized_format in {"md", "markdown"}:
                extension = "md"
            elif normalized_format == "json":
                extension = "json"
            elif normalized_format in {"html", "htm"}:
                extension = "html"
            elif normalized_format in {"yaml", "yml"}:
                extension = "yaml"

        workdir.mkdir(parents=True, exist_ok=True)
        filename = (
            str(filename_hint).strip()
            if isinstance(filename_hint, str) and filename_hint.strip()
            else ""
        )
        if not filename:
            filename = (self.filename or "").strip()
        if not filename:
            filename = f"{artifact_id or 'artifact'}.{extension}"
        elif "." not in filename:
            filename = f"{filename}.{extension}"
        file_path = workdir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(text, encoding="utf-8")

        mime = (
            str(mime_hint).strip()
            if isinstance(mime_hint, str) and mime_hint.strip()
            else ""
        )
        if not mime:
            mime = (self.mime or "").strip()
        if not mime:
            try:
                from kdcube_ai_app.tools.content_type import (
                    get_mime_type_enhanced,
                )

                mime = get_mime_type_enhanced(filename)
            except Exception:
                mime = ""
        if not mime:
            mime = {
                "md": "text/markdown",
                "json": "application/json",
                "html": "text/html",
                "yaml": "application/x-yaml",
            }.get(extension, "text/plain")

        if not (artifact.get("artifact_kind") or "").strip():
            artifact["artifact_kind"] = "file"
        final_visibility = visibility or (self.visibility or "").strip() or None
        if final_visibility:
            artifact["visibility"] = final_visibility
        artifact["value"] = {
            "type": "file",
            "path": filename,
            "text": text,
            "mime": mime,
            "filename": filename,
        }
        produced_file = {
            "filename": filename,
            "path": filename,
            "artifact_name": artifact_id,
            "mime": mime,
            "size": len(text.encode("utf-8")),
            "summary": artifact.get("summary") or "",
            "visibility": final_visibility or "internal",
            "kind": artifact.get("artifact_kind") or "file",
        }
        return artifact, produced_file

    def physical_path(self, *, run_outdir: pathlib.Path) -> pathlib.Path:
        """Resolve the artifact path for local execution and I/O."""
        if self.path:
            return pathlib.Path(run_outdir) / self.path
        return pathlib.Path("")

    def namespace_and_relpath(self) -> tuple[str, str]:
        """Return the physical workspace namespace and namespace-relative path."""
        candidates = [
            self.path,
            (
                self.raw.get("value")
                if isinstance(self.raw.get("value"), dict)
                else {}
            ).get("path")
            or "",
            (
                self.raw.get("inputs")
                if isinstance(self.raw.get("inputs"), dict)
                else {}
            ).get("path")
            or "",
        ]
        for candidate in candidates:
            candidate = str(candidate or "").strip()
            if not candidate:
                continue
            _, namespace, relpath = split_logical_artifact_path(candidate)
            if namespace and relpath:
                return namespace, relpath
            _, namespace, relpath = split_physical_artifact_path(candidate)
            if namespace and relpath:
                return namespace, relpath
            namespace = infer_artifact_namespace(candidate, default="")
            if namespace and candidate.startswith(f"{namespace}/"):
                return namespace, candidate[len(namespace) + 1 :].lstrip("/")
        fallback_relpath = (self.path or self.filename or "").strip().lstrip("/")
        inputs = (
            self.raw.get("inputs")
            if isinstance(self.raw.get("inputs"), dict)
            else {}
        )
        return (
            infer_artifact_namespace(
                inputs.get("path") or "",
                default=ARTIFACT_NAMESPACE_FILES,
            ),
            fallback_relpath,
        )

    def artifact_ref(self) -> str:
        """Return the durable ``conv:fi`` ref when turn and path are known."""
        namespace, relpath = self.namespace_and_relpath()
        if self.turn_id and namespace and relpath:
            return build_logical_artifact_path(
                turn_id=self.turn_id,
                namespace=namespace,
                relpath=relpath,
            )
        return ""
