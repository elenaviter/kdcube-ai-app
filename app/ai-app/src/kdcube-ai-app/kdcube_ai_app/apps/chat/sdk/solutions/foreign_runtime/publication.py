# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""Shared admission policy for foreign-agent conversation-file publication.

Generated or agent-authored files stay local until an adapter explicitly asks
the trusted parent to host them. This module validates that authority crossing
before bytes enter staging or conversation storage. Products may narrow the
shared ceiling and approve the concrete request, but cannot widen it.
"""

from __future__ import annotations

import inspect
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional, Sequence


SHARED_PUBLICATION_MAX_FILES = 50
SHARED_PUBLICATION_MAX_FILE_BYTES = 100 * 1024 * 1024
SHARED_PUBLICATION_MAX_TOTAL_BYTES = 250 * 1024 * 1024

SHARED_PUBLICATION_MIME_PREFIXES = (
    "audio/",
    "image/",
    "text/",
    "video/",
)
SHARED_PUBLICATION_MIME_TYPES = frozenset({
    "application/epub+zip",
    "application/gzip",
    "application/json",
    "application/ld+json",
    "application/msword",
    "application/pdf",
    "application/rtf",
    "application/vnd.apache.parquet",
    "application/vnd.ms-excel",
    "application/vnd.ms-powerpoint",
    "application/vnd.oasis.opendocument.presentation",
    "application/vnd.oasis.opendocument.spreadsheet",
    "application/vnd.oasis.opendocument.text",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/x-7z-compressed",
    "application/x-tar",
    "application/xml",
    "application/yaml",
    "application/zip",
})

_MIME_BY_SUFFIX = {
    ".csv": "text/csv",
    ".ipynb": "application/json",
    ".json": "application/json",
    ".md": "text/markdown",
    ".svg": "image/svg+xml",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
}


class WorkspacePublishError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class WorkspacePublicationFile:
    source_path: Path
    relative_path: str
    filename: str
    mime: str
    size_bytes: int


@dataclass(frozen=True)
class WorkspacePublicationRequest:
    tenant: str
    project: str
    user_id: str
    user_type: str
    conversation_id: str
    turn_id: str
    request_id: str
    files: tuple[WorkspacePublicationFile, ...]

    @property
    def total_bytes(self) -> int:
        return sum(item.size_bytes for item in self.files)


@dataclass(frozen=True)
class WorkspacePublicationDecision:
    allowed: bool
    code: str = ""
    reason: str = ""

    @classmethod
    def approved(cls) -> "WorkspacePublicationDecision":
        return cls(allowed=True)

    @classmethod
    def denied(
        cls,
        reason: str,
        *,
        code: str = "publish_not_approved",
    ) -> "WorkspacePublicationDecision":
        return cls(allowed=False, code=code, reason=str(reason or "publication was not approved"))


WorkspacePublicationApprover = Callable[
    [WorkspacePublicationRequest],
    WorkspacePublicationDecision | Awaitable[WorkspacePublicationDecision],
]


@dataclass(frozen=True)
class WorkspacePublicationPolicy:
    """Product narrowing layered below the immutable shared ceiling.

    Numeric values are clamped to the shared maximum. Supplying MIME rules
    creates an additional allow-list: a file must pass both shared and product
    type admission. The approver runs last under trusted parent identity.
    """

    max_files: Optional[int] = None
    max_file_bytes: Optional[int] = None
    max_total_bytes: Optional[int] = None
    allowed_mime_types: Optional[Sequence[str]] = None
    allowed_mime_prefixes: Optional[Sequence[str]] = None
    approver: Optional[WorkspacePublicationApprover] = None

    def __post_init__(self) -> None:
        for name in ("max_files", "max_file_bytes", "max_total_bytes"):
            value = getattr(self, name)
            if value is None:
                continue
            normalized = int(value)
            if normalized <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, normalized)
        if self.allowed_mime_types is not None:
            object.__setattr__(
                self,
                "allowed_mime_types",
                tuple(
                    sorted({
                        str(value or "").strip().lower()
                        for value in self.allowed_mime_types
                        if str(value or "").strip()
                    })
                ),
            )
        if self.allowed_mime_prefixes is not None:
            object.__setattr__(
                self,
                "allowed_mime_prefixes",
                tuple(
                    sorted({
                        str(value or "").strip().lower()
                        for value in self.allowed_mime_prefixes
                        if str(value or "").strip()
                    })
                ),
            )


def workspace_publication_mime(path: Path | str) -> str:
    source = Path(path)
    suffix = source.suffix.lower()
    return _MIME_BY_SUFFIX.get(suffix) or mimetypes.guess_type(source.name)[0] or "application/octet-stream"


def _mime_allowed(
    mime: str,
    *,
    exact: Sequence[str],
    prefixes: Sequence[str],
) -> bool:
    normalized = str(mime or "").strip().lower().split(";", 1)[0]
    return normalized in exact or any(normalized.startswith(prefix) for prefix in prefixes)


def _effective_limit(product_value: Optional[int], shared_value: int) -> int:
    return min(int(product_value), shared_value) if product_value is not None else shared_value


async def validate_workspace_publication(
    selected: Sequence[tuple[Path, str]],
    *,
    tenant: str,
    project: str,
    user_id: str,
    user_type: str,
    conversation_id: str,
    turn_id: str,
    request_id: str = "",
    policy: Optional[WorkspacePublicationPolicy] = None,
) -> WorkspacePublicationRequest:
    """Validate selected files and obtain any product-specific approval."""
    product = policy or WorkspacePublicationPolicy()
    max_files = _effective_limit(product.max_files, SHARED_PUBLICATION_MAX_FILES)
    max_file_bytes = _effective_limit(
        product.max_file_bytes,
        SHARED_PUBLICATION_MAX_FILE_BYTES,
    )
    max_total_bytes = _effective_limit(
        product.max_total_bytes,
        SHARED_PUBLICATION_MAX_TOTAL_BYTES,
    )

    if len(selected) > max_files:
        raise WorkspacePublishError(
            "publish_file_count_exceeded",
            f"publish accepts at most {max_files} files per request",
        )

    files: list[WorkspacePublicationFile] = []
    total_bytes = 0
    for source, relative in selected:
        size_bytes = source.stat().st_size
        if size_bytes > max_file_bytes:
            raise WorkspacePublishError(
                "publish_file_too_large",
                f"publish file exceeds {max_file_bytes} bytes: {relative}",
            )
        mime = workspace_publication_mime(source)
        if not _mime_allowed(
            mime,
            exact=tuple(SHARED_PUBLICATION_MIME_TYPES),
            prefixes=SHARED_PUBLICATION_MIME_PREFIXES,
        ):
            raise WorkspacePublishError(
                "publish_file_type_unsupported",
                f"publish does not support {mime} output: {relative}",
            )
        product_has_type_filter = (
            product.allowed_mime_types is not None
            or product.allowed_mime_prefixes is not None
        )
        if product_has_type_filter and not _mime_allowed(
            mime,
            exact=tuple(product.allowed_mime_types or ()),
            prefixes=tuple(product.allowed_mime_prefixes or ()),
        ):
            raise WorkspacePublishError(
                "publish_file_type_not_allowed",
                f"this application does not allow {mime} output: {relative}",
            )
        total_bytes += size_bytes
        if total_bytes > max_total_bytes:
            raise WorkspacePublishError(
                "publish_total_too_large",
                f"publish request exceeds {max_total_bytes} aggregate bytes",
            )
        files.append(
            WorkspacePublicationFile(
                source_path=source,
                relative_path=relative,
                filename=source.name,
                mime=mime,
                size_bytes=size_bytes,
            )
        )

    request = WorkspacePublicationRequest(
        tenant=str(tenant or ""),
        project=str(project or ""),
        user_id=str(user_id or ""),
        user_type=str(user_type or "registered"),
        conversation_id=str(conversation_id or ""),
        turn_id=str(turn_id or ""),
        request_id=str(request_id or ""),
        files=tuple(files),
    )
    if product.approver is None:
        return request

    decision = product.approver(request)
    if inspect.isawaitable(decision):
        decision = await decision
    if not isinstance(decision, WorkspacePublicationDecision):
        raise WorkspacePublishError(
            "publish_approval_invalid",
            "application publication approver returned an invalid decision",
        )
    if not decision.allowed:
        raise WorkspacePublishError(
            decision.code or "publish_not_approved",
            decision.reason or "application publication policy denied this request",
        )
    return request


__all__ = [
    "SHARED_PUBLICATION_MAX_FILES",
    "SHARED_PUBLICATION_MAX_FILE_BYTES",
    "SHARED_PUBLICATION_MAX_TOTAL_BYTES",
    "WorkspacePublicationApprover",
    "WorkspacePublicationDecision",
    "WorkspacePublicationFile",
    "WorkspacePublicationPolicy",
    "WorkspacePublicationRequest",
    "WorkspacePublishError",
    "validate_workspace_publication",
    "workspace_publication_mime",
]
