# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Provider-neutral document named service.

The ``docs`` namespace models word-processing documents. Google Docs is the
first transport, but the named-service contract does not expose Google access
tokens or require consumers to use Google-specific MCP tools. The document is a
flatter object than a spreadsheet: one ``docs.document`` object kind rather than
a spreadsheet-plus-tab pair.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import Any

from kdcube_ai_app.apps.chat.sdk.integrations.named_service_consent import (
    CONSENT_ERROR_CONTRACT,
    tool_error_response,
)
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers import (
    NamedServiceContext,
    NamedServiceProvider,
    NamedServiceProviderSpec,
    NamedServiceRequest,
    NamedServiceResponse,
    NamedServiceSearchScope,
    NamedServiceStreamResult,
    TRANSPORT_API,
    TRANSPORT_LOCAL,
    named_service_provider,
)
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.types import (
    BLOCK_PRODUCE,
    EVENT_RESOLVE,
    OBJECT_ACTION,
    OBJECT_DELETE,
    OBJECT_GET,
    OBJECT_LIST,
    OBJECT_SCHEMA,
    OBJECT_SEARCH,
    OBJECT_UPSERT,
    PROVIDER_ABOUT,
    PROVIDER_CAPABILITIES,
)


LOGGER = logging.getLogger("kdcube.sdk.integrations.docs.named_service")


DOCS_NAMESPACE = "docs"
PROVIDER_ID = "sdk.integrations.docs"
GOOGLE_PROVIDER_KEY = "google"
DOCS_READ_CLAIM = "docs:read"
DOCS_WRITE_CLAIM = "docs:write"
DOCS_COMMENT_CLAIM = "docs:comment"

DOCS_DOCUMENT_KIND = "docs.document"
DOCS_TRANSPORTS = (TRANSPORT_LOCAL, TRANSPORT_API)
DOCS_SNAPSHOT_SCHEMA = "kdcube.docs.snapshot.v1"
DOCS_SNAPSHOT_MEDIA_TYPE = "application/vnd.kdcube.docs.snapshot+json"

# The bounded verbs a caller may name through object.action. Each verb maps to
# the identically named underlying proxy operation.
ACTION_INSERT_TEXT = "insert_text"
ACTION_APPEND_TEXT = "append_text"
ACTION_REPLACE_TEXT = "replace_text"
ACTION_APPLY_TEXT_STYLE = "apply_text_style"
ACTION_INSERT_PAGE_BREAK = "insert_page_break"
ACTION_EMBED_IMAGE = "embed_image"
ACTION_EXPORT = "export"
ACTION_IMPORT = "import"
ACTION_LIST_COMMENTS = "list_comments"
ACTION_GET_COMMENT = "get_comment"
ACTION_CREATE_COMMENT = "create_comment"
ACTION_REPLY_COMMENT = "reply_comment"
ACTION_RESOLVE_COMMENT = "resolve_comment"
ACTION_DELETE_COMMENT = "delete_comment"

# Read-only actions gate on docs:read alone.
DOCS_READ_ACTIONS = frozenset(
    {ACTION_EXPORT, ACTION_LIST_COMMENTS, ACTION_GET_COMMENT}
)
# Body-mutating actions gate on docs:read + docs:write.
DOCS_WRITE_ACTIONS = frozenset(
    {
        ACTION_INSERT_TEXT,
        ACTION_APPEND_TEXT,
        ACTION_REPLACE_TEXT,
        ACTION_APPLY_TEXT_STYLE,
        ACTION_INSERT_PAGE_BREAK,
        ACTION_EMBED_IMAGE,
        ACTION_IMPORT,
    }
)
# Comment-thread actions gate on docs:read + docs:comment.
DOCS_COMMENT_ACTIONS = frozenset(
    {
        ACTION_CREATE_COMMENT,
        ACTION_REPLY_COMMENT,
        ACTION_RESOLVE_COMMENT,
        ACTION_DELETE_COMMENT,
    }
)

DOCS_ACTIONS = (
    ACTION_INSERT_TEXT,
    ACTION_APPEND_TEXT,
    ACTION_REPLACE_TEXT,
    ACTION_APPLY_TEXT_STYLE,
    ACTION_INSERT_PAGE_BREAK,
    ACTION_EMBED_IMAGE,
    ACTION_EXPORT,
    ACTION_IMPORT,
    ACTION_LIST_COMMENTS,
    ACTION_GET_COMMENT,
    ACTION_CREATE_COMMENT,
    ACTION_REPLY_COMMENT,
    ACTION_RESOLVE_COMMENT,
    ACTION_DELETE_COMMENT,
)

# How many characters of body text block.produce previews inline.
BLOCK_PREVIEW_CHARS = 800

ExecuteDocsOperation = Callable[..., Awaitable[Mapping[str, Any]]]


def _action_claim(action: str) -> str | tuple[str, str]:
    if action in DOCS_READ_ACTIONS:
        return DOCS_READ_CLAIM
    if action in DOCS_COMMENT_ACTIONS:
        return (DOCS_READ_CLAIM, DOCS_COMMENT_CLAIM)
    return (DOCS_READ_CLAIM, DOCS_WRITE_CLAIM)


DOCS_SEARCH_FILTERS = {
    "account_id": {
        "type": "string",
        "description": (
            "Optional connected document account id. Required when more than "
            "one connected account is eligible."
        ),
    },
    "cursor": {
        "type": "string",
        "description": "Optional next_cursor returned by an earlier search.",
    },
}

DOCS_SEARCH_SCOPES = (
    NamedServiceSearchScope(
        namespace=DOCS_NAMESPACE,
        label="documents",
        object_kind=DOCS_DOCUMENT_KIND,
        description=(
            "Find documents by title through an approved connected account. "
            "A blank query lists recently modified documents."
        ),
        filters_schema=DOCS_SEARCH_FILTERS,
    ),
)

DOCS_GRANT_HINTS = {
    "object.list": [DOCS_READ_CLAIM],
    "object.search": [DOCS_READ_CLAIM],
    "object.get": [DOCS_READ_CLAIM],
    "object.upsert": [DOCS_WRITE_CLAIM],
    "object.delete": [DOCS_COMMENT_CLAIM],
    "object.action": [DOCS_WRITE_CLAIM],
    **{
        f"object.action.{action}": (
            [DOCS_READ_CLAIM]
            if action in DOCS_READ_ACTIONS
            else [DOCS_COMMENT_CLAIM]
            if action in DOCS_COMMENT_ACTIONS
            else [DOCS_WRITE_CLAIM]
        )
        for action in DOCS_ACTIONS
    },
}


def _operation_connected_claims() -> dict[str, list[str]]:
    claims: dict[str, list[str]] = {
        "object.list": [DOCS_READ_CLAIM],
        "object.search": [DOCS_READ_CLAIM],
        "object.get": [DOCS_READ_CLAIM],
        "object.upsert": [DOCS_READ_CLAIM, DOCS_WRITE_CLAIM],
        "object.delete": [DOCS_READ_CLAIM, DOCS_COMMENT_CLAIM],
        "object.action": [DOCS_READ_CLAIM, DOCS_WRITE_CLAIM],
    }
    for action in DOCS_ACTIONS:
        claim = _action_claim(action)
        claims[f"object.action.{action}"] = (
            [claim] if isinstance(claim, str) else list(claim)
        )
    return claims


DOCS_CONNECTED_ACCOUNT_REQUIREMENTS = [
    {
        "provider_id": GOOGLE_PROVIDER_KEY,
        "provider_label": "Google",
        "claims": [DOCS_READ_CLAIM, DOCS_WRITE_CLAIM, DOCS_COMMENT_CLAIM],
        "claim_labels": {
            DOCS_READ_CLAIM: "read documents",
            DOCS_WRITE_CLAIM: "edit documents",
            DOCS_COMMENT_CLAIM: "comment on documents",
        },
        "claims_by_operation": _operation_connected_claims(),
    }
]

DOCS_PROVIDER_CATALOG = {
    GOOGLE_PROVIDER_KEY: {
        "provider_id": GOOGLE_PROVIDER_KEY,
        "label": "Google Docs",
        "claims": {
            "read": DOCS_READ_CLAIM,
            "write": DOCS_WRITE_CLAIM,
            "comment": DOCS_COMMENT_CLAIM,
        },
    },
}

DOCS_SCHEMA = {
    "namespace": DOCS_NAMESPACE,
    "refs": {
        "document": "docs:<provider>:<account_id>:document:<document_id>",
    },
    "object_kinds": {
        DOCS_DOCUMENT_KIND: {
            "description": "One document visible to a connected account.",
            "fields": [
                "ref",
                "provider",
                "account_id",
                "document_id",
                "title",
                "revision_id",
                "web_url",
                "text",
                "tabs",
                "end_index",
                "created_time",
                "modified_time",
            ],
        },
    },
    "search": {"filters": DOCS_SEARCH_FILTERS},
    "get": {
        "description": (
            "Read a document by ref. A document get returns metadata and the "
            "extracted body text. On a configured turnless transport it also "
            "offers a short-lived URL for the complete JSON snapshot."
        ),
        "filters": ["include_text"],
    },
    "materialization": {
        "description": (
            "A materializing client can resolve a document ref through "
            "object.get with response_mode=stream. The JSON snapshot carries "
            "document metadata, the extracted body text, and open comments."
        ),
        "schema": DOCS_SNAPSHOT_SCHEMA,
        "media_type": DOCS_SNAPSHOT_MEDIA_TYPE,
        "refs": ["document"],
    },
    "upsert": {
        "create": {
            "description": "Omit object_ref to create a document.",
            "object": ["title", "initial_text"],
        },
        "document": {
            "description": (
                "Use a document ref with object.text/index to insert or append "
                "body text, or object.replacements to substitute text."
            ),
            "object": ["text", "index", "replacements"],
        },
    },
    "actions": {
        ACTION_INSERT_TEXT: {
            "description": "Insert text at an index (defaults to the body end).",
            "object_ref": "document ref",
            "payload": ["text", "index"],
            "claim": "docs:write",
        },
        ACTION_APPEND_TEXT: {
            "description": "Append text at the end of the body.",
            "object_ref": "document ref",
            "payload": ["text"],
            "claim": "docs:write",
        },
        ACTION_REPLACE_TEXT: {
            "description": "Replace all matches of find with replace text.",
            "object_ref": "document ref",
            "payload": ["replacements"],
            "claim": "docs:write",
        },
        ACTION_APPLY_TEXT_STYLE: {
            "description": "Apply bounded character styling to a text range.",
            "object_ref": "document ref",
            "payload": [
                "start_index",
                "end_index",
                "bold",
                "italic",
                "underline",
                "strikethrough",
                "font_size",
                "link_url",
            ],
            "claim": "docs:write",
        },
        ACTION_INSERT_PAGE_BREAK: {
            "description": "Insert a page break (defaults to the body end).",
            "object_ref": "document ref",
            "payload": ["index"],
            "claim": "docs:write",
        },
        ACTION_EMBED_IMAGE: {
            "description": "Embed a public image URL inline in the document.",
            "object_ref": "document ref",
            "payload": ["image_uri", "index", "width_pt", "height_pt"],
            "claim": "docs:write",
        },
        ACTION_EXPORT: {
            "description": "Export the document to a bounded file format.",
            "object_ref": "document ref",
            "payload": ["format"],
            "claim": "docs:read",
        },
        ACTION_IMPORT: {
            "description": "Create a document by importing source content.",
            "object_ref": "none",
            "payload": ["title", "source_format", "content", "content_base64"],
            "claim": "docs:write",
        },
        ACTION_LIST_COMMENTS: {
            "description": "List comments on the document.",
            "object_ref": "document ref",
            "payload": ["include_resolved", "cursor", "limit"],
            "claim": "docs:read",
        },
        ACTION_GET_COMMENT: {
            "description": "Read one comment thread by id.",
            "object_ref": "document ref",
            "payload": ["comment_id"],
            "claim": "docs:read",
        },
        ACTION_CREATE_COMMENT: {
            "description": "Create a comment on the document.",
            "object_ref": "document ref",
            "payload": ["content", "quoted_text", "anchor"],
            "claim": "docs:comment",
        },
        ACTION_REPLY_COMMENT: {
            "description": "Reply to an existing comment.",
            "object_ref": "document ref",
            "payload": ["comment_id", "content"],
            "claim": "docs:comment",
        },
        ACTION_RESOLVE_COMMENT: {
            "description": "Resolve an existing comment thread.",
            "object_ref": "document ref",
            "payload": ["comment_id", "content"],
            "claim": "docs:comment",
        },
        ACTION_DELETE_COMMENT: {
            "description": "Delete an existing comment thread.",
            "object_ref": "document ref",
            "payload": ["comment_id"],
            "claim": "docs:comment",
        },
    },
    "account_selection": {
        "search": (
            "Pass filters.account_id when several connected accounts are "
            "eligible. Otherwise the response returns reason=account_required "
            "with labeled candidates."
        ),
        "refs": "Every returned object ref embeds its provider and account id.",
    },
    "consent_errors": CONSENT_ERROR_CONTRACT,
    "grant_hints": DOCS_GRANT_HINTS,
    "connected_account_claims": {
        GOOGLE_PROVIDER_KEY: {
            "read": DOCS_READ_CLAIM,
            "write": DOCS_WRITE_CLAIM,
            "comment": DOCS_COMMENT_CLAIM,
        }
    },
}

DOCS_INTRO = (
    "Use namespace `docs` for user-connected documents. Search by title, get a "
    "document ref to read metadata and body text, then use object.upsert or a "
    "declared object.action for explicit changes and comments."
)

DOCS_PRESENTATION = {
    "about": "Find, read, create, edit, and comment on documents you connect.",
    "third_party": "Google Docs is the first provider behind this namespace.",
    "operations": {
        "provider.about": {
            "label": "Service overview",
            "description": "What the document service does and how to use it.",
        },
        "provider.capabilities": {
            "label": "Capabilities",
            "description": "The operations and bounded actions this service supports.",
        },
        "object.list": {
            "label": "Recent documents",
            "description": "List recently modified documents.",
        },
        "object.search": {
            "label": "Search documents",
            "description": "Find documents by title.",
        },
        "object.get": {
            "label": "Read document",
            "description": "Inspect metadata and read the body text.",
        },
        "object.schema": {
            "label": "Document schema",
            "description": "Read refs, fields, limits, and action payloads.",
        },
        "object.upsert": {
            "label": "Create or update document",
            "description": "Create a document or edit its body text.",
        },
        "object.delete": {
            "label": "Delete document comment",
            "description": "Delete one comment; document-file deletion is not exposed.",
        },
    },
    "actions": {
        action: {"label": action.replace("_", " ").title(), **dict(meta)}
        for action, meta in DOCS_SCHEMA["actions"].items()
    },
}


def _operations() -> dict[str, Any]:
    return {
        PROVIDER_ABOUT: {"transports": DOCS_TRANSPORTS},
        PROVIDER_CAPABILITIES: {"transports": DOCS_TRANSPORTS},
        OBJECT_LIST: {"transports": DOCS_TRANSPORTS},
        OBJECT_SEARCH: {"transports": DOCS_TRANSPORTS},
        OBJECT_GET: {"transports": DOCS_TRANSPORTS},
        OBJECT_SCHEMA: {"transports": DOCS_TRANSPORTS},
        OBJECT_UPSERT: {"transports": DOCS_TRANSPORTS},
        OBJECT_DELETE: {"transports": DOCS_TRANSPORTS},
        OBJECT_ACTION: {"transports": DOCS_TRANSPORTS},
        EVENT_RESOLVE: {"transports": DOCS_TRANSPORTS},
        BLOCK_PRODUCE: {"transports": DOCS_TRANSPORTS},
    }


def _spec_metadata() -> dict[str, Any]:
    return {
        "provider_catalog": DOCS_PROVIDER_CATALOG,
        "grant_hints": DOCS_GRANT_HINTS,
        "connected_accounts": DOCS_CONNECTED_ACCOUNT_REQUIREMENTS,
        "canonical_refs": DOCS_SCHEMA["refs"],
        "presentation": DOCS_PRESENTATION,
        "actions": {
            name: str((meta or {}).get("description") or "")
            for name, meta in DOCS_SCHEMA["actions"].items()
        },
        "object_kinds": {
            kind: str((meta or {}).get("description") or "")
            for kind, meta in DOCS_SCHEMA["object_kinds"].items()
        },
    }


def docs_named_service_spec(*, bundle_id: str | None = None) -> NamedServiceProviderSpec:
    return NamedServiceProviderSpec(
        provider_id=PROVIDER_ID,
        bundle_id=bundle_id,
        namespace=DOCS_NAMESPACE,
        refs=("docs:*",),
        object_kinds=(DOCS_DOCUMENT_KIND,),
        search_scopes=DOCS_SEARCH_SCOPES,
        operations=_operations(),
        label="Documents",
        description=(
            "Provider-neutral document namespace over user-connected accounts."
        ),
        intro=DOCS_INTRO,
        metadata=_spec_metadata(),
    )


def _text(value: Any) -> str:
    return str(value or "").strip()


def _int(
    value: Any,
    *,
    default: int = 0,
    minimum: int = 0,
    maximum: int = 2_147_483_647,
) -> int:
    try:
        parsed = int(value if value is not None else default)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _is_materialization_request(request: NamedServiceRequest) -> bool:
    context = request.context if isinstance(request.context, Mapping) else {}
    payload = request.payload if isinstance(request.payload, Mapping) else {}
    if _text(request.response_mode).lower() == "stream":
        return True
    if context.get("materialize") or payload.get("materialize"):
        return True
    return _text(context.get("source") or payload.get("source")) == "react.pull"


def document_ref(account_id: Any, document_id: Any) -> str:
    return (
        f"{DOCS_NAMESPACE}:{GOOGLE_PROVIDER_KEY}:"
        f"{_text(account_id)}:document:{_text(document_id)}"
    )


def parse_docs_ref(value: Any) -> dict[str, Any]:
    ref = _text(value)
    parts = ref.split(":")
    if len(parts) != 5 or parts[0].lower() != DOCS_NAMESPACE:
        raise ValueError("Invalid docs object ref.")
    if parts[3] != "document" or not all(parts[index] for index in (1, 2, 4)):
        raise ValueError("Invalid docs document ref.")
    return {
        "ref": ref,
        "provider": parts[1].lower(),
        "account_id": parts[2],
        "document_id": parts[4],
        "object_kind": DOCS_DOCUMENT_KIND,
    }


def _document_object(
    value: Mapping[str, Any],
    *,
    account_id: str,
    provider: str = GOOGLE_PROVIDER_KEY,
) -> dict[str, Any]:
    row = dict(value or {})
    document_id = _text(row.get("document_id"))
    ref = document_ref(account_id, document_id)
    return {
        **row,
        "ref": ref,
        "object_kind": DOCS_DOCUMENT_KIND,
        "provider": provider,
        "account_id": account_id,
        "document_id": document_id,
    }


def _snapshot_filename(document_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", _text(document_id)).strip("._-")
    return f"{safe or 'document'}.docs.json"


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _snapshot_from_block_target(target: Mapping[str, Any]) -> dict[str, Any]:
    raw = target.get("raw") if isinstance(target.get("raw"), Mapping) else {}
    text = raw.get("text") or target.get("text") or ""
    if not isinstance(text, str) or not text.strip():
        return {}
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return {}
    if not isinstance(value, Mapping):
        return {}
    snapshot = dict(value)
    if _text(snapshot.get("schema")) != DOCS_SNAPSHOT_SCHEMA:
        return {}
    return snapshot


def _snapshot_inventory_text(
    snapshot: Mapping[str, Any],
    *,
    object_ref: str,
    target: Mapping[str, Any],
) -> str:
    obj = snapshot.get("object") if isinstance(snapshot.get("object"), Mapping) else {}
    comments = snapshot.get("comments") if isinstance(snapshot.get("comments"), list) else []
    materialization = (
        snapshot.get("materialization")
        if isinstance(snapshot.get("materialization"), Mapping)
        else {}
    )
    meta = target.get("meta") if isinstance(target.get("meta"), Mapping) else {}
    raw = target.get("raw") if isinstance(target.get("raw"), Mapping) else {}
    logical_path = _text(target.get("logical_path") or target.get("path"))
    physical_path = _text(
        target.get("physical_path")
        or meta.get("physical_path")
        or raw.get("physical_path")
    )
    body_text = obj.get("text") if isinstance(obj.get("text"), str) else ""
    preview = body_text[:BLOCK_PREVIEW_CHARS]
    tabs = [
        _text(tab.get("title"))
        for tab in obj.get("tabs") or []
        if isinstance(tab, Mapping) and _text(tab.get("title"))
    ]
    lines = [
        "[DOCS SNAPSHOT]",
        f"object_ref: {object_ref}",
        f"object_kind: {_text(snapshot.get('object_kind')) or _text(obj.get('object_kind'))}",
        f"title: {_text(obj.get('title')) or '<untitled>'}",
        f"document_id: {_text(obj.get('document_id'))}",
    ]
    web_url = _text(obj.get("web_url"))
    if web_url:
        lines.append(f"web_url: {web_url}")
    revision_id = _text(obj.get("revision_id"))
    if revision_id:
        lines.append(f"revision_id: {revision_id}")
    for label, key in (
        ("source_bytes", "source_bytes"),
        ("source_text_symbols", "source_text_symbols"),
        ("source_line_count", "source_line_count"),
    ):
        if meta.get(key) is not None:
            lines.append(f"{label}: {meta.get(key)}")
    lines.extend(
        [
            f"materialized_path: {logical_path}",
            f"physical_path: {physical_path or '<not exposed>'}",
            f"snapshot_schema: {_text(snapshot.get('schema'))}",
            f"word_count: {_word_count(body_text)}",
            f"char_count: {len(body_text)}",
            f"end_index: {_int(obj.get('end_index'))}",
            f"comment_count: {len(comments)}",
            f"text_materialized: {'yes' if body_text else 'no'}",
        ]
    )
    complete_text = materialization.get("complete_text")
    if complete_text is not None:
        lines.append(f"complete_text: {'yes' if complete_text else 'no'}")
    lines.append("tabs:")
    if not tabs:
        lines.append("- none reported")
    for title in tabs:
        lines.append(f"- {title}")
    lines.append("text_preview:")
    if preview:
        lines.append(preview)
        if len(body_text) > len(preview):
            lines.append("[preview truncated]")
    else:
        lines.append("- no body text")
    lines.extend(
        [
            "snapshot_layout:",
            "- document metadata and body text: object",
            "- comment threads: comments[]",
        ]
    )
    return "\n".join(lines)


async def _json_chunks(
    value: Any,
    *,
    chunk_bytes: int = 64 * 1024,
) -> AsyncIterator[bytes]:
    """Encode a JSON artifact incrementally without monopolizing the proc loop."""
    encoder = json.JSONEncoder(ensure_ascii=False, indent=2)
    pending = bytearray()
    for piece in encoder.iterencode(value):
        encoded = piece.encode("utf-8")
        offset = 0
        if pending:
            take = min(chunk_bytes - len(pending), len(encoded))
            pending.extend(encoded[:take])
            offset = take
            if len(pending) == chunk_bytes:
                yield bytes(pending)
                pending.clear()
                await asyncio.sleep(0)
        while len(encoded) - offset >= chunk_bytes:
            yield encoded[offset : offset + chunk_bytes]
            offset += chunk_bytes
            await asyncio.sleep(0)
        pending.extend(encoded[offset:])
    if pending:
        yield bytes(pending)


@named_service_provider(
    provider_id=PROVIDER_ID,
    namespace=DOCS_NAMESPACE,
    refs=("docs:*",),
    object_kinds=(DOCS_DOCUMENT_KIND,),
    search_scopes=DOCS_SEARCH_SCOPES,
    operations=_operations(),
    label="Documents",
    description="Provider-neutral document namespace over connected accounts.",
    intro=DOCS_INTRO,
    metadata=_spec_metadata(),
)
class DocsNamedServiceProvider(NamedServiceProvider):
    def __init__(
        self,
        *,
        execute_operation: ExecuteDocsOperation,
        bundle_id: str | None = None,
        file_url_factory: Any = None,
    ) -> None:
        super().__init__(docs_named_service_spec(bundle_id=bundle_id))
        self._execute_operation = execute_operation
        self._file_url_factory = file_url_factory

    def _provider_identity(self) -> dict[str, Any]:
        return {"provider_id": PROVIDER_ID, "bundle_id": self.spec.bundle_id}

    async def _download_url(
        self,
        ctx: NamedServiceContext,
        *,
        ref: str,
    ) -> dict[str, Any] | None:
        if self._file_url_factory is None:
            return None
        try:
            out = self._file_url_factory(ctx, {"ref": ref})
            if hasattr(out, "__await__"):
                out = await out
        except Exception:
            LOGGER.exception("docs snapshot url factory failed for %s", ref)
            return None
        return dict(out) if isinstance(out, Mapping) and out.get("url") else None

    def _invalid_ref(
        self, request: NamedServiceRequest, exc: Exception
    ) -> NamedServiceResponse:
        return NamedServiceResponse.error_response(
            code="invalid_docs_ref",
            message=str(exc),
            status=400,
            provider=self._provider_identity(),
            namespace=request.namespace or DOCS_NAMESPACE,
            object_ref=request.object_ref,
        )

    async def _execute(
        self,
        *,
        request: NamedServiceRequest,
        operation: str,
        claim: str | Sequence[str],
        payload: Mapping[str, Any],
        account_id: str,
    ) -> tuple[dict[str, Any] | None, NamedServiceResponse | None]:
        tool_name = f"named_services.{DOCS_NAMESPACE}.{request.operation}"
        if request.action:
            tool_name = f"{tool_name}.{request.action}"
        result = await self._execute_operation(
            operation=operation,
            claim=claim,
            tool_name=tool_name,
            payload=dict(payload or {}),
            account_id=_text(account_id),
        )
        if not isinstance(result, Mapping) or not result.get("ok"):
            envelope = dict(result or {}) if isinstance(result, Mapping) else {}
            return None, tool_error_response(
                envelope,
                request=request,
                namespace=DOCS_NAMESPACE,
                provider_identity=self._provider_identity(),
                default_code="docs_operation_failed",
                fallback_message="The document operation failed.",
            )
        ret = result.get("ret")
        return (dict(ret or {}) if isinstance(ret, Mapping) else {}), None

    def _provider_not_supported(
        self,
        request: NamedServiceRequest,
        parsed: Mapping[str, Any],
    ) -> NamedServiceResponse | None:
        provider = _text(parsed.get("provider"))
        if provider == GOOGLE_PROVIDER_KEY:
            return None
        return NamedServiceResponse.error_response(
            code="docs_provider_not_implemented",
            message=f"Document provider is not implemented: {provider}",
            status=501,
            provider=self._provider_identity(),
            namespace=request.namespace or DOCS_NAMESPACE,
            object_ref=request.object_ref,
        )

    async def provider_about(
        self, ctx: NamedServiceContext, request: NamedServiceRequest
    ) -> NamedServiceResponse:
        del ctx
        return NamedServiceResponse.ok_response(
            provider=self._provider_identity(),
            namespace=request.namespace or DOCS_NAMESPACE,
            extra={
                "title": "KDCube Documents",
                "description": (
                    "Provider-neutral document namespace. Google Docs is the "
                    "first connected-account provider."
                ),
                "workflow": [
                    "Call object.search to find a document by title.",
                    "Call object.get with its ref to read metadata and body text.",
                    "Call object.upsert or a declared object.action for bounded changes.",
                    "Use the comment actions to read or manage comment threads.",
                ],
                "providers": DOCS_PROVIDER_CATALOG,
                "schema": DOCS_SCHEMA,
            },
        )

    async def provider_capabilities(
        self, ctx: NamedServiceContext, request: NamedServiceRequest
    ) -> NamedServiceResponse:
        del ctx
        return NamedServiceResponse.ok_response(
            provider=self._provider_identity(),
            namespace=request.namespace or DOCS_NAMESPACE,
            capabilities={
                "list": True,
                "search": True,
                "get": True,
                "upsert": True,
                "delete": "comments_only",
                "actions": list(DOCS_ACTIONS),
                "providers": DOCS_PROVIDER_CATALOG,
                "grant_hints": DOCS_GRANT_HINTS,
                "connected_account_claims": DOCS_SCHEMA[
                    "connected_account_claims"
                ],
            },
        )

    async def object_schema(
        self, ctx: NamedServiceContext, request: NamedServiceRequest
    ) -> NamedServiceResponse:
        del ctx
        return NamedServiceResponse.ok_response(
            provider=self._provider_identity(),
            namespace=request.namespace or DOCS_NAMESPACE,
            extra={"schema": DOCS_SCHEMA},
        )

    async def event_resolve(
        self, ctx: NamedServiceContext, request: NamedServiceRequest
    ) -> NamedServiceResponse:
        del ctx
        try:
            parsed = parse_docs_ref(request.object_ref)
        except ValueError as exc:
            return self._invalid_ref(request, exc)
        unsupported = self._provider_not_supported(request, parsed)
        if unsupported is not None:
            return unsupported
        canonical_ref = document_ref(parsed["account_id"], parsed["document_id"])
        return NamedServiceResponse.ok_response(
            provider=self._provider_identity(),
            namespace=request.namespace or DOCS_NAMESPACE,
            object_ref=canonical_ref,
            extra={
                "event_source_id": f"named_services.{DOCS_NAMESPACE}",
                "object_ref": canonical_ref,
                "target_surface": "sdk.docs.snapshot",
            },
        )

    async def block_produce(
        self, ctx: NamedServiceContext, request: NamedServiceRequest
    ) -> NamedServiceResponse:
        target_value = request.payload.get("target")
        target = dict(target_value) if isinstance(target_value, Mapping) else {}
        object_ref = _text(
            request.object_ref
            or target.get("object_ref")
            or target.get("ref")
        )
        try:
            parsed = parse_docs_ref(object_ref)
        except ValueError as exc:
            return self._invalid_ref(request, exc)
        unsupported = self._provider_not_supported(request, parsed)
        if unsupported is not None:
            return unsupported

        snapshot = _snapshot_from_block_target(target)
        if not snapshot:
            metadata, error = await self._execute(
                request=request,
                operation="get",
                claim=DOCS_READ_CLAIM,
                payload={"document_ref": parsed["document_id"], "include_text": True},
                account_id=parsed["account_id"],
            )
            if error is not None:
                return NamedServiceResponse.ok_response(
                    provider=self._provider_identity(),
                    namespace=request.namespace or DOCS_NAMESPACE,
                    object_ref=object_ref,
                    extra={"blocks": []},
                    warnings=[
                        {
                            "code": "docs_block_produce_get_failed",
                            "message": (
                                error.error.message
                                if error.error is not None
                                else "Document metadata could not be loaded."
                            ),
                        }
                    ],
                )
            obj = _document_object(
                metadata or {},
                account_id=_text((metadata or {}).get("account_id"))
                or parsed["account_id"],
                provider=parsed["provider"],
            )
            snapshot = {
                "schema": DOCS_SNAPSHOT_SCHEMA,
                "object_ref": object_ref,
                "object_kind": DOCS_DOCUMENT_KIND,
                "object": obj,
                "comments": [],
                "materialization": {
                    "text_materialized": bool(obj.get("text")),
                    "complete_text": None,
                    "inventory_source": "provider_metadata_fallback",
                },
            }

        text = _snapshot_inventory_text(
            snapshot,
            object_ref=object_ref,
            target=target,
        )
        meta = target.get("meta") if isinstance(target.get("meta"), Mapping) else {}
        source_stats = {
            key: meta.get(key)
            for key in (
                "source_tokens",
                "source_text_symbols",
                "source_bytes",
                "source_line_count",
            )
            if meta.get(key) is not None
        }
        snapshot_object = (
            snapshot.get("object")
            if isinstance(snapshot.get("object"), Mapping)
            else {}
        )
        body_text = (
            snapshot_object.get("text")
            if isinstance(snapshot_object.get("text"), str)
            else ""
        )
        snapshot_comments = (
            snapshot.get("comments")
            if isinstance(snapshot.get("comments"), list)
            else []
        )
        source_stats.update(
            {
                "object_ref": object_ref,
                "object_kind": _text(snapshot.get("object_kind")),
                "snapshot_schema": _text(snapshot.get("schema")),
                "title": _text(snapshot_object.get("title")),
                "document_id": _text(snapshot_object.get("document_id")),
                "word_count": _word_count(body_text),
                "char_count": len(body_text),
                "comment_count": len(snapshot_comments),
            }
        )
        source_stats = {
            key: value for key, value in source_stats.items() if value is not None
        }
        block = {
            "turn": target.get("turn_id") or ctx.turn_id or "",
            "type": "react.tool.result",
            "call_id": target.get("tool_call_id") or "",
            "tool_id": "named_services.docs",
            "event_source_id": f"named_services.{DOCS_NAMESPACE}",
            "mime": "text/markdown",
            "path": object_ref,
            "text": text,
            "original_object_stats": source_stats,
            "meta": {
                "tool_call_id": target.get("tool_call_id") or "",
                "tool_id": target.get("tool_id") or "react.read",
                "turn_id": target.get("turn_id") or ctx.turn_id or "",
                "object_ref": object_ref,
                "source_namespace": DOCS_NAMESPACE,
                "materialized_path": target.get("logical_path")
                or target.get("path")
                or "",
                "physical_path": target.get("physical_path")
                or meta.get("physical_path")
                or "",
                "object_kind": _text(snapshot.get("object_kind")),
                "mime": DOCS_SNAPSHOT_MEDIA_TYPE,
                "render_policy": "docs.named_service.block_produce",
            },
        }
        LOGGER.info(
            "[docs.named_service.block_produce] produced object_ref=%s "
            "materialized_path=%s text_symbols=%s",
            object_ref,
            target.get("logical_path") or target.get("path") or "",
            len(text),
        )
        return NamedServiceResponse.ok_response(
            provider=self._provider_identity(),
            namespace=request.namespace or DOCS_NAMESPACE,
            object_ref=object_ref,
            extra={"blocks": [block]},
        )

    async def object_list(
        self, ctx: NamedServiceContext, request: NamedServiceRequest
    ) -> NamedServiceResponse:
        del ctx
        filters = dict(request.filters or {})
        return await self._search(
            request=request,
            query="",
            account_id=_text(filters.get("account_id") or request.payload.get("account_id")),
        )

    async def object_search(
        self, ctx: NamedServiceContext, request: NamedServiceRequest
    ) -> NamedServiceResponse:
        del ctx
        filters = dict(request.filters or {})
        return await self._search(
            request=request,
            query=_text(request.query),
            account_id=_text(filters.get("account_id") or request.payload.get("account_id")),
        )

    async def _search(
        self,
        *,
        request: NamedServiceRequest,
        query: str,
        account_id: str,
    ) -> NamedServiceResponse:
        filters = dict(request.filters or {})
        ret, error = await self._execute(
            request=request,
            operation="search",
            claim=DOCS_READ_CLAIM,
            payload={
                "query": query,
                "limit": _int(request.limit, default=20, minimum=1, maximum=50),
                "cursor": _text(request.cursor or filters.get("cursor")),
            },
            account_id=account_id,
        )
        if error is not None:
            return error
        ret = ret or {}
        resolved_account_id = _text(ret.get("account_id") or account_id)
        items = [
            _document_object(row, account_id=resolved_account_id)
            for row in ret.get("items") or []
            if isinstance(row, Mapping)
        ]
        return NamedServiceResponse.ok_response(
            provider=self._provider_identity(),
            namespace=request.namespace or DOCS_NAMESPACE,
            items=items,
            next_cursor=_text(ret.get("next_cursor")) or None,
            extra={
                "count": len(items),
                "query": query,
                "account_id": resolved_account_id,
            },
        )

    async def object_get(
        self, ctx: NamedServiceContext, request: NamedServiceRequest
    ) -> NamedServiceResponse | NamedServiceStreamResult:
        try:
            parsed = parse_docs_ref(request.object_ref)
        except ValueError as exc:
            return self._invalid_ref(request, exc)
        unsupported = self._provider_not_supported(request, parsed)
        if unsupported is not None:
            return unsupported
        if _is_materialization_request(request):
            return await self._materialize_snapshot(request=request, parsed=parsed)
        filters = dict(request.filters or {})
        include_text = filters.get("include_text")
        if include_text is None:
            include_text = request.payload.get("include_text")
        payload: dict[str, Any] = {"document_ref": parsed["document_id"]}
        if include_text is not None:
            payload["include_text"] = bool(include_text)
        ret, error = await self._execute(
            request=request,
            operation="get",
            claim=DOCS_READ_CLAIM,
            payload=payload,
            account_id=parsed["account_id"],
        )
        if error is not None:
            return error
        obj = _document_object(
            ret or {},
            account_id=parsed["account_id"],
            provider=parsed["provider"],
        )
        if not ctx.turn_id:
            url_info = await self._download_url(ctx, ref=obj["ref"])
            if url_info is not None:
                snapshot_download = {
                    "schema": DOCS_SNAPSHOT_SCHEMA,
                    "media_type": DOCS_SNAPSHOT_MEDIA_TYPE,
                    "filename": _snapshot_filename(parsed["document_id"]),
                    "download": {"encoding": "url", **url_info},
                }
                # Put the complete-artifact escape hatch before potentially
                # large inline body text in serialized MCP responses.
                obj = {"snapshot": snapshot_download, **obj}
        return NamedServiceResponse.ok_response(
            provider=self._provider_identity(),
            namespace=request.namespace or DOCS_NAMESPACE,
            object_ref=obj["ref"],
            object=obj,
        )

    async def _materialize_snapshot(
        self,
        *,
        request: NamedServiceRequest,
        parsed: Mapping[str, Any],
    ) -> NamedServiceResponse | NamedServiceStreamResult:
        document_id = _text(parsed.get("document_id"))
        metadata, error = await self._execute(
            request=request,
            operation="get",
            claim=DOCS_READ_CLAIM,
            payload={"document_ref": document_id, "include_text": True},
            account_id=_text(parsed.get("account_id")),
        )
        if error is not None:
            return error
        metadata = metadata or {}
        account_id = _text(metadata.get("account_id") or parsed.get("account_id"))
        obj = _document_object(
            metadata,
            account_id=account_id,
            provider=_text(parsed.get("provider")) or GOOGLE_PROVIDER_KEY,
        )

        comments: list[dict[str, Any]] = []
        comments_complete = True
        comment_result, comment_error = await self._execute(
            request=request,
            operation="list_comments",
            claim=DOCS_READ_CLAIM,
            payload={"document_ref": document_id, "include_resolved": False},
            account_id=account_id,
        )
        if comment_error is not None:
            comments_complete = False
        else:
            comments = [
                dict(row)
                for row in (comment_result or {}).get("comments") or []
                if isinstance(row, Mapping)
            ]

        object_ref = _text(obj.get("ref") or request.object_ref)
        body_text = obj.get("text") if isinstance(obj.get("text"), str) else ""
        snapshot = {
            "schema": DOCS_SNAPSHOT_SCHEMA,
            "object_ref": object_ref,
            "object_kind": DOCS_DOCUMENT_KIND,
            "object": obj,
            "comments": comments,
            "materialization": {
                "text_materialized": bool(body_text),
                "complete_text": True,
                "comments_materialized": comments_complete,
                "comment_count": len(comments),
                "word_count": _word_count(body_text),
                "char_count": len(body_text),
                "delivery": (
                    "The complete document body text is included. The comment "
                    "actions remain available for full comment-thread reads."
                ),
            },
        }
        response = NamedServiceResponse.ok_response(
            provider=self._provider_identity(),
            namespace=request.namespace or DOCS_NAMESPACE,
            object_ref=object_ref,
            attrs={
                "materialization": {
                    "schema": DOCS_SNAPSHOT_SCHEMA,
                    "media_type": DOCS_SNAPSHOT_MEDIA_TYPE,
                    "word_count": _word_count(body_text),
                    "char_count": len(body_text),
                    "comment_count": len(comments),
                    "complete_text": True,
                }
            },
        )
        return NamedServiceStreamResult(
            response=response,
            chunks=_json_chunks(snapshot),
            filename=_snapshot_filename(document_id),
            media_type=DOCS_SNAPSHOT_MEDIA_TYPE,
        )

    async def object_upsert(
        self, ctx: NamedServiceContext, request: NamedServiceRequest
    ) -> NamedServiceResponse:
        del ctx
        body = {**dict(request.payload or {}), **dict(request.object or {})}
        if not request.object_ref:
            operation = "create"
            payload = {
                key: body.get(key)
                for key in ("title", "initial_text")
                if body.get(key) is not None
            }
            if request.idempotency_key:
                payload["idempotency_key"] = request.idempotency_key
            account_id = _text(body.get("account_id"))
            parsed = None
        else:
            try:
                parsed = parse_docs_ref(request.object_ref)
            except ValueError as exc:
                return self._invalid_ref(request, exc)
            unsupported = self._provider_not_supported(request, parsed)
            if unsupported is not None:
                return unsupported
            account_id = parsed["account_id"]
            if body.get("replacements") is not None:
                operation = "replace_text"
                payload = {
                    "document_ref": parsed["document_id"],
                    "replacements": body.get("replacements"),
                }
            elif body.get("index") is not None:
                operation = "insert_text"
                payload = {
                    "document_ref": parsed["document_id"],
                    "text": body.get("text"),
                    "index": body.get("index"),
                }
            else:
                operation = "append_text"
                payload = {
                    "document_ref": parsed["document_id"],
                    "text": body.get("text"),
                }
            if request.idempotency_key:
                payload["idempotency_key"] = request.idempotency_key
        ret, error = await self._execute(
            request=request,
            operation=operation,
            claim=(DOCS_READ_CLAIM, DOCS_WRITE_CLAIM),
            payload=payload,
            account_id=account_id,
        )
        if error is not None:
            return error
        return self._mutation_response(request=request, ret=ret or {}, parsed=parsed)

    async def object_action(
        self, ctx: NamedServiceContext, request: NamedServiceRequest
    ) -> NamedServiceResponse:
        del ctx
        action = _text(request.action)
        if action not in DOCS_ACTIONS:
            return NamedServiceResponse.error_response(
                code="docs_action_not_supported",
                message=f"Unsupported document action: {action or '<missing>'}.",
                status=400,
                provider=self._provider_identity(),
                namespace=request.namespace or DOCS_NAMESPACE,
                object_ref=request.object_ref,
            )
        # import creates a new document; it takes no existing document ref.
        if action == ACTION_IMPORT:
            body = {**dict(request.payload or {}), **dict(request.object or {})}
            payload = {
                key: body.get(key)
                for key in ("title", "source_format", "content", "content_base64")
                if body.get(key) is not None
            }
            if request.idempotency_key:
                payload["idempotency_key"] = request.idempotency_key
            ret, error = await self._execute(
                request=request,
                operation=ACTION_IMPORT,
                claim=_action_claim(ACTION_IMPORT),
                payload=payload,
                account_id=_text(body.get("account_id")),
            )
            if error is not None:
                return error
            return self._mutation_response(request=request, ret=ret or {}, parsed=None)

        try:
            parsed = parse_docs_ref(request.object_ref)
        except ValueError as exc:
            return self._invalid_ref(request, exc)
        unsupported = self._provider_not_supported(request, parsed)
        if unsupported is not None:
            return unsupported
        payload = dict(request.payload or {})
        payload["document_ref"] = parsed["document_id"]
        if request.idempotency_key:
            payload["idempotency_key"] = request.idempotency_key
        ret, error = await self._execute(
            request=request,
            operation=action,
            claim=_action_claim(action),
            payload=payload,
            account_id=parsed["account_id"],
        )
        if error is not None:
            return error
        return self._mutation_response(request=request, ret=ret or {}, parsed=parsed)

    async def object_delete(
        self, ctx: NamedServiceContext, request: NamedServiceRequest
    ) -> NamedServiceResponse:
        del ctx
        try:
            parsed = parse_docs_ref(request.object_ref)
        except ValueError as exc:
            return self._invalid_ref(request, exc)
        unsupported = self._provider_not_supported(request, parsed)
        if unsupported is not None:
            return unsupported
        comment_id = _text(
            request.payload.get("comment_id") or (request.object or {}).get("comment_id")
        )
        if not comment_id:
            return NamedServiceResponse.error_response(
                code="docs_document_delete_not_supported",
                message=(
                    "Document-file deletion is not exposed. object.delete removes "
                    "one comment; pass payload.comment_id."
                ),
                status=400,
                provider=self._provider_identity(),
                namespace=request.namespace or DOCS_NAMESPACE,
                object_ref=request.object_ref,
            )
        ret, error = await self._execute(
            request=request,
            operation=ACTION_DELETE_COMMENT,
            claim=(DOCS_READ_CLAIM, DOCS_COMMENT_CLAIM),
            payload={
                "document_ref": parsed["document_id"],
                "comment_id": comment_id,
            },
            account_id=parsed["account_id"],
        )
        if error is not None:
            return error
        return self._mutation_response(request=request, ret=ret or {}, parsed=parsed)

    def _mutation_response(
        self,
        *,
        request: NamedServiceRequest,
        ret: Mapping[str, Any],
        parsed: Mapping[str, Any] | None,
    ) -> NamedServiceResponse:
        result = dict(ret or {})
        account_id = _text(result.get("account_id")) or _text(
            (parsed or {}).get("account_id")
        )
        provider = _text((parsed or {}).get("provider")) or GOOGLE_PROVIDER_KEY
        obj = _document_object(result, account_id=account_id, provider=provider)
        return NamedServiceResponse.ok_response(
            provider=self._provider_identity(),
            namespace=request.namespace or DOCS_NAMESPACE,
            object_ref=obj["ref"],
            object=obj,
            extra={"action": request.action or request.operation, "result": result},
        )


def make_docs_named_service_provider(
    *,
    execute_operation: ExecuteDocsOperation,
    bundle_id: str | None = None,
    file_url_factory: Any = None,
) -> DocsNamedServiceProvider:
    return DocsNamedServiceProvider(
        execute_operation=execute_operation,
        bundle_id=bundle_id,
        file_url_factory=file_url_factory,
    )


__all__ = [
    "ACTION_APPEND_TEXT",
    "ACTION_APPLY_TEXT_STYLE",
    "ACTION_CREATE_COMMENT",
    "ACTION_DELETE_COMMENT",
    "ACTION_EMBED_IMAGE",
    "ACTION_EXPORT",
    "ACTION_GET_COMMENT",
    "ACTION_IMPORT",
    "ACTION_INSERT_PAGE_BREAK",
    "ACTION_INSERT_TEXT",
    "ACTION_LIST_COMMENTS",
    "ACTION_REPLACE_TEXT",
    "ACTION_REPLY_COMMENT",
    "ACTION_RESOLVE_COMMENT",
    "DOCS_ACTIONS",
    "DOCS_COMMENT_CLAIM",
    "DOCS_CONNECTED_ACCOUNT_REQUIREMENTS",
    "DOCS_DOCUMENT_KIND",
    "DOCS_GRANT_HINTS",
    "DOCS_NAMESPACE",
    "DOCS_READ_CLAIM",
    "DOCS_SCHEMA",
    "DOCS_SNAPSHOT_MEDIA_TYPE",
    "DOCS_SNAPSHOT_SCHEMA",
    "DOCS_WRITE_CLAIM",
    "DocsNamedServiceProvider",
    "GOOGLE_PROVIDER_KEY",
    "document_ref",
    "docs_named_service_spec",
    "make_docs_named_service_provider",
    "parse_docs_ref",
]
