# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Serializable Google Docs operations for a trusted parent caller.

Like ``sheets_proxy`` this module owns no Connection Hub state and no ``@venv``
boundary. A trusted app resolves an access token, invokes one bounded operation,
and receives plain serializable data back. Unlike the Sheets proxy it needs no
heavy blocking dependency (no ``gspread``): it speaks raw REST to the Google
Docs API and Drive API over async ``httpx``, exactly as ``gmail_tools`` does, so
it runs directly on the proc event loop with no subprocess.

Operations span two Google APIs:
  - Docs API  (docs.googleapis.com/v1): read, create, and typed edits.
  - Drive API (www.googleapis.com/drive/v3): search, copy, export, import,
    comments.

Provider failures are normalized through the shared ``provider_errors`` helper,
so the service layer's ``credential_failure`` handling matches Gmail and Slack.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import httpx

from kdcube_ai_app.apps.chat.sdk.integrations.provider_errors import (
    ProviderFailure,
    provider_failure_from_exception,
    provider_failure_from_payload,
)

DOCS_API = "https://docs.googleapis.com/v1"
DRIVE_API = "https://www.googleapis.com/drive/v3"
DRIVE_UPLOAD_API = "https://www.googleapis.com/upload/drive/v3"

DOCS_MIME_TYPE = "application/vnd.google-apps.document"
_PROVIDER_ID = "google"
_SERVICE = "google_docs"
_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

MAX_SEARCH_RESULTS = 50
MAX_TEXT_CHARS = 200_000
MAX_EXPORT_BYTES = 10 * 1024 * 1024  # Drive files.export ceiling
MAX_IMPORT_BYTES = 10 * 1024 * 1024
MAX_TITLE_CHARS = 300
MAX_REPLACEMENTS = 50
MAX_COMMENT_CHARS = 20_000
MAX_COMMENTS = 100

_DOC_URL_RE = re.compile(
    r"https?://docs\.google\.com/document/(?:u/\d+/)?d/([A-Za-z0-9_-]+)"
)
_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# Drive export mime targets a caller may name by short alias.
_EXPORT_FORMATS: dict[str, tuple[str, str]] = {
    # alias: (mime_type, file extension)
    "pdf": ("application/pdf", "pdf"),
    "docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "docx",
    ),
    "odt": ("application/vnd.oasis.opendocument.text", "odt"),
    "rtf": ("application/rtf", "rtf"),
    "txt": ("text/plain", "txt"),
    "html": ("text/html", "html"),
    "epub": ("application/epub+zip", "epub"),
    "markdown": ("text/markdown", "md"),
    "md": ("text/markdown", "md"),
}

# Source mime a caller may hand to import for conversion into a Google Doc.
_IMPORT_FORMATS: dict[str, str] = {
    "txt": "text/plain",
    "text": "text/plain",
    "html": "text/html",
    "markdown": "text/markdown",
    "md": "text/markdown",
    "docx": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ),
    "odt": "application/vnd.oasis.opendocument.text",
    "rtf": "application/rtf",
}

_TEXT_STYLE_BOOL_FIELDS = {"bold", "italic", "underline", "strikethrough"}


class DocsValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code or "invalid_request")


class _DocsApiError(Exception):
    """Carries a normalized provider failure up to the central handler."""

    def __init__(self, failure: ProviderFailure) -> None:
        super().__init__(failure.message)
        self.failure = failure


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _document_id(document_ref: Any) -> str:
    value = _clean(document_ref)
    if not value:
        raise DocsValidationError(
            "document_ref_required",
            "document_ref must be a Google Docs URL or document id.",
        )
    match = _DOC_URL_RE.search(value)
    if match:
        return match.group(1)
    if _ID_RE.fullmatch(value):
        return value
    raise DocsValidationError(
        "invalid_document_ref",
        "document_ref must be a Google Docs URL or document id.",
    )


def _web_url(document_id: str) -> str:
    return f"https://docs.google.com/document/d/{document_id}/edit"


def _headers(access_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {access_token}"}


def _raise_for_status(
    response: httpx.Response, *, operation: str, mutating: bool
) -> None:
    if response.status_code < 400:
        return
    body: Mapping[str, Any] | None
    try:
        parsed = response.json()
        body = parsed if isinstance(parsed, Mapping) else None
    except Exception:
        body = None
    failure = provider_failure_from_payload(
        body,
        provider_status=response.status_code,
        provider=_PROVIDER_ID,
        service=_SERVICE,
        operation=operation,
        fallback="Google Docs operation failed.",
        mutating=mutating,
        retry_after=_clean(
            response.headers.get("Retry-After")
            or response.headers.get("retry-after")
        ),
    )
    raise _DocsApiError(failure)


# --------------------------------------------------------------------------- #
# Document text extraction (Docs API body -> plain text)
# --------------------------------------------------------------------------- #

def _extract_paragraph_text(paragraph: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for element in paragraph.get("elements") or []:
        if not isinstance(element, Mapping):
            continue
        text_run = element.get("textRun")
        if isinstance(text_run, Mapping):
            parts.append(str(text_run.get("content") or ""))
    return "".join(parts)


def _iter_structural_text(content: Any) -> Iterator[str]:
    """Yield readable text from paragraphs, tables, and table-of-contents blocks."""

    for block in content or []:
        if not isinstance(block, Mapping):
            continue
        paragraph = block.get("paragraph")
        if isinstance(paragraph, Mapping):
            yield _extract_paragraph_text(paragraph)
            continue
        table = block.get("table")
        if isinstance(table, Mapping):
            yield "[table]\n"
            for row in table.get("tableRows") or []:
                if not isinstance(row, Mapping):
                    continue
                for index, cell in enumerate(row.get("tableCells") or []):
                    if index:
                        yield " | "
                    if isinstance(cell, Mapping):
                        yield from _iter_structural_text(cell.get("content"))
                yield "\n"
            continue
        table_of_contents = block.get("tableOfContents")
        if isinstance(table_of_contents, Mapping):
            yield from _iter_structural_text(table_of_contents.get("content"))


def _iter_document_text(document: Mapping[str, Any]) -> Iterator[str]:
    tabs = document.get("tabs")

    def _walk_tab(tab: Mapping[str, Any]):
        properties = (
            tab.get("tabProperties")
            if isinstance(tab.get("tabProperties"), Mapping)
            else {}
        )
        title = _clean(properties.get("title"))
        if title:
            yield f"[tab: {title}]\n"
        document_tab = (
            tab.get("documentTab")
            if isinstance(tab.get("documentTab"), Mapping)
            else {}
        )
        body = document_tab.get("body")
        if isinstance(body, Mapping):
            yield from _iter_structural_text(body.get("content"))
        for child in tab.get("childTabs") or []:
            if isinstance(child, Mapping):
                yield from _walk_tab(child)

    if isinstance(tabs, list) and tabs:
        for tab in tabs:
            if isinstance(tab, Mapping):
                yield from _walk_tab(tab)
        return
    body = document.get("body")
    if isinstance(body, Mapping):
        yield from _iter_structural_text(body.get("content"))


def _extract_document_text(document: Mapping[str, Any], *, limit: int) -> str:
    chunks: list[str] = []
    remaining = max(0, limit)
    truncated = False
    for piece in _iter_document_text(document):
        if not piece:
            continue
        if len(piece) > remaining:
            chunks.append(piece[:remaining])
            truncated = True
            break
        chunks.append(piece)
        remaining -= len(piece)
        if remaining == 0:
            truncated = True
            break
    if truncated:
        chunks.append("\n[truncated]")
    return "".join(chunks)


def _default_document_body(document: Mapping[str, Any]) -> Mapping[str, Any]:
    body = document.get("body")
    if isinstance(body, Mapping):
        return body

    def _find(tabs: Any) -> Mapping[str, Any]:
        for tab in tabs or []:
            if not isinstance(tab, Mapping):
                continue
            document_tab = tab.get("documentTab")
            if isinstance(document_tab, Mapping) and isinstance(
                document_tab.get("body"), Mapping
            ):
                return document_tab["body"]
            nested = _find(tab.get("childTabs"))
            if nested:
                return nested
        return {}

    return _find(document.get("tabs"))


def _body_end_index(document: Mapping[str, Any]) -> int:
    """The insertion index just before the body's trailing newline."""
    body = _default_document_body(document)
    content = body.get("content")
    end = 1
    for block in content or []:
        if isinstance(block, Mapping):
            end = max(end, _int(block.get("endIndex"), default=end))
    return max(1, end - 1)


def _document_tabs(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def _walk(tabs: Any, *, parent_tab_id: str = "") -> None:
        for tab in tabs or []:
            if not isinstance(tab, Mapping):
                continue
            properties = (
                tab.get("tabProperties")
                if isinstance(tab.get("tabProperties"), Mapping)
                else {}
            )
            tab_id = _clean(properties.get("tabId"))
            records.append(
                {
                    "tab_id": tab_id,
                    "title": _clean(properties.get("title")),
                    "parent_tab_id": parent_tab_id,
                }
            )
            _walk(tab.get("childTabs"), parent_tab_id=tab_id)

    _walk(document.get("tabs"))
    return records


# --------------------------------------------------------------------------- #
# Read operations (docs:read)
# --------------------------------------------------------------------------- #

async def _search(
    client: httpx.AsyncClient, *, access_token: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    query = _clean(payload.get("query"))
    limit = max(1, min(_int(payload.get("limit"), default=20), MAX_SEARCH_RESULTS))
    escaped = query.replace("\\", "\\\\").replace("'", "\\'")
    cursor = _clean(payload.get("cursor"))

    async def _list(
        *,
        title_clause: str = "",
        page_token: str = "",
        page_size: int = limit,
    ) -> dict[str, Any]:
        clauses = [f"mimeType = '{DOCS_MIME_TYPE}'", "trashed = false"]
        if title_clause:
            clauses.append(title_clause)
        params = {
            "q": " and ".join(clauses),
            "pageSize": page_size,
            "orderBy": "modifiedTime desc",
            "spaces": "drive",
            "corpora": "user",
            "includeItemsFromAllDrives": "true",
            "supportsAllDrives": "true",
            "fields": (
                "nextPageToken,incompleteSearch,files(id,name,mimeType,parents,"
                "createdTime,modifiedTime,ownedByMe,webViewLink,"
                "capabilities(canCopy),owners(displayName,emailAddress))"
            ),
        }
        if page_token:
            params["pageToken"] = page_token
        response = await client.get(
            f"{DRIVE_API}/files", headers=_headers(access_token), params=params
        )
        _raise_for_status(response, operation="search", mutating=False)
        value = response.json()
        return dict(value) if isinstance(value, Mapping) else {}

    # Drive defines `name contains` as prefix matching. On the first page,
    # issue the exact-title query separately, then add prefix matches and
    # de-duplicate. Cursor pages continue only the prefix query so exact hits
    # are not repeated.
    exact_body: dict[str, Any] = {}
    if escaped and not cursor:
        exact_body = await _list(title_clause=f"name = '{escaped}'")

    rows: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    for body in (exact_body,):
        for row in body.get("files") or []:
            if not isinstance(row, Mapping):
                continue
            document_id = _clean(row.get("id"))
            if not document_id or document_id in seen_ids:
                continue
            seen_ids.add(document_id)
            rows.append(row)

    # Reserve the remainder of the requested page for title-prefix matches. This
    # keeps nextPageToken aligned with rows actually returned; fetching a full
    # prefix page and slicing after prepending exact hits would skip rows.
    prefix_body: dict[str, Any] = {}
    prefix_budget = limit if cursor else max(0, limit - len(rows))
    if prefix_budget:
        prefix_body = await _list(
            title_clause=f"name contains '{escaped}'" if escaped else "",
            page_token=cursor,
            page_size=prefix_budget,
        )
        for row in prefix_body.get("files") or []:
            if not isinstance(row, Mapping):
                continue
            document_id = _clean(row.get("id"))
            if not document_id or document_id in seen_ids:
                continue
            seen_ids.add(document_id)
            rows.append(row)

    items: list[dict[str, Any]] = []
    for row in rows[:limit]:
        document_id = _clean(row.get("id"))
        owners = [
            {
                "display_name": _clean(owner.get("displayName")),
                "email": _clean(owner.get("emailAddress")),
            }
            for owner in (row.get("owners") or [])
            if isinstance(owner, Mapping)
        ]
        items.append(
            {
                "document_id": document_id,
                "title": _clean(row.get("name")),
                "created_time": _clean(row.get("createdTime")),
                "modified_time": _clean(row.get("modifiedTime")),
                "owned_by_me": bool(row.get("ownedByMe")),
                "owners": owners,
                "web_url": _clean(row.get("webViewLink")) or _web_url(document_id),
                "mime_type": _clean(row.get("mimeType")) or DOCS_MIME_TYPE,
                "parent_ids": [
                    _clean(parent)
                    for parent in (row.get("parents") or [])
                    if _clean(parent)
                ],
                "copyable": bool((row.get("capabilities") or {}).get("canCopy")),
                "exact_title_match": bool(
                    query and _clean(row.get("name")).casefold() == query.casefold()
                ),
            }
        )
    exact_match_count = sum(1 for item in items if item["exact_title_match"])
    return {
        "items": items,
        "count": len(items),
        "next_cursor": _clean(prefix_body.get("nextPageToken")),
        "exact_match_count": exact_match_count,
        "incomplete_search": bool(
            exact_body.get("incompleteSearch")
            or prefix_body.get("incompleteSearch")
        ),
        "match_mode": (
            "exact_then_title_prefix"
            if query and not cursor
            else "title_prefix_page"
            if query
            else "recent_documents"
        ),
    }


async def _get(
    client: httpx.AsyncClient, *, access_token: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    document_id = _document_id(payload.get("document_ref"))
    response = await client.get(
        f"{DOCS_API}/documents/{document_id}",
        headers=_headers(access_token),
        params={"includeTabsContent": "true"},
    )
    _raise_for_status(response, operation="get", mutating=False)
    document = response.json()
    document = dict(document) if isinstance(document, Mapping) else {}
    include_text = payload.get("include_text")
    text = ""
    if include_text is None or bool(include_text):
        text = _extract_document_text(document, limit=MAX_TEXT_CHARS)
    tabs = _document_tabs(document)
    return {
        "document_id": document_id,
        "title": _clean(document.get("title")),
        "revision_id": _clean(document.get("revisionId")),
        "web_url": _web_url(document_id),
        "text": text,
        "tabs": tabs,
        "end_index": _body_end_index(document),
    }


async def _export(
    client: httpx.AsyncClient, *, access_token: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    document_id = _document_id(payload.get("document_ref"))
    fmt = _clean(payload.get("format")).lower() or "pdf"
    if fmt not in _EXPORT_FORMATS:
        raise DocsValidationError(
            "invalid_format",
            f"format must be one of: {', '.join(sorted(_EXPORT_FORMATS))}.",
        )
    mime_type, extension = _EXPORT_FORMATS[fmt]
    response = await client.get(
        f"{DRIVE_API}/files/{document_id}/export",
        headers=_headers(access_token),
        params={"mimeType": mime_type},
    )
    _raise_for_status(response, operation="export", mutating=False)
    content = response.content or b""
    if len(content) > MAX_EXPORT_BYTES:
        raise DocsValidationError(
            "export_too_large",
            f"Exported document is {len(content)} bytes; the limit is "
            f"{MAX_EXPORT_BYTES}. Export a smaller document or a lighter format.",
        )
    return {
        "document_id": document_id,
        "web_url": _web_url(document_id),
        "format": fmt,
        "mime_type": mime_type,
        "extension": extension,
        "byte_size": len(content),
        "content_base64": base64.b64encode(content).decode("ascii"),
    }


# --------------------------------------------------------------------------- #
# Write operations (docs:write) — typed batchUpdate, never raw JSON
# --------------------------------------------------------------------------- #

async def _fetch_document(
    client: httpx.AsyncClient, *, access_token: str, document_id: str
) -> dict[str, Any]:
    response = await client.get(
        f"{DOCS_API}/documents/{document_id}", headers=_headers(access_token)
    )
    _raise_for_status(response, operation="get", mutating=False)
    body = response.json()
    return dict(body) if isinstance(body, Mapping) else {}


async def _batch_update(
    client: httpx.AsyncClient,
    *,
    access_token: str,
    document_id: str,
    requests: list[dict[str, Any]],
    operation: str,
) -> dict[str, Any]:
    response = await client.post(
        f"{DOCS_API}/documents/{document_id}:batchUpdate",
        headers=_headers(access_token),
        json={"requests": requests},
    )
    _raise_for_status(response, operation=operation, mutating=True)
    body = response.json()
    return dict(body) if isinstance(body, Mapping) else {}


def _bounded_text(value: Any, *, field: str = "text") -> str:
    text = str(value if value is not None else "")
    if not text:
        raise DocsValidationError(f"{field}_required", f"{field} must not be empty.")
    if len(text) > MAX_TEXT_CHARS:
        raise DocsValidationError(
            "text_too_large",
            f"{field} is {len(text)} characters; the limit is {MAX_TEXT_CHARS}.",
        )
    return text


async def _create(
    client: httpx.AsyncClient, *, access_token: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    title = _clean(payload.get("title"))
    if not title or len(title) > MAX_TITLE_CHARS:
        raise DocsValidationError(
            "invalid_title",
            f"title is required and must be at most {MAX_TITLE_CHARS} characters.",
        )
    response = await client.post(
        f"{DOCS_API}/documents",
        headers=_headers(access_token),
        json={"title": title},
    )
    _raise_for_status(response, operation="create", mutating=True)
    document = response.json()
    document = dict(document) if isinstance(document, Mapping) else {}
    document_id = _clean(document.get("documentId"))
    result: dict[str, Any] = {
        "document_id": document_id,
        "title": _clean(document.get("title")) or title,
        "web_url": _web_url(document_id),
        "revision_id": _clean(document.get("revisionId")),
        "idempotency_key": _clean(payload.get("idempotency_key")),
    }
    initial_text = str(payload.get("initial_text") or "")
    if initial_text:
        _bounded_text(initial_text, field="initial_text")
        update = await _batch_update(
            client,
            access_token=access_token,
            document_id=document_id,
            requests=[{"insertText": {"location": {"index": 1}, "text": initial_text}}],
            operation="create",
        )
        result["revision_id"] = _clean(
            (update.get("writeControl") or {}).get("requiredRevisionId")
        ) or result["revision_id"]
        result["completed_stages"] = ["create", "write_initial_text"]
    return result


async def _copy(
    client: httpx.AsyncClient, *, access_token: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    source_document_id = _document_id(payload.get("document_ref"))
    title = _clean(payload.get("title"))
    if not title or len(title) > MAX_TITLE_CHARS:
        raise DocsValidationError(
            "invalid_title",
            f"title is required and must be at most {MAX_TITLE_CHARS} characters.",
        )
    body: dict[str, Any] = {"name": title}
    parent_id = _clean(payload.get("parent_id"))
    if parent_id:
        body["parents"] = [parent_id]
    response = await client.post(
        f"{DRIVE_API}/files/{source_document_id}/copy",
        headers=_headers(access_token),
        params={
            "supportsAllDrives": "true",
            "fields": (
                "id,name,mimeType,parents,createdTime,modifiedTime,ownedByMe,"
                "webViewLink,capabilities(canCopy)"
            ),
        },
        json=body,
    )
    _raise_for_status(response, operation="copy", mutating=True)
    row = response.json()
    row = dict(row) if isinstance(row, Mapping) else {}
    document_id = _clean(row.get("id"))
    return {
        "source_document_id": source_document_id,
        "document_id": document_id,
        "title": _clean(row.get("name")) or title,
        "web_url": _clean(row.get("webViewLink")) or _web_url(document_id),
        "mime_type": _clean(row.get("mimeType")) or DOCS_MIME_TYPE,
        "parent_ids": [
            _clean(parent)
            for parent in (row.get("parents") or [])
            if _clean(parent)
        ],
        "copied": True,
        "idempotency_key": _clean(payload.get("idempotency_key")),
    }


async def _insert_text(
    client: httpx.AsyncClient, *, access_token: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    document_id = _document_id(payload.get("document_ref"))
    text = _bounded_text(payload.get("text"))
    index = payload.get("index")
    if index is None:
        document = await _fetch_document(
            client, access_token=access_token, document_id=document_id
        )
        location = {"index": _body_end_index(document)}
    else:
        idx = _int(index, default=1)
        if idx < 1:
            raise DocsValidationError("invalid_index", "index must be >= 1.")
        location = {"index": idx}
    await _batch_update(
        client,
        access_token=access_token,
        document_id=document_id,
        requests=[{"insertText": {"location": location, "text": text}}],
        operation="insert_text",
    )
    return {
        "document_id": document_id,
        "web_url": _web_url(document_id),
        "inserted_chars": len(text),
        "index": location["index"],
    }


async def _append_text(
    client: httpx.AsyncClient, *, access_token: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    document_id = _document_id(payload.get("document_ref"))
    text = _bounded_text(payload.get("text"))
    document = await _fetch_document(
        client, access_token=access_token, document_id=document_id
    )
    index = _body_end_index(document)
    await _batch_update(
        client,
        access_token=access_token,
        document_id=document_id,
        requests=[{"insertText": {"location": {"index": index}, "text": text}}],
        operation="append_text",
    )
    return {
        "document_id": document_id,
        "web_url": _web_url(document_id),
        "appended_chars": len(text),
        "index": index,
    }


async def _replace_text(
    client: httpx.AsyncClient, *, access_token: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    document_id = _document_id(payload.get("document_ref"))
    raw = payload.get("replacements")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise DocsValidationError(
            "replacements_required",
            "replacements must be a list of {find, replace} objects.",
        )
    if not raw or len(raw) > MAX_REPLACEMENTS:
        raise DocsValidationError(
            "request_too_large" if raw else "replacements_required",
            f"Provide 1-{MAX_REPLACEMENTS} replacements.",
        )
    requests: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise DocsValidationError(
                "invalid_replacement", "Each replacement needs find and replace."
            )
        find = str(item.get("find") or "")
        if not find:
            raise DocsValidationError("invalid_replacement", "find must not be empty.")
        requests.append(
            {
                "replaceAllText": {
                    "containsText": {
                        "text": find,
                        "matchCase": bool(item.get("match_case")),
                    },
                    "replaceText": str(item.get("replace") or ""),
                }
            }
        )
    result = await _batch_update(
        client,
        access_token=access_token,
        document_id=document_id,
        requests=requests,
        operation="replace_text",
    )
    occurrences = 0
    for reply in result.get("replies") or []:
        if isinstance(reply, Mapping):
            occurrences += _int(
                (reply.get("replaceAllText") or {}).get("occurrencesChanged")
            )
    return {
        "document_id": document_id,
        "web_url": _web_url(document_id),
        "replacements": len(requests),
        "occurrences_changed": occurrences,
    }


async def _apply_text_style(
    client: httpx.AsyncClient, *, access_token: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    document_id = _document_id(payload.get("document_ref"))
    start = _int(payload.get("start_index"), default=-1)
    end = _int(payload.get("end_index"), default=-1)
    if start < 1 or end <= start:
        raise DocsValidationError(
            "invalid_range",
            "start_index must be >= 1 and end_index must be greater than start_index.",
        )
    style: dict[str, Any] = {}
    fields: list[str] = []
    for name in _TEXT_STYLE_BOOL_FIELDS:
        if payload.get(name) is not None:
            style[name] = bool(payload.get(name))
            fields.append(name)
    font_size = payload.get("font_size")
    if font_size is not None:
        size = _int(font_size)
        if size < 6 or size > 96:
            raise DocsValidationError("invalid_font_size", "font_size must be 6-96.")
        style["fontSize"] = {"magnitude": size, "unit": "PT"}
        fields.append("fontSize")
    link_url = _clean(payload.get("link_url"))
    if link_url:
        style["link"] = {"url": link_url}
        fields.append("link")
    if not fields:
        raise DocsValidationError(
            "style_required",
            "Provide at least one of: bold, italic, underline, strikethrough, "
            "font_size, link_url.",
        )
    await _batch_update(
        client,
        access_token=access_token,
        document_id=document_id,
        requests=[
            {
                "updateTextStyle": {
                    "range": {"startIndex": start, "endIndex": end},
                    "textStyle": style,
                    "fields": ",".join(fields),
                }
            }
        ],
        operation="apply_text_style",
    )
    return {
        "document_id": document_id,
        "web_url": _web_url(document_id),
        "range": {"start_index": start, "end_index": end},
        "applied_fields": sorted(fields),
    }


async def _insert_page_break(
    client: httpx.AsyncClient, *, access_token: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    document_id = _document_id(payload.get("document_ref"))
    index = payload.get("index")
    if index is None:
        document = await _fetch_document(
            client, access_token=access_token, document_id=document_id
        )
        location = {"index": _body_end_index(document)}
    else:
        idx = _int(index, default=1)
        if idx < 1:
            raise DocsValidationError("invalid_index", "index must be >= 1.")
        location = {"index": idx}
    await _batch_update(
        client,
        access_token=access_token,
        document_id=document_id,
        requests=[{"insertPageBreak": {"location": location}}],
        operation="insert_page_break",
    )
    return {
        "document_id": document_id,
        "web_url": _web_url(document_id),
        "index": location["index"],
    }


async def _embed_image(
    client: httpx.AsyncClient, *, access_token: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    document_id = _document_id(payload.get("document_ref"))
    image_uri = _clean(payload.get("image_uri"))
    if not image_uri.startswith(("http://", "https://")):
        raise DocsValidationError(
            "invalid_image_uri",
            "image_uri must be a public http(s) URL that Google can fetch once "
            "at insert time (PNG/JPEG/GIF, <=25MB, <=2000px per side).",
        )
    index = payload.get("index")
    if index is None:
        document = await _fetch_document(
            client, access_token=access_token, document_id=document_id
        )
        location = {"index": _body_end_index(document)}
    else:
        idx = _int(index, default=1)
        if idx < 1:
            raise DocsValidationError("invalid_index", "index must be >= 1.")
        location = {"index": idx}
    insert: dict[str, Any] = {"location": location, "uri": image_uri}
    width = payload.get("width_pt")
    height = payload.get("height_pt")
    if width is not None or height is not None:
        object_size: dict[str, Any] = {}
        if width is not None:
            object_size["width"] = {"magnitude": _int(width), "unit": "PT"}
        if height is not None:
            object_size["height"] = {"magnitude": _int(height), "unit": "PT"}
        insert["objectSize"] = object_size
    result = await _batch_update(
        client,
        access_token=access_token,
        document_id=document_id,
        requests=[{"insertInlineImage": insert}],
        operation="embed_image",
    )
    object_id = ""
    for reply in result.get("replies") or []:
        if isinstance(reply, Mapping):
            object_id = _clean(
                (reply.get("insertInlineImage") or {}).get("objectId")
            ) or object_id
    return {
        "document_id": document_id,
        "web_url": _web_url(document_id),
        "index": location["index"],
        "object_id": object_id,
    }


async def _import_document(
    client: httpx.AsyncClient, *, access_token: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    title = _clean(payload.get("title"))
    if not title or len(title) > MAX_TITLE_CHARS:
        raise DocsValidationError(
            "invalid_title",
            f"title is required and must be at most {MAX_TITLE_CHARS} characters.",
        )
    fmt = _clean(payload.get("source_format")).lower() or "markdown"
    if fmt not in _IMPORT_FORMATS:
        raise DocsValidationError(
            "invalid_source_format",
            f"source_format must be one of: {', '.join(sorted(_IMPORT_FORMATS))}.",
        )
    source_mime = _IMPORT_FORMATS[fmt]
    content_b64 = _clean(payload.get("content_base64"))
    if content_b64:
        try:
            raw = base64.b64decode(content_b64, validate=True)
        except Exception as exc:  # noqa: BLE001
            raise DocsValidationError(
                "invalid_content_base64", "content_base64 is not valid base64."
            ) from exc
    else:
        text = str(payload.get("content") or "")
        if not text:
            raise DocsValidationError(
                "content_required",
                "Provide content (text) or content_base64 (bytes) to import.",
            )
        raw = text.encode("utf-8")
    if len(raw) > MAX_IMPORT_BYTES:
        raise DocsValidationError(
            "import_too_large",
            f"Import payload is {len(raw)} bytes; the limit is {MAX_IMPORT_BYTES}.",
        )
    metadata = {"name": title, "mimeType": DOCS_MIME_TYPE}
    files = {
        "metadata": ("metadata", _json_dumps(metadata), "application/json"),
        "file": ("source", raw, source_mime),
    }
    response = await client.post(
        f"{DRIVE_UPLOAD_API}/files",
        headers=_headers(access_token),
        params={"uploadType": "multipart", "fields": "id,name,webViewLink"},
        files=files,
    )
    _raise_for_status(response, operation="import", mutating=True)
    body = response.json()
    body = dict(body) if isinstance(body, Mapping) else {}
    document_id = _clean(body.get("id"))
    return {
        "document_id": document_id,
        "title": _clean(body.get("name")) or title,
        "web_url": _clean(body.get("webViewLink")) or _web_url(document_id),
        "source_format": fmt,
        "byte_size": len(raw),
        "idempotency_key": _clean(payload.get("idempotency_key")),
    }


# --------------------------------------------------------------------------- #
# Comment operations (docs:comment) — Drive comments/replies API
# --------------------------------------------------------------------------- #

_COMMENT_FIELDS = (
    "id,content,anchor,resolved,createdTime,modifiedTime,"
    "author(displayName,emailAddress),quotedFileContent(value),"
    "replies(id,content,action,createdTime,author(displayName,emailAddress))"
)


def _comment_row(row: Mapping[str, Any]) -> dict[str, Any]:
    author = row.get("author") if isinstance(row.get("author"), Mapping) else {}
    quoted = (
        row.get("quotedFileContent")
        if isinstance(row.get("quotedFileContent"), Mapping)
        else {}
    )
    replies = [
        {
            "reply_id": _clean(reply.get("id")),
            "content": _clean(reply.get("content")),
            "action": _clean(reply.get("action")),
            "created_time": _clean(reply.get("createdTime")),
            "author": _clean((reply.get("author") or {}).get("displayName")),
        }
        for reply in (row.get("replies") or [])
        if isinstance(reply, Mapping)
    ]
    return {
        "comment_id": _clean(row.get("id")),
        "content": _clean(row.get("content")),
        "anchor": _clean(row.get("anchor")),
        "resolved": bool(row.get("resolved")),
        "quoted_text": _clean(quoted.get("value")),
        "created_time": _clean(row.get("createdTime")),
        "modified_time": _clean(row.get("modifiedTime")),
        "author": _clean(author.get("displayName")),
        "author_email": _clean(author.get("emailAddress")),
        "replies": replies,
    }


async def _list_comments(
    client: httpx.AsyncClient, *, access_token: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    document_id = _document_id(payload.get("document_ref"))
    limit = max(1, min(_int(payload.get("limit"), default=50), MAX_COMMENTS))
    params = {
        "pageSize": limit,
        "fields": f"nextPageToken,comments({_COMMENT_FIELDS})",
    }
    if bool(payload.get("include_resolved")):
        params["includeDeleted"] = "false"
    cursor = _clean(payload.get("cursor"))
    if cursor:
        params["pageToken"] = cursor
    response = await client.get(
        f"{DRIVE_API}/files/{document_id}/comments",
        headers=_headers(access_token),
        params=params,
    )
    _raise_for_status(response, operation="list_comments", mutating=False)
    body = response.json()
    items = [
        _comment_row(row)
        for row in (body.get("comments") or [])
        if isinstance(row, Mapping)
    ]
    if not bool(payload.get("include_resolved")):
        items = [item for item in items if not item["resolved"]]
    return {
        "document_id": document_id,
        "web_url": _web_url(document_id),
        "comments": items,
        "count": len(items),
        "next_cursor": _clean(body.get("nextPageToken")),
    }


async def _get_comment(
    client: httpx.AsyncClient, *, access_token: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    document_id = _document_id(payload.get("document_ref"))
    comment_id = _clean(payload.get("comment_id"))
    if not comment_id:
        raise DocsValidationError("comment_id_required", "comment_id is required.")
    response = await client.get(
        f"{DRIVE_API}/files/{document_id}/comments/{comment_id}",
        headers=_headers(access_token),
        params={"fields": _COMMENT_FIELDS},
    )
    _raise_for_status(response, operation="get_comment", mutating=False)
    body = response.json()
    return {
        "document_id": document_id,
        "web_url": _web_url(document_id),
        "comment": _comment_row(body if isinstance(body, Mapping) else {}),
    }


def _bounded_comment(value: Any) -> str:
    content = _clean(value)
    if not content:
        raise DocsValidationError("content_required", "content must not be empty.")
    if len(content) > MAX_COMMENT_CHARS:
        raise DocsValidationError(
            "content_too_large",
            f"content is {len(content)} characters; the limit is {MAX_COMMENT_CHARS}.",
        )
    return content


async def _create_comment(
    client: httpx.AsyncClient, *, access_token: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    document_id = _document_id(payload.get("document_ref"))
    content = _bounded_comment(payload.get("content"))
    body: dict[str, Any] = {"content": content}
    quoted = _clean(payload.get("quoted_text"))
    if quoted:
        body["quotedFileContent"] = {"value": quoted[:MAX_COMMENT_CHARS]}
    anchor = _clean(payload.get("anchor"))
    if anchor:
        body["anchor"] = anchor
    response = await client.post(
        f"{DRIVE_API}/files/{document_id}/comments",
        headers=_headers(access_token),
        params={"fields": _COMMENT_FIELDS},
        json=body,
    )
    _raise_for_status(response, operation="create_comment", mutating=True)
    result = response.json()
    return {
        "document_id": document_id,
        "web_url": _web_url(document_id),
        "comment": _comment_row(result if isinstance(result, Mapping) else {}),
        "idempotency_key": _clean(payload.get("idempotency_key")),
    }


async def _reply_comment(
    client: httpx.AsyncClient, *, access_token: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    document_id = _document_id(payload.get("document_ref"))
    comment_id = _clean(payload.get("comment_id"))
    if not comment_id:
        raise DocsValidationError("comment_id_required", "comment_id is required.")
    content = _bounded_comment(payload.get("content"))
    response = await client.post(
        f"{DRIVE_API}/files/{document_id}/comments/{comment_id}/replies",
        headers=_headers(access_token),
        params={"fields": "id,content,action,createdTime,author(displayName)"},
        json={"content": content},
    )
    _raise_for_status(response, operation="reply_comment", mutating=True)
    reply = response.json()
    reply = dict(reply) if isinstance(reply, Mapping) else {}
    return {
        "document_id": document_id,
        "web_url": _web_url(document_id),
        "comment_id": comment_id,
        "reply_id": _clean(reply.get("id")),
        "content": _clean(reply.get("content")),
        "idempotency_key": _clean(payload.get("idempotency_key")),
    }


async def _resolve_comment(
    client: httpx.AsyncClient, *, access_token: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    document_id = _document_id(payload.get("document_ref"))
    comment_id = _clean(payload.get("comment_id"))
    if not comment_id:
        raise DocsValidationError("comment_id_required", "comment_id is required.")
    # A comment is resolved by posting a reply carrying action=resolve.
    content = _clean(payload.get("content")) or "Resolved."
    response = await client.post(
        f"{DRIVE_API}/files/{document_id}/comments/{comment_id}/replies",
        headers=_headers(access_token),
        params={"fields": "id,action,content"},
        json={"content": content, "action": "resolve"},
    )
    _raise_for_status(response, operation="resolve_comment", mutating=True)
    reply = response.json()
    reply = dict(reply) if isinstance(reply, Mapping) else {}
    return {
        "document_id": document_id,
        "web_url": _web_url(document_id),
        "comment_id": comment_id,
        "reply_id": _clean(reply.get("id")),
        "action": _clean(reply.get("action")) or "resolve",
    }


async def _delete_comment(
    client: httpx.AsyncClient, *, access_token: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    document_id = _document_id(payload.get("document_ref"))
    comment_id = _clean(payload.get("comment_id"))
    if not comment_id:
        raise DocsValidationError("comment_id_required", "comment_id is required.")
    response = await client.delete(
        f"{DRIVE_API}/files/{document_id}/comments/{comment_id}",
        headers=_headers(access_token),
    )
    _raise_for_status(response, operation="delete_comment", mutating=True)
    return {
        "document_id": document_id,
        "web_url": _web_url(document_id),
        "deleted_comment_id": comment_id,
    }


def _json_dumps(value: Mapping[str, Any]) -> bytes:
    import json

    return json.dumps(value).encode("utf-8")


_OPERATIONS = {
    # read (docs:read)
    "search": _search,
    "get": _get,
    "export": _export,
    "list_comments": _list_comments,
    "get_comment": _get_comment,
    # write (docs:write)
    "create": _create,
    "copy": _copy,
    "insert_text": _insert_text,
    "append_text": _append_text,
    "replace_text": _replace_text,
    "apply_text_style": _apply_text_style,
    "insert_page_break": _insert_page_break,
    "embed_image": _embed_image,
    "import": _import_document,
    # comment (docs:comment)
    "create_comment": _create_comment,
    "reply_comment": _reply_comment,
    "resolve_comment": _resolve_comment,
    "delete_comment": _delete_comment,
}

MUTATING_OPERATIONS = frozenset(
    {
        "create",
        "copy",
        "insert_text",
        "append_text",
        "replace_text",
        "apply_text_style",
        "insert_page_break",
        "embed_image",
        "import",
        "create_comment",
        "reply_comment",
        "resolve_comment",
        "delete_comment",
    }
)


async def execute_google_docs_operation(
    *,
    operation: str,
    access_token: str,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one bounded Google Docs operation and return a serializable envelope.

    Returns ``{"ok": True, "error": None, "ret": {...}}`` on success or
    ``{"ok": False, "error": {...}, "ret": {...}}`` on a validation or provider
    failure. The provider failure ``ret`` carries the normalized category so the
    service layer can decide whether to refresh/reconnect.
    """
    op = _clean(operation)
    token = _clean(access_token)
    body = dict(payload or {})
    where = f"google_docs.{op or 'unknown'}"
    if not token:
        return {
            "ok": False,
            "error": {
                "code": "credential_missing_access_token",
                "message": "The connected Google credential has no access token.",
                "where": where,
                "managed": True,
            },
            "ret": {"outcome_unknown": False},
        }
    handler = _OPERATIONS.get(op)
    if handler is None:
        return {
            "ok": False,
            "error": {
                "code": "unsupported_operation",
                "message": f"Unsupported Google Docs operation: {op or '<empty>'}.",
                "where": where,
                "managed": True,
            },
            "ret": {"outcome_unknown": False},
        }
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            ret = await handler(client, access_token=token, payload=body)
        return {"ok": True, "error": None, "ret": ret}
    except DocsValidationError as exc:
        return {
            "ok": False,
            "error": {
                "code": exc.code,
                "message": str(exc),
                "where": where,
                "managed": True,
            },
            "ret": {"outcome_unknown": False},
        }
    except _DocsApiError as exc:
        return exc.failure.error_result(where=where)
    except httpx.HTTPError as exc:
        failure = provider_failure_from_exception(
            exc,
            provider=_PROVIDER_ID,
            service=_SERVICE,
            operation=op,
            fallback="Google Docs did not return a response.",
            mutating=op in MUTATING_OPERATIONS,
        )
        return failure.error_result(where=where)


__all__ = [
    "execute_google_docs_operation",
    "MUTATING_OPERATIONS",
    "MAX_SEARCH_RESULTS",
    "MAX_TEXT_CHARS",
    "MAX_EXPORT_BYTES",
    "DocsValidationError",
]
