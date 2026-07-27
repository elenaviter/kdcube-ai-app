# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Provider-neutral spreadsheet named service.

The ``sheets`` namespace models spreadsheets and tabs. Google Sheets is the
first transport, but the named-service contract does not expose Google access
tokens or require consumers to use Google-specific MCP tools.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
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
    TRANSPORT_API,
    TRANSPORT_LOCAL,
    named_service_provider,
)
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.types import (
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


SHEETS_NAMESPACE = "sheets"
PROVIDER_ID = "sdk.integrations.sheets"
GOOGLE_PROVIDER_KEY = "google"
SHEETS_READ_CLAIM = "sheets:read"
SHEETS_WRITE_CLAIM = "sheets:write"

SHEETS_SPREADSHEET_KIND = "sheets.spreadsheet"
SHEETS_TAB_KIND = "sheets.tab"
SHEETS_TRANSPORTS = (TRANSPORT_LOCAL, TRANSPORT_API)

ACTION_UPDATE_VALUES = "update_values"
ACTION_APPEND_ROWS = "append_rows"
ACTION_CLEAR_VALUES = "clear_values"
ACTION_ADD_TAB = "add_tab"
ACTION_UPDATE_TAB = "update_tab"
ACTION_DELETE_TAB = "delete_tab"
ACTION_FORMAT_RANGE = "format_range"

SHEETS_ACTIONS = (
    ACTION_UPDATE_VALUES,
    ACTION_APPEND_ROWS,
    ACTION_CLEAR_VALUES,
    ACTION_ADD_TAB,
    ACTION_UPDATE_TAB,
    ACTION_DELETE_TAB,
    ACTION_FORMAT_RANGE,
)

ExecuteSheetsOperation = Callable[..., Awaitable[Mapping[str, Any]]]


SHEETS_SEARCH_FILTERS = {
    "account_id": {
        "type": "string",
        "description": (
            "Optional connected spreadsheet account id. Required when more "
            "than one connected account is eligible."
        ),
    },
    "cursor": {
        "type": "string",
        "description": "Optional next_cursor returned by an earlier search.",
    },
}

SHEETS_SEARCH_SCOPES = (
    NamedServiceSearchScope(
        namespace=SHEETS_NAMESPACE,
        label="spreadsheets",
        object_kind=SHEETS_SPREADSHEET_KIND,
        description=(
            "Find spreadsheets by title through an approved connected account. "
            "A blank query lists recently modified spreadsheets."
        ),
        filters_schema=SHEETS_SEARCH_FILTERS,
    ),
)

SHEETS_GRANT_HINTS = {
    "object.list": [SHEETS_READ_CLAIM],
    "object.search": [SHEETS_READ_CLAIM],
    "object.get": [SHEETS_READ_CLAIM],
    "object.upsert": [SHEETS_WRITE_CLAIM],
    "object.delete": [SHEETS_WRITE_CLAIM],
    "object.action": [SHEETS_WRITE_CLAIM],
    **{
        f"object.action.{action}": [SHEETS_WRITE_CLAIM]
        for action in SHEETS_ACTIONS
    },
}

SHEETS_CONNECTED_ACCOUNT_REQUIREMENTS = [
    {
        "provider_id": GOOGLE_PROVIDER_KEY,
        "provider_label": "Google",
        "claims": [SHEETS_READ_CLAIM, SHEETS_WRITE_CLAIM],
        "claim_labels": {
            SHEETS_READ_CLAIM: "read spreadsheets",
            SHEETS_WRITE_CLAIM: "edit spreadsheets",
        },
        "claims_by_operation": {
            "object.list": [SHEETS_READ_CLAIM],
            "object.search": [SHEETS_READ_CLAIM],
            "object.get": [SHEETS_READ_CLAIM],
            "object.upsert": [SHEETS_READ_CLAIM, SHEETS_WRITE_CLAIM],
            "object.delete": [SHEETS_READ_CLAIM, SHEETS_WRITE_CLAIM],
            "object.action": [SHEETS_READ_CLAIM, SHEETS_WRITE_CLAIM],
            **{
                f"object.action.{action}": [
                    SHEETS_READ_CLAIM,
                    SHEETS_WRITE_CLAIM,
                ]
                for action in SHEETS_ACTIONS
            },
        },
    }
]

SHEETS_PROVIDER_CATALOG = {
    GOOGLE_PROVIDER_KEY: {
        "provider_id": GOOGLE_PROVIDER_KEY,
        "label": "Google Sheets",
        "claims": {
            "read": SHEETS_READ_CLAIM,
            "write": SHEETS_WRITE_CLAIM,
        },
    },
}

SHEETS_SCHEMA = {
    "namespace": SHEETS_NAMESPACE,
    "refs": {
        "spreadsheet": (
            "sheets:<provider>:<account_id>:spreadsheet:<spreadsheet_id>"
        ),
        "tab": (
            "sheets:<provider>:<account_id>:spreadsheet:<spreadsheet_id>:"
            "tab:<sheet_id>"
        ),
    },
    "object_kinds": {
        SHEETS_SPREADSHEET_KIND: {
            "description": "One spreadsheet visible to a connected account.",
            "fields": [
                "ref",
                "provider",
                "account_id",
                "spreadsheet_id",
                "title",
                "web_url",
                "created_time",
                "modified_time",
                "tabs",
                "named_ranges",
            ],
        },
        SHEETS_TAB_KIND: {
            "description": "One stable tab inside a spreadsheet.",
            "fields": [
                "ref",
                "spreadsheet_ref",
                "provider",
                "account_id",
                "spreadsheet_id",
                "sheet_id",
                "title",
                "index",
                "row_count",
                "column_count",
            ],
        },
    },
    "search": {"filters": SHEETS_SEARCH_FILTERS},
    "get": {
        "description": (
            "Read a spreadsheet or tab by ref. A spreadsheet get returns "
            "metadata by default. Pass filters.ranges with explicit A1 ranges "
            "to read bounded cell values instead."
        ),
        "filters": [
            "ranges",
            "major_dimension",
            "value_render_option",
            "date_time_render_option",
        ],
    },
    "upsert": {
        "create": {
            "description": "Omit object_ref to create a spreadsheet.",
            "object": [
                "title",
                "first_tab_title",
                "initial_values",
                "value_input_option",
            ],
        },
        "spreadsheet": {
            "description": (
                "Use a spreadsheet ref and object.updates=[{range, values}] "
                "to replace bounded values."
            ),
            "object": ["updates", "value_input_option"],
        },
        "tab": {
            "description": "Use a tab ref to rename, resize, or freeze that tab.",
            "object": [
                "title",
                "rows",
                "columns",
                "frozen_rows",
                "frozen_columns",
            ],
        },
    },
    "actions": {
        ACTION_UPDATE_VALUES: {
            "description": "Replace values in explicit ranges.",
            "object_ref": "spreadsheet ref",
            "payload": ["updates", "value_input_option"],
        },
        ACTION_APPEND_ROWS: {
            "description": "Append rows after a logical table range.",
            "object_ref": "spreadsheet ref",
            "payload": ["range", "rows", "value_input_option", "idempotency_key"],
        },
        ACTION_CLEAR_VALUES: {
            "description": "Clear values from explicit A1 ranges.",
            "object_ref": "spreadsheet ref",
            "payload": ["ranges"],
        },
        ACTION_ADD_TAB: {
            "description": "Add a bounded tab to a spreadsheet.",
            "object_ref": "spreadsheet ref",
            "payload": ["title", "rows", "columns", "index"],
        },
        ACTION_UPDATE_TAB: {
            "description": "Rename, resize, or freeze a tab.",
            "object_ref": "tab ref",
            "payload": [
                "title",
                "rows",
                "columns",
                "frozen_rows",
                "frozen_columns",
            ],
        },
        ACTION_DELETE_TAB: {
            "description": "Delete one tab. Spreadsheet deletion is not exposed.",
            "object_ref": "tab ref",
            "payload": [],
        },
        ACTION_FORMAT_RANGE: {
            "description": "Apply bounded common formatting to a tab range.",
            "object_ref": "tab ref",
            "payload": [
                "range",
                "bold",
                "italic",
                "font_size",
                "text_color",
                "background_color",
                "horizontal_alignment",
                "vertical_alignment",
                "wrap_strategy",
                "number_format_type",
                "number_format_pattern",
                "border_style",
                "border_color",
            ],
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
    "grant_hints": SHEETS_GRANT_HINTS,
    "connected_account_claims": {
        GOOGLE_PROVIDER_KEY: {
            "read": SHEETS_READ_CLAIM,
            "write": SHEETS_WRITE_CLAIM,
        }
    },
}

SHEETS_INTRO = (
    "Use namespace `sheets` for user-connected spreadsheets. Search by title, "
    "get a spreadsheet ref to inspect metadata or bounded ranges, then use "
    "object.upsert or a declared object.action for explicit changes."
)

SHEETS_PRESENTATION = {
    "about": "Find, read, create, and edit spreadsheets you connect.",
    "third_party": "Google Sheets is the first provider behind this namespace.",
    "operations": {
        "provider.about": {
            "label": "Service overview",
            "description": "What the spreadsheet service does and how to use it.",
        },
        "provider.capabilities": {
            "label": "Capabilities",
            "description": "The operations and bounded actions this service supports.",
        },
        "object.list": {
            "label": "Recent spreadsheets",
            "description": "List recently modified spreadsheets.",
        },
        "object.search": {
            "label": "Search spreadsheets",
            "description": "Find spreadsheets by title.",
        },
        "object.get": {
            "label": "Read spreadsheet",
            "description": "Inspect metadata or read explicit ranges.",
        },
        "object.schema": {
            "label": "Spreadsheet schema",
            "description": "Read refs, fields, limits, and action payloads.",
        },
        "object.upsert": {
            "label": "Create or update spreadsheet",
            "description": "Create a spreadsheet or update explicit values/tab properties.",
        },
        "object.delete": {
            "label": "Delete spreadsheet tab",
            "description": "Delete one tab; spreadsheet-file deletion is not exposed.",
        },
    },
    "actions": {
        action: {"label": action.replace("_", " ").title(), **dict(meta)}
        for action, meta in SHEETS_SCHEMA["actions"].items()
    },
}


def _operations() -> dict[str, Any]:
    return {
        PROVIDER_ABOUT: {"transports": SHEETS_TRANSPORTS},
        PROVIDER_CAPABILITIES: {"transports": SHEETS_TRANSPORTS},
        OBJECT_LIST: {"transports": SHEETS_TRANSPORTS},
        OBJECT_SEARCH: {"transports": SHEETS_TRANSPORTS},
        OBJECT_GET: {"transports": SHEETS_TRANSPORTS},
        OBJECT_SCHEMA: {"transports": SHEETS_TRANSPORTS},
        OBJECT_UPSERT: {"transports": SHEETS_TRANSPORTS},
        OBJECT_DELETE: {"transports": SHEETS_TRANSPORTS},
        OBJECT_ACTION: {"transports": SHEETS_TRANSPORTS},
    }


def sheets_named_service_spec(*, bundle_id: str | None = None) -> NamedServiceProviderSpec:
    return NamedServiceProviderSpec(
        provider_id=PROVIDER_ID,
        bundle_id=bundle_id,
        namespace=SHEETS_NAMESPACE,
        refs=("sheets:*",),
        object_kinds=(SHEETS_SPREADSHEET_KIND, SHEETS_TAB_KIND),
        search_scopes=SHEETS_SEARCH_SCOPES,
        operations=_operations(),
        label="Spreadsheets",
        description=(
            "Provider-neutral spreadsheet namespace over user-connected accounts."
        ),
        intro=SHEETS_INTRO,
        metadata={
            "provider_catalog": SHEETS_PROVIDER_CATALOG,
            "grant_hints": SHEETS_GRANT_HINTS,
            "connected_accounts": SHEETS_CONNECTED_ACCOUNT_REQUIREMENTS,
            "canonical_ref": (
                "sheets:<provider>:<account_id>:spreadsheet:<spreadsheet_id>"
            ),
            "presentation": SHEETS_PRESENTATION,
            "actions": {
                name: str((meta or {}).get("description") or "")
                for name, meta in SHEETS_SCHEMA["actions"].items()
            },
            "object_kinds": {
                kind: str((meta or {}).get("description") or "")
                for kind, meta in SHEETS_SCHEMA["object_kinds"].items()
            },
        },
    )


def _text(value: Any) -> str:
    return str(value or "").strip()


def _int(value: Any, *, default: int = 0, minimum: int = 0, maximum: int = 10_000) -> int:
    try:
        parsed = int(value if value is not None else default)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def spreadsheet_ref(
    *, provider: str, account_id: str, spreadsheet_id: str
) -> str:
    return (
        f"{SHEETS_NAMESPACE}:{_text(provider) or GOOGLE_PROVIDER_KEY}:"
        f"{_text(account_id)}:spreadsheet:{_text(spreadsheet_id)}"
    )


def tab_ref(
    *,
    provider: str,
    account_id: str,
    spreadsheet_id: str,
    sheet_id: Any,
) -> str:
    parent_ref = spreadsheet_ref(
        provider=provider,
        account_id=account_id,
        spreadsheet_id=spreadsheet_id,
    )
    return f"{parent_ref}:tab:{_int(sheet_id)}"


def parse_sheets_ref(value: Any) -> dict[str, Any]:
    ref = _text(value)
    parts = ref.split(":")
    if len(parts) not in {5, 7} or parts[0].lower() != SHEETS_NAMESPACE:
        raise ValueError("Invalid sheets object ref.")
    if parts[3] != "spreadsheet" or not all(parts[index] for index in (1, 2, 4)):
        raise ValueError("Invalid sheets spreadsheet ref.")
    parsed: dict[str, Any] = {
        "ref": ref,
        "provider": parts[1].lower(),
        "account_id": parts[2],
        "spreadsheet_id": parts[4],
        "kind": "spreadsheet",
    }
    if len(parts) == 7:
        if parts[5] != "tab":
            raise ValueError("Invalid sheets tab ref.")
        try:
            parsed["sheet_id"] = int(parts[6])
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid sheets tab sheet_id.") from exc
        if parsed["sheet_id"] < 0:
            raise ValueError("Invalid sheets tab sheet_id.")
        parsed["kind"] = "tab"
    return parsed


def _spreadsheet_object(
    value: Mapping[str, Any],
    *,
    account_id: str,
    provider: str = GOOGLE_PROVIDER_KEY,
) -> dict[str, Any]:
    row = dict(value or {})
    spreadsheet_id = _text(row.get("spreadsheet_id"))
    ref = spreadsheet_ref(
        provider=provider,
        account_id=account_id,
        spreadsheet_id=spreadsheet_id,
    )
    tabs = [
        _tab_object(
            tab,
            account_id=account_id,
            spreadsheet_id=spreadsheet_id,
            provider=provider,
        )
        for tab in row.get("tabs") or []
        if isinstance(tab, Mapping)
    ]
    first_tab_raw = row.get("first_tab")
    first_tab = (
        _tab_object(
            first_tab_raw,
            account_id=account_id,
            spreadsheet_id=spreadsheet_id,
            provider=provider,
        )
        if isinstance(first_tab_raw, Mapping)
        else None
    )
    result = {
        **row,
        "ref": ref,
        "object_kind": SHEETS_SPREADSHEET_KIND,
        "provider": provider,
        "account_id": account_id,
        "spreadsheet_id": spreadsheet_id,
    }
    if "tabs" in row:
        result["tabs"] = tabs
    if first_tab is not None:
        result["first_tab"] = first_tab
    return result


def _tab_object(
    value: Mapping[str, Any],
    *,
    account_id: str,
    spreadsheet_id: str,
    provider: str = GOOGLE_PROVIDER_KEY,
) -> dict[str, Any]:
    row = dict(value or {})
    sheet_id = _int(row.get("sheet_id"))
    parent_ref = spreadsheet_ref(
        provider=provider,
        account_id=account_id,
        spreadsheet_id=spreadsheet_id,
    )
    return {
        **row,
        "ref": tab_ref(
            provider=provider,
            account_id=account_id,
            spreadsheet_id=spreadsheet_id,
            sheet_id=sheet_id,
        ),
        "object_kind": SHEETS_TAB_KIND,
        "spreadsheet_ref": parent_ref,
        "provider": provider,
        "account_id": account_id,
        "spreadsheet_id": spreadsheet_id,
        "sheet_id": sheet_id,
    }


@named_service_provider(
    provider_id=PROVIDER_ID,
    namespace=SHEETS_NAMESPACE,
    refs=("sheets:*",),
    object_kinds=(SHEETS_SPREADSHEET_KIND, SHEETS_TAB_KIND),
    search_scopes=SHEETS_SEARCH_SCOPES,
    operations=_operations(),
    label="Spreadsheets",
    description="Provider-neutral spreadsheet namespace over connected accounts.",
    intro=SHEETS_INTRO,
    metadata={
        "provider_catalog": SHEETS_PROVIDER_CATALOG,
        "grant_hints": SHEETS_GRANT_HINTS,
        "connected_accounts": SHEETS_CONNECTED_ACCOUNT_REQUIREMENTS,
        "presentation": SHEETS_PRESENTATION,
    },
)
class SheetsNamedServiceProvider(NamedServiceProvider):
    def __init__(
        self,
        *,
        execute_operation: ExecuteSheetsOperation,
        bundle_id: str | None = None,
    ) -> None:
        super().__init__(sheets_named_service_spec(bundle_id=bundle_id))
        self._execute_operation = execute_operation

    def _provider_identity(self) -> dict[str, Any]:
        return {"provider_id": PROVIDER_ID, "bundle_id": self.spec.bundle_id}

    def _invalid_ref(
        self, request: NamedServiceRequest, exc: Exception
    ) -> NamedServiceResponse:
        return NamedServiceResponse.error_response(
            code="invalid_sheets_ref",
            message=str(exc),
            status=400,
            provider=self._provider_identity(),
            namespace=request.namespace or SHEETS_NAMESPACE,
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
        tool_name = f"named_services.{SHEETS_NAMESPACE}.{request.operation}"
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
                namespace=SHEETS_NAMESPACE,
                provider_identity=self._provider_identity(),
                default_code="sheets_operation_failed",
                fallback_message="The spreadsheet operation failed.",
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
            code="sheets_provider_not_implemented",
            message=f"Spreadsheet provider is not implemented: {provider}",
            status=501,
            provider=self._provider_identity(),
            namespace=request.namespace or SHEETS_NAMESPACE,
            object_ref=request.object_ref,
        )

    async def provider_about(
        self, ctx: NamedServiceContext, request: NamedServiceRequest
    ) -> NamedServiceResponse:
        del ctx
        return NamedServiceResponse.ok_response(
            provider=self._provider_identity(),
            namespace=request.namespace or SHEETS_NAMESPACE,
            extra={
                "title": "KDCube Spreadsheets",
                "description": (
                    "Provider-neutral spreadsheet namespace. Google Sheets is "
                    "the first connected-account provider."
                ),
                "workflow": [
                    "Call object.search to find a spreadsheet by title.",
                    "Call object.get with its ref to inspect metadata.",
                    "Pass filters.ranges to object.get to read explicit A1 ranges.",
                    "Call object.upsert or a declared object.action for bounded changes.",
                ],
                "providers": SHEETS_PROVIDER_CATALOG,
                "schema": SHEETS_SCHEMA,
            },
        )

    async def provider_capabilities(
        self, ctx: NamedServiceContext, request: NamedServiceRequest
    ) -> NamedServiceResponse:
        del ctx
        return NamedServiceResponse.ok_response(
            provider=self._provider_identity(),
            namespace=request.namespace or SHEETS_NAMESPACE,
            capabilities={
                "list": True,
                "search": True,
                "get": True,
                "upsert": True,
                "delete": "tabs_only",
                "actions": list(SHEETS_ACTIONS),
                "providers": SHEETS_PROVIDER_CATALOG,
                "grant_hints": SHEETS_GRANT_HINTS,
                "connected_account_claims": SHEETS_SCHEMA[
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
            namespace=request.namespace or SHEETS_NAMESPACE,
            extra={"schema": SHEETS_SCHEMA},
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
            claim=SHEETS_READ_CLAIM,
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
            _spreadsheet_object(row, account_id=resolved_account_id)
            for row in ret.get("items") or []
            if isinstance(row, Mapping)
        ]
        return NamedServiceResponse.ok_response(
            provider=self._provider_identity(),
            namespace=request.namespace or SHEETS_NAMESPACE,
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
    ) -> NamedServiceResponse:
        del ctx
        try:
            parsed = parse_sheets_ref(request.object_ref)
        except ValueError as exc:
            return self._invalid_ref(request, exc)
        unsupported = self._provider_not_supported(request, parsed)
        if unsupported is not None:
            return unsupported
        filters = dict(request.filters or {})
        ranges = filters.get("ranges") or request.payload.get("ranges")
        if parsed["kind"] == "tab" and ranges:
            return NamedServiceResponse.error_response(
                code="sheets_spreadsheet_ref_required_for_ranges",
                message=(
                    "Range reads require the parent sheets spreadsheet ref; "
                    "put the tab title in each A1 range."
                ),
                status=400,
                provider=self._provider_identity(),
                namespace=request.namespace or SHEETS_NAMESPACE,
                object_ref=request.object_ref,
            )
        operation = "read" if ranges else "describe"
        payload: dict[str, Any] = {"spreadsheet_ref": parsed["spreadsheet_id"]}
        if ranges:
            payload.update(
                {
                    "ranges": ranges,
                    "major_dimension": filters.get("major_dimension"),
                    "value_render_option": filters.get("value_render_option"),
                    "date_time_render_option": filters.get(
                        "date_time_render_option"
                    ),
                }
            )
        ret, error = await self._execute(
            request=request,
            operation=operation,
            claim=SHEETS_READ_CLAIM,
            payload=payload,
            account_id=parsed["account_id"],
        )
        if error is not None:
            return error
        ret = ret or {}
        if parsed["kind"] == "tab":
            tabs = ret.get("tabs") or []
            tab = next(
                (
                    row
                    for row in tabs
                    if isinstance(row, Mapping)
                    and _int(row.get("sheet_id")) == parsed["sheet_id"]
                ),
                None,
            )
            if tab is None:
                return NamedServiceResponse.error_response(
                    code="sheets_tab_not_found",
                    message="The spreadsheet tab was not found.",
                    status=404,
                    provider=self._provider_identity(),
                    namespace=request.namespace or SHEETS_NAMESPACE,
                    object_ref=request.object_ref,
                )
            obj = _tab_object(
                tab,
                account_id=parsed["account_id"],
                spreadsheet_id=parsed["spreadsheet_id"],
                provider=parsed["provider"],
            )
        else:
            obj = _spreadsheet_object(
                ret,
                account_id=parsed["account_id"],
                provider=parsed["provider"],
            )
        return NamedServiceResponse.ok_response(
            provider=self._provider_identity(),
            namespace=request.namespace or SHEETS_NAMESPACE,
            object_ref=obj["ref"],
            object=obj,
        )

    async def object_upsert(
        self, ctx: NamedServiceContext, request: NamedServiceRequest
    ) -> NamedServiceResponse:
        del ctx
        body = {**dict(request.payload or {}), **dict(request.object or {})}
        if not request.object_ref:
            operation = "create_spreadsheet"
            payload = {
                key: body.get(key)
                for key in (
                    "title",
                    "first_tab_title",
                    "initial_values",
                    "value_input_option",
                )
                if body.get(key) is not None
            }
            if request.idempotency_key:
                payload["idempotency_key"] = request.idempotency_key
            account_id = _text(body.get("account_id"))
            parsed = None
        else:
            try:
                parsed = parse_sheets_ref(request.object_ref)
            except ValueError as exc:
                return self._invalid_ref(request, exc)
            unsupported = self._provider_not_supported(request, parsed)
            if unsupported is not None:
                return unsupported
            account_id = parsed["account_id"]
            if parsed["kind"] == "tab":
                operation = "update_tab"
                payload = {
                    key: body.get(key)
                    for key in (
                        "title",
                        "rows",
                        "columns",
                        "frozen_rows",
                        "frozen_columns",
                    )
                    if body.get(key) is not None
                }
                payload.update(
                    {
                        "spreadsheet_ref": parsed["spreadsheet_id"],
                        "sheet_id": parsed["sheet_id"],
                    }
                )
            else:
                operation = "update_values"
                payload = {
                    "spreadsheet_ref": parsed["spreadsheet_id"],
                    "updates": body.get("updates"),
                    "value_input_option": body.get("value_input_option"),
                }
        ret, error = await self._execute(
            request=request,
            operation=operation,
            claim=(SHEETS_READ_CLAIM, SHEETS_WRITE_CLAIM),
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
        if action not in SHEETS_ACTIONS:
            return NamedServiceResponse.error_response(
                code="sheets_action_not_supported",
                message=f"Unsupported spreadsheet action: {action or '<missing>'}.",
                status=400,
                provider=self._provider_identity(),
                namespace=request.namespace or SHEETS_NAMESPACE,
                object_ref=request.object_ref,
            )
        try:
            parsed = parse_sheets_ref(request.object_ref)
        except ValueError as exc:
            return self._invalid_ref(request, exc)
        unsupported = self._provider_not_supported(request, parsed)
        if unsupported is not None:
            return unsupported
        if action in {ACTION_UPDATE_TAB, ACTION_DELETE_TAB, ACTION_FORMAT_RANGE}:
            if parsed["kind"] != "tab":
                return NamedServiceResponse.error_response(
                    code="sheets_tab_ref_required",
                    message=f"Action {action} requires a sheets tab ref.",
                    status=400,
                    provider=self._provider_identity(),
                    namespace=request.namespace or SHEETS_NAMESPACE,
                    object_ref=request.object_ref,
                )
        elif parsed["kind"] != "spreadsheet":
            return NamedServiceResponse.error_response(
                code="sheets_spreadsheet_ref_required",
                message=f"Action {action} requires a sheets spreadsheet ref.",
                status=400,
                provider=self._provider_identity(),
                namespace=request.namespace or SHEETS_NAMESPACE,
                object_ref=request.object_ref,
            )
        payload = dict(request.payload or {})
        payload["spreadsheet_ref"] = parsed["spreadsheet_id"]
        if parsed["kind"] == "tab":
            payload["sheet_id"] = parsed["sheet_id"]
        if request.idempotency_key:
            payload["idempotency_key"] = request.idempotency_key
        ret, error = await self._execute(
            request=request,
            operation=action,
            claim=(SHEETS_READ_CLAIM, SHEETS_WRITE_CLAIM),
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
            parsed = parse_sheets_ref(request.object_ref)
        except ValueError as exc:
            return self._invalid_ref(request, exc)
        unsupported = self._provider_not_supported(request, parsed)
        if unsupported is not None:
            return unsupported
        if parsed["kind"] != "tab":
            return NamedServiceResponse.error_response(
                code="sheets_spreadsheet_delete_not_supported",
                message=(
                    "Spreadsheet-file deletion is not exposed. object.delete "
                    "accepts only a sheets tab ref."
                ),
                status=400,
                provider=self._provider_identity(),
                namespace=request.namespace or SHEETS_NAMESPACE,
                object_ref=request.object_ref,
            )
        ret, error = await self._execute(
            request=request,
            operation="delete_tab",
            claim=(SHEETS_READ_CLAIM, SHEETS_WRITE_CLAIM),
            payload={
                "spreadsheet_ref": parsed["spreadsheet_id"],
                "sheet_id": parsed["sheet_id"],
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
        spreadsheet_id = _text(result.get("spreadsheet_id")) or _text(
            (parsed or {}).get("spreadsheet_id")
        )
        provider = _text((parsed or {}).get("provider")) or GOOGLE_PROVIDER_KEY
        tab = result.get("tab")
        if isinstance(tab, Mapping):
            obj = _tab_object(
                tab,
                account_id=account_id,
                spreadsheet_id=spreadsheet_id,
                provider=provider,
            )
        elif _text((parsed or {}).get("kind")) == "tab":
            obj = _tab_object(
                {**result, "sheet_id": (parsed or {}).get("sheet_id")},
                account_id=account_id,
                spreadsheet_id=spreadsheet_id,
                provider=provider,
            )
            if "deleted_sheet_id" in result:
                obj["deleted"] = True
        else:
            obj = _spreadsheet_object(
                result,
                account_id=account_id,
                provider=provider,
            )
        return NamedServiceResponse.ok_response(
            provider=self._provider_identity(),
            namespace=request.namespace or SHEETS_NAMESPACE,
            object_ref=obj["ref"],
            object=obj,
            extra={"action": request.action or request.operation, "result": result},
        )


def make_sheets_named_service_provider(
    *,
    execute_operation: ExecuteSheetsOperation,
    bundle_id: str | None = None,
) -> SheetsNamedServiceProvider:
    return SheetsNamedServiceProvider(
        execute_operation=execute_operation,
        bundle_id=bundle_id,
    )


__all__ = [
    "ACTION_ADD_TAB",
    "ACTION_APPEND_ROWS",
    "ACTION_CLEAR_VALUES",
    "ACTION_DELETE_TAB",
    "ACTION_FORMAT_RANGE",
    "ACTION_UPDATE_TAB",
    "ACTION_UPDATE_VALUES",
    "GOOGLE_PROVIDER_KEY",
    "SHEETS_CONNECTED_ACCOUNT_REQUIREMENTS",
    "SHEETS_GRANT_HINTS",
    "SHEETS_NAMESPACE",
    "SHEETS_SCHEMA",
    "SHEETS_SPREADSHEET_KIND",
    "SHEETS_TAB_KIND",
    "SheetsNamedServiceProvider",
    "make_sheets_named_service_provider",
    "parse_sheets_ref",
    "sheets_named_service_spec",
    "spreadsheet_ref",
    "tab_ref",
]
