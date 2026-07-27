# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Serializable Google Sheets operations for bundle-owned venv callers.

This module deliberately owns no Connection Hub state and no ``@venv``
boundary. A trusted app resolves an access token, invokes this proxy inside its
own bundle venv, and receives plain serializable data back.
"""

from __future__ import annotations

import re
import traceback
from collections.abc import Callable, Mapping, Sequence
from typing import Any


MAX_SEARCH_RESULTS = 50
MAX_RANGES = 20
MAX_WRITE_CELLS = 10_000
MAX_APPEND_ROWS = 1_000
MAX_TAB_CELLS = 1_000_000
MAX_TITLE_CHARS = 200
MAX_TAB_TITLE_CHARS = 100

_SHEETS_URL_RE = re.compile(
    r"https?://docs\.google\.com/spreadsheets/(?:u/\d+/)?d/([A-Za-z0-9_-]+)"
)
_A1_RECT_RE = re.compile(
    r"^(?:(?:'[^']*(?:''[^']*)*'|[^!]+)!)?\$?([A-Z]+)\$?(\d+):\$?([A-Z]+)\$?(\d+)$",
    re.IGNORECASE,
)
_HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")
_VALUE_INPUT_OPTIONS = {"RAW", "USER_ENTERED"}
_MAJOR_DIMENSIONS = {"ROWS", "COLUMNS"}
_VALUE_RENDER_OPTIONS = {
    "FORMATTED_VALUE",
    "UNFORMATTED_VALUE",
    "FORMULA",
}
_DATE_TIME_RENDER_OPTIONS = {"SERIAL_NUMBER", "FORMATTED_STRING"}
_HORIZONTAL_ALIGNMENTS = {"LEFT", "CENTER", "RIGHT"}
_VERTICAL_ALIGNMENTS = {"TOP", "MIDDLE", "BOTTOM"}
_WRAP_STRATEGIES = {"OVERFLOW_CELL", "LEGACY_WRAP", "CLIP", "WRAP"}
_NUMBER_FORMAT_TYPES = {
    "TEXT",
    "NUMBER",
    "PERCENT",
    "CURRENCY",
    "DATE",
    "TIME",
    "DATE_TIME",
    "SCIENTIFIC",
}
_BORDER_STYLES = {
    "DOTTED",
    "DASHED",
    "SOLID",
    "SOLID_MEDIUM",
    "SOLID_THICK",
    "DOUBLE",
}
_MUTATING_OPERATIONS = {
    "update_values",
    "append_rows",
    "clear_values",
    "create_spreadsheet",
    "add_tab",
    "update_tab",
    "delete_tab",
    "format_range",
}


class SheetsValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code or "invalid_request")


class SheetsProviderError(RuntimeError):
    def __init__(self, *, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = int(status or 0)
        self.code = str(code or "google_sheets_api_error")


class SheetsOperationFailure(RuntimeError):
    """Preserve the provider failure stage and any resource already created."""

    def __init__(
        self,
        *,
        stage: str,
        cause: Exception,
        mutation_attempted: bool,
        partial_result: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(str(cause).strip() or cause.__class__.__name__)
        self.stage = str(stage or "provider_call")
        self.mutation_attempted = bool(mutation_attempted)
        self.partial_result = dict(partial_result or {})


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _column_number(label: str) -> int:
    value = 0
    for char in str(label or "").upper():
        if not ("A" <= char <= "Z"):
            return 0
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value


def _estimated_range_cells(range_name: str) -> int | None:
    match = _A1_RECT_RE.match(_clean(range_name))
    if not match:
        return None
    start_col, start_row, end_col, end_row = match.groups()
    col_a = _column_number(start_col)
    col_b = _column_number(end_col)
    row_a = int(start_row)
    row_b = int(end_row)
    if col_b < col_a or row_b < row_a:
        raise SheetsValidationError("invalid_range", f"Invalid A1 range: {range_name}")
    return (col_b - col_a + 1) * (row_b - row_a + 1)


def _bounded_ranges(
    value: Any,
    *,
    max_cells: int | None,
    max_ranges: int | None = MAX_RANGES,
) -> list[str]:
    raw = (
        value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else [value]
    )
    ranges = [_clean(item) for item in raw if _clean(item)]
    if not ranges:
        raise SheetsValidationError(
            "range_required", "At least one A1 range is required."
        )
    if max_ranges is not None and len(ranges) > max_ranges:
        raise SheetsValidationError(
            "request_too_large",
            f"At most {max_ranges} ranges are allowed per call.",
        )
    estimated = 0
    for range_name in ranges:
        if len(range_name) > 512:
            raise SheetsValidationError("invalid_range", "An A1 range is too long.")
        cells = _estimated_range_cells(range_name)
        if cells is not None:
            estimated += cells
    if max_cells is not None and estimated > max_cells:
        raise SheetsValidationError(
            "request_too_large",
            f"The explicit ranges cover {estimated} cells; the limit is {max_cells}.",
        )
    return ranges


def _matrix(
    value: Any, *, max_cells: int, max_rows: int | None = None
) -> list[list[Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise SheetsValidationError("values_required", "values must be a list of rows.")
    rows: list[list[Any]] = []
    cell_count = 0
    for raw_row in value:
        if not isinstance(raw_row, Sequence) or isinstance(raw_row, (str, bytes)):
            raise SheetsValidationError(
                "invalid_values", "Each values item must be a row list."
            )
        row: list[Any] = []
        for cell in raw_row:
            if cell is not None and not isinstance(cell, (str, int, float, bool)):
                raise SheetsValidationError(
                    "invalid_cell_value",
                    "Cell values must be strings, numbers, booleans, or null.",
                )
            row.append(cell)
        rows.append(row)
        cell_count += len(row)
    if not rows:
        raise SheetsValidationError(
            "values_required", "At least one values row is required."
        )
    if max_rows is not None and len(rows) > max_rows:
        raise SheetsValidationError(
            "request_too_large",
            f"At most {max_rows} rows are allowed per call.",
        )
    if cell_count > max_cells:
        raise SheetsValidationError(
            "request_too_large",
            f"The request contains {cell_count} cells; the limit is {max_cells}.",
        )
    return rows


def _spreadsheet_id(spreadsheet_ref: Any) -> str:
    value = _clean(spreadsheet_ref)
    if not value:
        raise SheetsValidationError(
            "spreadsheet_ref_required",
            "spreadsheet_ref must be a Google Sheets URL or spreadsheet id.",
        )
    match = _SHEETS_URL_RE.search(value)
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]+", value):
        return value
    raise SheetsValidationError(
        "invalid_spreadsheet_ref",
        "spreadsheet_ref must be a Google Sheets URL or spreadsheet id.",
    )


def _web_url(spreadsheet_id: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit"


def _quote_tab_title(title: str) -> str:
    return "'" + str(title or "").replace("'", "''") + "'"


def _enum(value: Any, allowed: set[str], *, field: str, default: str) -> str:
    normalized = _clean(value).upper() or default
    if normalized not in allowed:
        raise SheetsValidationError(
            f"invalid_{field}",
            f"{field} must be one of: {', '.join(sorted(allowed))}.",
        )
    return normalized


def _authorize(access_token: str):
    import gspread
    from google.oauth2.credentials import Credentials

    return gspread.authorize(Credentials(token=access_token))


def _drive_files_list(
    *, access_token: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    import requests

    query = _clean(payload.get("query"))
    limit = max(1, min(_int(payload.get("limit"), default=20), MAX_SEARCH_RESULTS))
    escaped = query.replace("\\", "\\\\").replace("'", "\\'")
    clauses = [
        "mimeType = 'application/vnd.google-apps.spreadsheet'",
        "trashed = false",
    ]
    if escaped:
        clauses.append(f"name contains '{escaped}'")
    params = {
        "q": " and ".join(clauses),
        "pageSize": limit,
        "orderBy": "modifiedTime desc",
        "fields": (
            "nextPageToken,files(id,name,createdTime,modifiedTime,ownedByMe,"
            "webViewLink,driveId,owners(displayName,emailAddress))"
        ),
    }
    cursor = _clean(payload.get("cursor"))
    if cursor:
        params["pageToken"] = cursor
    response = requests.get(
        "https://www.googleapis.com/drive/v3/files",
        headers={"Authorization": f"Bearer {access_token}"},
        params=params,
        timeout=(10, 30),
    )
    if response.status_code >= 400:
        message = "Google Drive spreadsheet search failed."
        code = "google_drive_api_error"
        try:
            body = response.json()
            error = body.get("error") if isinstance(body, Mapping) else None
            if isinstance(error, Mapping):
                message = _clean(error.get("message")) or message
                code = _clean(error.get("status")) or code
        except Exception:
            pass
        raise SheetsProviderError(
            status=response.status_code, code=code, message=message
        )
    body = response.json()
    return dict(body or {}) if isinstance(body, Mapping) else {}


def _drive_create_spreadsheet(
    *, access_token: str, title: str
) -> dict[str, Any]:
    import requests

    response = requests.post(
        "https://www.googleapis.com/drive/v3/files",
        headers={"Authorization": f"Bearer {access_token}"},
        params={
            "supportsAllDrives": "true",
            "fields": "id,name,webViewLink",
        },
        json={
            "name": title,
            "mimeType": "application/vnd.google-apps.spreadsheet",
        },
        timeout=(10, 30),
    )
    response.raise_for_status()
    try:
        body = response.json()
    except Exception as exc:
        raise SheetsProviderError(
            status=response.status_code,
            code="google_drive_invalid_response",
            message="Google Drive created a response that could not be decoded.",
        ) from exc
    result = dict(body or {}) if isinstance(body, Mapping) else {}
    if not _clean(result.get("id")):
        raise SheetsProviderError(
            status=response.status_code,
            code="google_drive_invalid_response",
            message="Google Drive did not return the created spreadsheet id.",
        )
    return result


def _open_spreadsheet(client: Any, spreadsheet_ref: Any):
    spreadsheet_id = _spreadsheet_id(spreadsheet_ref)
    return spreadsheet_id, client.open_by_key(spreadsheet_id)


def _sheet_result(worksheet: Any) -> dict[str, Any]:
    return {
        "sheet_id": _int(getattr(worksheet, "id", 0)),
        "title": _clean(getattr(worksheet, "title", "")),
        "index": _int(getattr(worksheet, "index", 0)),
        "row_count": _int(getattr(worksheet, "row_count", 0)),
        "column_count": _int(getattr(worksheet, "col_count", 0)),
        "frozen_row_count": _int(getattr(worksheet, "frozen_row_count", 0)),
        "frozen_column_count": _int(getattr(worksheet, "frozen_col_count", 0)),
        "url": _clean(getattr(worksheet, "url", "")),
    }


def _search(*, access_token: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    body = _drive_files_list(access_token=access_token, payload=payload)
    items: list[dict[str, Any]] = []
    for row in body.get("files") or []:
        if not isinstance(row, Mapping):
            continue
        spreadsheet_id = _clean(row.get("id"))
        owners = []
        for owner in row.get("owners") or []:
            if isinstance(owner, Mapping):
                owners.append(
                    {
                        "display_name": _clean(owner.get("displayName")),
                        "email": _clean(owner.get("emailAddress")),
                    }
                )
        items.append(
            {
                "spreadsheet_id": spreadsheet_id,
                "title": _clean(row.get("name")),
                "created_time": _clean(row.get("createdTime")),
                "modified_time": _clean(row.get("modifiedTime")),
                "owned_by_me": bool(row.get("ownedByMe")),
                "drive_id": _clean(row.get("driveId")),
                "owners": owners,
                "web_url": _clean(row.get("webViewLink")) or _web_url(spreadsheet_id),
            }
        )
    return {
        "items": items,
        "count": len(items),
        "next_cursor": _clean(body.get("nextPageToken")),
    }


def _describe(*, client: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    spreadsheet_id, spreadsheet = _open_spreadsheet(
        client, payload.get("spreadsheet_ref")
    )
    metadata = spreadsheet.fetch_sheet_metadata(params={"includeGridData": False})
    metadata = dict(metadata or {}) if isinstance(metadata, Mapping) else {}
    properties = dict(metadata.get("properties") or {})
    tabs: list[dict[str, Any]] = []
    for row in metadata.get("sheets") or []:
        if not isinstance(row, Mapping):
            continue
        props = dict(row.get("properties") or {})
        grid = dict(props.get("gridProperties") or {})
        tabs.append(
            {
                "sheet_id": _int(props.get("sheetId")),
                "title": _clean(props.get("title")),
                "index": _int(props.get("index")),
                "sheet_type": _clean(props.get("sheetType")) or "GRID",
                "row_count": _int(grid.get("rowCount")),
                "column_count": _int(grid.get("columnCount")),
                "frozen_row_count": _int(grid.get("frozenRowCount")),
                "frozen_column_count": _int(grid.get("frozenColumnCount")),
                "hidden": bool(props.get("hidden")),
            }
        )
    named_ranges: list[dict[str, Any]] = []
    for row in metadata.get("namedRanges") or []:
        if isinstance(row, Mapping):
            named_ranges.append(
                {
                    "named_range_id": _clean(row.get("namedRangeId")),
                    "name": _clean(row.get("name")),
                    "range": dict(row.get("range") or {}),
                }
            )
    return {
        "spreadsheet_id": spreadsheet_id,
        "title": _clean(properties.get("title"))
        or _clean(getattr(spreadsheet, "title", "")),
        "locale": _clean(properties.get("locale")),
        "time_zone": _clean(properties.get("timeZone")),
        "web_url": _web_url(spreadsheet_id),
        "tabs": tabs,
        "named_ranges": named_ranges,
    }


def _read(*, client: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    spreadsheet_id, spreadsheet = _open_spreadsheet(
        client, payload.get("spreadsheet_ref")
    )
    ranges = _bounded_ranges(
        payload.get("ranges"),
        max_cells=None,
        max_ranges=None,
    )
    params = {
        "majorDimension": _enum(
            payload.get("major_dimension"),
            _MAJOR_DIMENSIONS,
            field="major_dimension",
            default="ROWS",
        ),
        "valueRenderOption": _enum(
            payload.get("value_render_option"),
            _VALUE_RENDER_OPTIONS,
            field="value_render_option",
            default="FORMATTED_VALUE",
        ),
        "dateTimeRenderOption": _enum(
            payload.get("date_time_render_option"),
            _DATE_TIME_RENDER_OPTIONS,
            field="date_time_render_option",
            default="FORMATTED_STRING",
        ),
    }
    raw = spreadsheet.values_batch_get(ranges, params=params)
    raw = dict(raw or {}) if isinstance(raw, Mapping) else {}
    value_ranges: list[dict[str, Any]] = []
    cell_count = 0
    row_count = 0
    for item in raw.get("valueRanges") or []:
        if not isinstance(item, Mapping):
            continue
        values = item.get("values") or []
        if not isinstance(values, list):
            values = []
        rows = [list(row) for row in values if isinstance(row, list)]
        row_count += len(rows)
        cell_count += sum(len(row) for row in rows)
        value_ranges.append(
            {
                "range": _clean(item.get("range")),
                "major_dimension": _clean(item.get("majorDimension"))
                or params["majorDimension"],
                "values": rows,
            }
        )
    return {
        "spreadsheet_id": spreadsheet_id,
        "web_url": _web_url(spreadsheet_id),
        "ranges": value_ranges,
        "range_count": len(value_ranges),
        "row_count": row_count,
        "cell_count": cell_count,
    }


def _update_values(*, client: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    spreadsheet_id, spreadsheet = _open_spreadsheet(
        client, payload.get("spreadsheet_ref")
    )
    raw_updates = payload.get("updates")
    if not isinstance(raw_updates, Sequence) or isinstance(raw_updates, (str, bytes)):
        raise SheetsValidationError(
            "updates_required", "updates must be a list of {range, values} objects."
        )
    if not raw_updates or len(raw_updates) > MAX_RANGES:
        raise SheetsValidationError(
            "request_too_large" if raw_updates else "updates_required",
            f"Provide 1-{MAX_RANGES} value updates.",
        )
    data = []
    total_cells = 0
    affected_ranges = []
    for row in raw_updates:
        if not isinstance(row, Mapping):
            raise SheetsValidationError(
                "invalid_updates", "Each update must contain range and values."
            )
        range_name = _bounded_ranges([row.get("range")], max_cells=MAX_WRITE_CELLS)[0]
        values = _matrix(row.get("values"), max_cells=MAX_WRITE_CELLS)
        total_cells += sum(len(item) for item in values)
        affected_ranges.append(range_name)
        data.append({"range": range_name, "majorDimension": "ROWS", "values": values})
    if total_cells > MAX_WRITE_CELLS:
        raise SheetsValidationError(
            "request_too_large",
            f"The updates contain {total_cells} cells; the limit is {MAX_WRITE_CELLS}.",
        )
    value_input_option = _enum(
        payload.get("value_input_option"),
        _VALUE_INPUT_OPTIONS,
        field="value_input_option",
        default="USER_ENTERED",
    )
    result = spreadsheet.values_batch_update(
        {
            "valueInputOption": value_input_option,
            "includeValuesInResponse": False,
            "data": data,
        }
    )
    result = dict(result or {}) if isinstance(result, Mapping) else {}
    return {
        "spreadsheet_id": spreadsheet_id,
        "web_url": _web_url(spreadsheet_id),
        "affected_ranges": affected_ranges,
        "updated_rows": _int(result.get("totalUpdatedRows")),
        "updated_columns": _int(result.get("totalUpdatedColumns")),
        "updated_cells": _int(result.get("totalUpdatedCells"), default=total_cells),
        "updated_sheets": _int(result.get("totalUpdatedSheets")),
    }


def _append_rows(*, client: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    spreadsheet_id, spreadsheet = _open_spreadsheet(
        client, payload.get("spreadsheet_ref")
    )
    range_name = _bounded_ranges([payload.get("range")], max_cells=MAX_WRITE_CELLS)[0]
    rows = _matrix(
        payload.get("rows"),
        max_cells=MAX_WRITE_CELLS,
        max_rows=MAX_APPEND_ROWS,
    )
    value_input_option = _enum(
        payload.get("value_input_option"),
        _VALUE_INPUT_OPTIONS,
        field="value_input_option",
        default="USER_ENTERED",
    )
    result = spreadsheet.values_append(
        range_name,
        params={
            "valueInputOption": value_input_option,
            "insertDataOption": "INSERT_ROWS",
            "includeValuesInResponse": False,
        },
        body={"majorDimension": "ROWS", "values": rows},
    )
    result = dict(result or {}) if isinstance(result, Mapping) else {}
    updates = dict(result.get("updates") or {})
    return {
        "spreadsheet_id": spreadsheet_id,
        "web_url": _web_url(spreadsheet_id),
        "table_range": _clean(result.get("tableRange")),
        "updated_range": _clean(updates.get("updatedRange")),
        "updated_rows": _int(updates.get("updatedRows"), default=len(rows)),
        "updated_columns": _int(updates.get("updatedColumns")),
        "updated_cells": _int(
            updates.get("updatedCells"),
            default=sum(len(row) for row in rows),
        ),
        "idempotency_key": _clean(payload.get("idempotency_key")),
    }


def _clear_values(*, client: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    spreadsheet_id, spreadsheet = _open_spreadsheet(
        client, payload.get("spreadsheet_ref")
    )
    ranges = _bounded_ranges(payload.get("ranges"), max_cells=MAX_WRITE_CELLS)
    result = spreadsheet.values_batch_clear(params={}, body={"ranges": ranges})
    result = dict(result or {}) if isinstance(result, Mapping) else {}
    return {
        "spreadsheet_id": spreadsheet_id,
        "web_url": _web_url(spreadsheet_id),
        "cleared_ranges": list(result.get("clearedRanges") or ranges),
    }


def _create_spreadsheet(
    *,
    client: Any,
    access_token: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    title = _clean(payload.get("title"))
    if not title or len(title) > MAX_TITLE_CHARS:
        raise SheetsValidationError(
            "invalid_title",
            f"title is required and must be at most {MAX_TITLE_CHARS} characters.",
        )
    first_tab_title = _clean(payload.get("first_tab_title")) or "Sheet1"
    if len(first_tab_title) > MAX_TAB_TITLE_CHARS:
        raise SheetsValidationError(
            "invalid_tab_title",
            f"first_tab_title must be at most {MAX_TAB_TITLE_CHARS} characters.",
        )
    initial_values = payload.get("initial_values")
    values: list[list[Any]] | None = None
    value_input_option = "USER_ENTERED"
    if initial_values not in (None, []):
        values = _matrix(initial_values, max_cells=MAX_WRITE_CELLS)
        value_input_option = _enum(
            payload.get("value_input_option"),
            _VALUE_INPUT_OPTIONS,
            field="value_input_option",
            default="USER_ENTERED",
        )

    try:
        drive_file = _drive_create_spreadsheet(
            access_token=access_token,
            title=title,
        )
    except Exception as exc:
        raise SheetsOperationFailure(
            stage="create_file",
            cause=exc,
            mutation_attempted=True,
        ) from exc

    spreadsheet_id = _clean(drive_file.get("id"))
    partial_result: dict[str, Any] = {
        "resource_created": True,
        "spreadsheet_id": spreadsheet_id,
        "title": _clean(drive_file.get("name")) or title,
        "web_url": _clean(drive_file.get("webViewLink"))
        or _web_url(spreadsheet_id),
        "completed_stages": ["create_file"],
    }
    try:
        _spreadsheet_id_value, spreadsheet = _open_spreadsheet(
            client, spreadsheet_id
        )
    except Exception as exc:
        raise SheetsOperationFailure(
            stage="open_created_spreadsheet",
            cause=exc,
            mutation_attempted=False,
            partial_result=partial_result,
        ) from exc
    try:
        worksheet = spreadsheet.sheet1
    except Exception as exc:
        raise SheetsOperationFailure(
            stage="read_first_tab",
            cause=exc,
            mutation_attempted=False,
            partial_result=partial_result,
        ) from exc

    partial_result["first_tab"] = _sheet_result(worksheet)
    if first_tab_title != _clean(getattr(worksheet, "title", "")):
        try:
            worksheet.update_title(first_tab_title)
        except Exception as exc:
            raise SheetsOperationFailure(
                stage="rename_first_tab",
                cause=exc,
                mutation_attempted=True,
                partial_result=partial_result,
            ) from exc
        partial_result["first_tab"] = _sheet_result(worksheet)
        partial_result["completed_stages"].append("rename_first_tab")

    updated_range = ""
    if values is not None:
        try:
            result = spreadsheet.values_update(
                f"{_quote_tab_title(first_tab_title)}!A1",
                params={"valueInputOption": value_input_option},
                body={"majorDimension": "ROWS", "values": values},
            )
        except Exception as exc:
            raise SheetsOperationFailure(
                stage="write_initial_values",
                cause=exc,
                mutation_attempted=True,
                partial_result=partial_result,
            ) from exc
        if isinstance(result, Mapping):
            updated_range = _clean(result.get("updatedRange"))
        partial_result["completed_stages"].append("write_initial_values")
    return {
        "spreadsheet_id": spreadsheet_id,
        "title": title,
        "web_url": _clean(getattr(spreadsheet, "url", "")) or _web_url(spreadsheet_id),
        "first_tab": _sheet_result(worksheet),
        "updated_range": updated_range,
        "idempotency_key": _clean(payload.get("idempotency_key")),
    }


def _add_tab(*, client: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    spreadsheet_id, spreadsheet = _open_spreadsheet(
        client, payload.get("spreadsheet_ref")
    )
    title = _clean(payload.get("title"))
    rows = max(1, _int(payload.get("rows"), default=1_000))
    columns = max(1, _int(payload.get("columns"), default=26))
    if not title or len(title) > MAX_TAB_TITLE_CHARS:
        raise SheetsValidationError(
            "invalid_tab_title",
            f"title is required and must be at most {MAX_TAB_TITLE_CHARS} characters.",
        )
    if rows * columns > MAX_TAB_CELLS:
        raise SheetsValidationError(
            "request_too_large",
            f"A new tab may contain at most {MAX_TAB_CELLS} cells.",
        )
    index = payload.get("index")
    worksheet = spreadsheet.add_worksheet(
        title=title,
        rows=rows,
        cols=columns,
        index=_int(index) if index is not None else None,
    )
    return {
        "spreadsheet_id": spreadsheet_id,
        "web_url": _web_url(spreadsheet_id),
        "tab": _sheet_result(worksheet),
    }


def _update_tab(*, client: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    spreadsheet_id, spreadsheet = _open_spreadsheet(
        client, payload.get("spreadsheet_ref")
    )
    sheet_id = _int(payload.get("sheet_id"), default=-1)
    if sheet_id < 0:
        raise SheetsValidationError(
            "sheet_id_required", "sheet_id must be a non-negative integer."
        )
    worksheet = spreadsheet.get_worksheet_by_id(sheet_id)
    title = _clean(payload.get("title"))
    rows = payload.get("rows")
    columns = payload.get("columns")
    frozen_rows = payload.get("frozen_rows")
    frozen_columns = payload.get("frozen_columns")
    if not any(
        value is not None and value != ""
        for value in (title, rows, columns, frozen_rows, frozen_columns)
    ):
        raise SheetsValidationError(
            "tab_update_required",
            "Provide title, rows, columns, frozen_rows, or frozen_columns.",
        )
    if title:
        if len(title) > MAX_TAB_TITLE_CHARS:
            raise SheetsValidationError(
                "invalid_tab_title",
                f"title must be at most {MAX_TAB_TITLE_CHARS} characters.",
            )
        worksheet.update_title(title)
    if rows is not None or columns is not None:
        target_rows = _int(rows, default=_int(getattr(worksheet, "row_count", 0)))
        target_columns = _int(columns, default=_int(getattr(worksheet, "col_count", 0)))
        if (
            target_rows < 1
            or target_columns < 1
            or target_rows * target_columns > MAX_TAB_CELLS
        ):
            raise SheetsValidationError(
                "invalid_tab_size",
                f"Tab dimensions must be positive and cover at most {MAX_TAB_CELLS} cells.",
            )
        worksheet.resize(
            rows=target_rows if rows is not None else None,
            cols=target_columns if columns is not None else None,
        )
    if frozen_rows is not None or frozen_columns is not None:
        freeze_rows = _int(
            frozen_rows, default=_int(getattr(worksheet, "frozen_row_count", 0))
        )
        freeze_columns = _int(
            frozen_columns,
            default=_int(getattr(worksheet, "frozen_col_count", 0)),
        )
        if freeze_rows < 0 or freeze_columns < 0:
            raise SheetsValidationError(
                "invalid_frozen_count",
                "frozen_rows and frozen_columns must be zero or positive.",
            )
        worksheet.freeze(
            rows=freeze_rows if frozen_rows is not None else None,
            cols=freeze_columns if frozen_columns is not None else None,
        )
    return {
        "spreadsheet_id": spreadsheet_id,
        "web_url": _web_url(spreadsheet_id),
        "tab": _sheet_result(worksheet),
    }


def _delete_tab(*, client: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    spreadsheet_id, spreadsheet = _open_spreadsheet(
        client, payload.get("spreadsheet_ref")
    )
    sheet_id = _int(payload.get("sheet_id"), default=-1)
    if sheet_id < 0:
        raise SheetsValidationError(
            "sheet_id_required", "sheet_id must be a non-negative integer."
        )
    spreadsheet.del_worksheet_by_id(sheet_id)
    return {
        "spreadsheet_id": spreadsheet_id,
        "web_url": _web_url(spreadsheet_id),
        "deleted_sheet_id": sheet_id,
    }


def _color(value: Any, *, field: str) -> dict[str, float] | None:
    text = _clean(value)
    if not text:
        return None
    if not _HEX_COLOR_RE.fullmatch(text):
        raise SheetsValidationError(f"invalid_{field}", f"{field} must be #RRGGBB.")
    return {
        "red": int(text[1:3], 16) / 255.0,
        "green": int(text[3:5], 16) / 255.0,
        "blue": int(text[5:7], 16) / 255.0,
    }


def _format_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    text_format: dict[str, Any] = {}
    if payload.get("bold") is not None:
        text_format["bold"] = bool(payload.get("bold"))
    if payload.get("italic") is not None:
        text_format["italic"] = bool(payload.get("italic"))
    if payload.get("font_size") is not None:
        font_size = _int(payload.get("font_size"))
        if font_size < 6 or font_size > 72:
            raise SheetsValidationError("invalid_font_size", "font_size must be 6-72.")
        text_format["fontSize"] = font_size
    foreground = _color(payload.get("text_color"), field="text_color")
    if foreground is not None:
        text_format["foregroundColor"] = foreground
    if text_format:
        result["textFormat"] = text_format
    background = _color(payload.get("background_color"), field="background_color")
    if background is not None:
        result["backgroundColor"] = background
    horizontal = _clean(payload.get("horizontal_alignment"))
    if horizontal:
        result["horizontalAlignment"] = _enum(
            horizontal,
            _HORIZONTAL_ALIGNMENTS,
            field="horizontal_alignment",
            default="LEFT",
        )
    vertical = _clean(payload.get("vertical_alignment"))
    if vertical:
        result["verticalAlignment"] = _enum(
            vertical,
            _VERTICAL_ALIGNMENTS,
            field="vertical_alignment",
            default="BOTTOM",
        )
    wrap = _clean(payload.get("wrap_strategy"))
    if wrap:
        result["wrapStrategy"] = _enum(
            wrap,
            _WRAP_STRATEGIES,
            field="wrap_strategy",
            default="WRAP",
        )
    number_type = _clean(payload.get("number_format_type"))
    number_pattern = _clean(payload.get("number_format_pattern"))
    if number_type or number_pattern:
        if not number_type:
            raise SheetsValidationError(
                "number_format_type_required",
                "number_format_type is required when number_format_pattern is set.",
            )
        number_format = {
            "type": _enum(
                number_type,
                _NUMBER_FORMAT_TYPES,
                field="number_format_type",
                default="TEXT",
            )
        }
        if number_pattern:
            number_format["pattern"] = number_pattern
        result["numberFormat"] = number_format
    border_style = _clean(payload.get("border_style"))
    if border_style:
        border = {
            "style": _enum(
                border_style,
                _BORDER_STYLES,
                field="border_style",
                default="SOLID",
            )
        }
        border_color = _color(payload.get("border_color"), field="border_color")
        if border_color is not None:
            border["color"] = border_color
        result["borders"] = {
            "top": dict(border),
            "bottom": dict(border),
            "left": dict(border),
            "right": dict(border),
        }
    if not result:
        raise SheetsValidationError(
            "format_required",
            "Provide at least one supported formatting property.",
        )
    return result


def _format_range(*, client: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    spreadsheet_id, spreadsheet = _open_spreadsheet(
        client, payload.get("spreadsheet_ref")
    )
    sheet_id = _int(payload.get("sheet_id"), default=-1)
    if sheet_id < 0:
        raise SheetsValidationError(
            "sheet_id_required", "sheet_id must be a non-negative integer."
        )
    range_name = _bounded_ranges([payload.get("range")], max_cells=MAX_WRITE_CELLS)[0]
    worksheet = spreadsheet.get_worksheet_by_id(sheet_id)
    format_payload = _format_payload(payload)
    worksheet.format(range_name, format_payload)
    return {
        "spreadsheet_id": spreadsheet_id,
        "web_url": _web_url(spreadsheet_id),
        "sheet_id": sheet_id,
        "range": range_name,
        "format_fields": sorted(format_payload),
    }


_CLIENT_OPERATIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "describe": _describe,
    "read": _read,
    "update_values": _update_values,
    "append_rows": _append_rows,
    "clear_values": _clear_values,
    "create_spreadsheet": _create_spreadsheet,
    "add_tab": _add_tab,
    "update_tab": _update_tab,
    "delete_tab": _delete_tab,
    "format_range": _format_range,
}


def _exception_chain(exc: Exception) -> list[Exception]:
    chain: list[Exception] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while isinstance(current, Exception) and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return chain


def _provider_reason(error: Mapping[str, Any]) -> str:
    for row in error.get("details") or []:
        if isinstance(row, Mapping) and _clean(row.get("reason")):
            return _clean(row.get("reason"))
    for row in error.get("errors") or []:
        if isinstance(row, Mapping) and _clean(row.get("reason")):
            return _clean(row.get("reason"))
    return ""


def _redact_error_text(value: Any, *, access_token: str) -> str:
    text = _clean(value)
    if access_token:
        text = text.replace(access_token, "[REDACTED]")
    return re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [REDACTED]", text)


def _provider_error(
    exc: Exception,
    *,
    operation: str,
    access_token: str,
) -> dict[str, Any]:
    chain = _exception_chain(exc)
    status = 0
    provider_code = ""
    provider_reason = ""
    message = ""
    provider_exception = next(
        (item for item in chain if isinstance(item, SheetsProviderError)),
        None,
    )
    detail_exception = next(
        (item for item in chain if not isinstance(item, SheetsOperationFailure)),
        exc,
    )
    provider_error_type = detail_exception.__class__.__name__
    for candidate in chain:
        candidate_message = _redact_error_text(candidate, access_token=access_token)
        if candidate_message and not message:
            message = candidate_message
        candidate_status = _int(getattr(candidate, "status", 0))
        candidate_code = _clean(getattr(candidate, "code", ""))
        if candidate_status and not status:
            status = candidate_status
        if candidate_code and not provider_code:
            provider_code = candidate_code
        response = getattr(candidate, "response", None)
        if response is None:
            continue
        provider_error_type = candidate.__class__.__name__
        status = _int(getattr(response, "status_code", status))
        try:
            body = response.json()
            error = body.get("error") if isinstance(body, Mapping) else None
            if isinstance(error, Mapping):
                provider_code = (
                    _clean(error.get("status"))
                    or _clean(error.get("code"))
                    or provider_code
                )
                provider_reason = _provider_reason(error) or provider_reason
                message = (
                    _redact_error_text(
                        error.get("message"), access_token=access_token
                    )
                    or message
                )
        except Exception:
            pass
        break

    reason_key = re.sub(r"[^A-Z0-9]", "", provider_reason.upper())
    error_type_key = provider_error_type.lower()
    if status == 401 or reason_key in {
        "ACCESSTOKENSCOPEINSUFFICIENT",
        "AUTHERROR",
        "INSUFFICIENTPERMISSIONS",
        "INVALIDCREDENTIALS",
        "UNAUTHENTICATED",
    } or "refresherror" in error_type_key:
        normalized_code = "google_sheets_authorization_failed"
    elif reason_key in {"ACCESSNOTCONFIGURED", "APIDISABLED", "SERVICEDISABLED"}:
        normalized_code = "google_sheets_provider_configuration_error"
    elif status == 403:
        normalized_code = "google_sheets_access_denied"
    elif status == 404:
        normalized_code = "google_sheets_not_found"
    elif status == 429:
        normalized_code = "google_sheets_rate_limited"
    elif status >= 500:
        normalized_code = "google_sheets_provider_unavailable"
    elif status == 0 and any(
        marker in error_type_key
        for marker in ("connection", "timeout", "transport")
    ):
        normalized_code = "google_sheets_transport_error"
    elif provider_exception is not None:
        normalized_code = (
            _clean(provider_exception.code) or "google_sheets_api_error"
        )
    else:
        normalized_code = "google_sheets_provider_error"

    stage_failure = exc if isinstance(exc, SheetsOperationFailure) else None
    mutation_attempted = (
        stage_failure.mutation_attempted
        if stage_failure is not None
        else operation in _MUTATING_OPERATIONS
    )
    partial_result = (
        dict(stage_failure.partial_result) if stage_failure is not None else {}
    )
    outcome_unknown = bool(mutation_attempted and (status == 0 or status >= 500))
    safe_message = message or "Google Sheets operation failed."
    return {
        "ok": False,
        "error": {
            "code": normalized_code,
            "message": safe_message[:1_000],
            "provider": "google",
            "operation": operation,
            "category": normalized_code.removeprefix("google_sheets_"),
            "provider_status": status,
            "provider_code": provider_code,
            "provider_reason": provider_reason,
            "retryable": status == 0 or status in {408, 429} or status >= 500,
            "stage": stage_failure.stage if stage_failure is not None else operation,
            "outcome_unknown": outcome_unknown,
            "partial_result": partial_result,
            "_diagnostics": {
                "traceback": _redact_error_text(
                    "".join(traceback.format_exception(exc)),
                    access_token=access_token,
                )[:8_000],
                "exception_chain": [
                    {
                        "type": item.__class__.__name__,
                        "message": _redact_error_text(
                            item, access_token=access_token
                        )[:1_000],
                    }
                    for item in chain
                ]
            },
        },
    }


def execute_google_sheets_operation(
    *,
    operation: str,
    access_token: str,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one bounded provider operation and return a serializable envelope."""

    op = _clean(operation)
    token = _clean(access_token)
    body = dict(payload or {})
    if not token:
        return {
            "ok": False,
            "error": {
                "code": "credential_missing_access_token",
                "message": "The connected Google credential has no access token.",
                "provider_status": 0,
                "outcome_unknown": False,
            },
        }
    try:
        if op == "search":
            ret = _search(access_token=token, payload=body)
        else:
            handler = _CLIENT_OPERATIONS.get(op)
            if handler is None:
                raise SheetsValidationError(
                    "unsupported_operation",
                    f"Unsupported Google Sheets operation: {op or '<empty>'}.",
                )
            try:
                client = _authorize(token)
            except Exception as exc:
                raise SheetsOperationFailure(
                    stage="authorize_client",
                    cause=exc,
                    mutation_attempted=False,
                ) from exc
            if op == "create_spreadsheet":
                ret = handler(
                    client=client,
                    access_token=token,
                    payload=body,
                )
            else:
                ret = handler(client=client, payload=body)
        return {"ok": True, "error": None, "ret": ret}
    except SheetsValidationError as exc:
        return {
            "ok": False,
            "error": {
                "code": exc.code,
                "message": str(exc),
                "provider_status": 0,
                "outcome_unknown": False,
            },
        }
    except Exception as exc:
        return _provider_error(
            exc,
            operation=op,
            access_token=token,
        )


__all__ = [
    "MAX_APPEND_ROWS",
    "MAX_RANGES",
    "MAX_SEARCH_RESULTS",
    "MAX_TAB_CELLS",
    "MAX_WRITE_CELLS",
    "execute_google_sheets_operation",
]
