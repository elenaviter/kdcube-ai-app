from __future__ import annotations

from typing import Any

from kdcube_ai_app.apps.chat.sdk.integrations.google import sheets_proxy


class FakeWorksheet:
    def __init__(self, *, sheet_id: int = 11, title: str = "Data") -> None:
        self.id = sheet_id
        self.title = title
        self.index = 0
        self.row_count = 100
        self.col_count = 10
        self.frozen_row_count = 0
        self.frozen_col_count = 0
        self.url = (
            f"https://docs.google.com/spreadsheets/d/sheet-123/edit#gid={sheet_id}"
        )
        self.formatted: tuple[Any, Any] | None = None

    def update_title(self, title: str):
        self.title = title
        return {}

    def resize(self, *, rows=None, cols=None):
        if rows is not None:
            self.row_count = rows
        if cols is not None:
            self.col_count = cols
        return {}

    def freeze(self, *, rows=None, cols=None):
        if rows is not None:
            self.frozen_row_count = rows
        if cols is not None:
            self.frozen_col_count = cols
        return {}

    def format(self, ranges, format):
        self.formatted = (ranges, format)
        return {}


class FakeSpreadsheet:
    def __init__(
        self, *, spreadsheet_id: str = "sheet-123", title: str = "Roadmap"
    ) -> None:
        self.id = spreadsheet_id
        self.title = title
        self.url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit"
        self.sheet1 = FakeWorksheet(title="Sheet1")
        self.worksheets = {self.sheet1.id: self.sheet1}
        self.last_batch_update: dict[str, Any] | None = None
        self.last_append: tuple[Any, Any, Any] | None = None
        self.last_clear: dict[str, Any] | None = None
        self.deleted_sheet_id: int | None = None

    def fetch_sheet_metadata(self, params=None):
        return {
            "properties": {
                "title": self.title,
                "locale": "en_US",
                "timeZone": "Europe/Kyiv",
            },
            "sheets": [
                {
                    "properties": {
                        "sheetId": 11,
                        "title": "Data",
                        "index": 0,
                        "sheetType": "GRID",
                        "gridProperties": {
                            "rowCount": 100,
                            "columnCount": 10,
                            "frozenRowCount": 1,
                        },
                    }
                }
            ],
            "namedRanges": [
                {"namedRangeId": "nr-1", "name": "Input", "range": {"sheetId": 11}}
            ],
        }

    def values_batch_get(self, ranges, params=None):
        return {
            "valueRanges": [
                {
                    "range": ranges[0],
                    "majorDimension": (params or {}).get("majorDimension", "ROWS"),
                    "values": [["Name", "Value"], ["A", 1]],
                }
            ]
        }

    def values_batch_update(self, body):
        self.last_batch_update = body
        return {
            "totalUpdatedRows": 2,
            "totalUpdatedColumns": 2,
            "totalUpdatedCells": 4,
            "totalUpdatedSheets": 1,
        }

    def values_append(self, range, params, body):
        self.last_append = (range, params, body)
        return {
            "tableRange": "Data!A1:B2",
            "updates": {
                "updatedRange": "Data!A3:B3",
                "updatedRows": 1,
                "updatedColumns": 2,
                "updatedCells": 2,
            },
        }

    def values_batch_clear(self, params=None, body=None):
        self.last_clear = body
        return {"clearedRanges": list((body or {}).get("ranges") or [])}

    def values_update(self, range, params=None, body=None):
        return {"updatedRange": range}

    def add_worksheet(self, *, title, rows, cols, index=None):
        worksheet = FakeWorksheet(sheet_id=22, title=title)
        worksheet.row_count = rows
        worksheet.col_count = cols
        worksheet.index = index or 0
        self.worksheets[worksheet.id] = worksheet
        return worksheet

    def get_worksheet_by_id(self, sheet_id):
        return self.worksheets.setdefault(sheet_id, FakeWorksheet(sheet_id=sheet_id))

    def del_worksheet_by_id(self, sheet_id):
        self.deleted_sheet_id = sheet_id
        self.worksheets.pop(sheet_id, None)
        return {}


class FakeClient:
    def __init__(self) -> None:
        self.spreadsheet = FakeSpreadsheet()
        self.created: list[str] = []

    def open_by_key(self, spreadsheet_id: str):
        self.spreadsheet.id = spreadsheet_id
        return self.spreadsheet

    def create(self, title: str):
        self.created.append(title)
        return FakeSpreadsheet(spreadsheet_id="created-456", title=title)


def _execute(monkeypatch, operation: str, payload: dict[str, Any]):
    client = FakeClient()
    monkeypatch.setattr(sheets_proxy, "_authorize", lambda _token: client)
    result = sheets_proxy.execute_google_sheets_operation(
        operation=operation,
        access_token="secret-token",
        payload=payload,
    )
    return client, result


def test_search_returns_stable_drive_metadata(monkeypatch):
    monkeypatch.setattr(
        sheets_proxy,
        "_drive_files_list",
        lambda **_kwargs: {
            "files": [
                {
                    "id": "sheet-123",
                    "name": "Roadmap",
                    "modifiedTime": "2026-07-27T10:00:00Z",
                    "ownedByMe": True,
                    "owners": [
                        {"displayName": "Elena", "emailAddress": "owner@example.com"}
                    ],
                }
            ],
            "nextPageToken": "next-page",
        },
    )
    result = sheets_proxy.execute_google_sheets_operation(
        operation="search",
        access_token="secret-token",
        payload={"query": "road", "limit": 10},
    )

    assert result["ok"] is True
    assert result["ret"]["next_cursor"] == "next-page"
    assert result["ret"]["items"][0] == {
        "spreadsheet_id": "sheet-123",
        "title": "Roadmap",
        "created_time": "",
        "modified_time": "2026-07-27T10:00:00Z",
        "owned_by_me": True,
        "drive_id": "",
        "owners": [{"display_name": "Elena", "email": "owner@example.com"}],
        "web_url": "https://docs.google.com/spreadsheets/d/sheet-123/edit",
    }


def test_describe_and_read_accept_full_spreadsheet_url(monkeypatch):
    client, described = _execute(
        monkeypatch,
        "describe",
        {
            "spreadsheet_ref": "https://docs.google.com/spreadsheets/d/sheet-123/edit#gid=11"
        },
    )
    assert described["ok"] is True
    assert described["ret"]["spreadsheet_id"] == "sheet-123"
    assert described["ret"]["tabs"][0]["sheet_id"] == 11
    assert described["ret"]["named_ranges"][0]["name"] == "Input"

    monkeypatch.setattr(sheets_proxy, "_authorize", lambda _token: client)
    read = sheets_proxy.execute_google_sheets_operation(
        operation="read",
        access_token="secret-token",
        payload={"spreadsheet_ref": "sheet-123", "ranges": ["Data!A1:B2"]},
    )
    assert read["ok"] is True
    assert read["ret"]["cell_count"] == 4
    assert read["ret"]["ranges"][0]["values"][1] == ["A", 1]


def test_value_operations_are_bounded_and_return_affected_ranges(monkeypatch):
    client, updated = _execute(
        monkeypatch,
        "update_values",
        {
            "spreadsheet_ref": "sheet-123",
            "updates": [{"range": "Data!A1:B2", "values": [["A", 1], ["B", 2]]}],
            "value_input_option": "USER_ENTERED",
        },
    )
    assert updated["ok"] is True
    assert updated["ret"]["updated_cells"] == 4
    assert client.spreadsheet.last_batch_update["data"][0]["range"] == "Data!A1:B2"

    monkeypatch.setattr(sheets_proxy, "_authorize", lambda _token: client)
    appended = sheets_proxy.execute_google_sheets_operation(
        operation="append_rows",
        access_token="secret-token",
        payload={
            "spreadsheet_ref": "sheet-123",
            "range": "Data!A1:B1",
            "rows": [["C", 3]],
            "idempotency_key": "op-1",
        },
    )
    assert appended["ok"] is True
    assert appended["ret"]["updated_range"] == "Data!A3:B3"
    assert appended["ret"]["idempotency_key"] == "op-1"

    cleared = sheets_proxy.execute_google_sheets_operation(
        operation="clear_values",
        access_token="secret-token",
        payload={"spreadsheet_ref": "sheet-123", "ranges": ["Data!A10:B20"]},
    )
    assert cleared["ok"] is True
    assert cleared["ret"]["cleared_ranges"] == ["Data!A10:B20"]


def test_create_tab_update_delete_and_format(monkeypatch):
    client, created = _execute(
        monkeypatch,
        "create_spreadsheet",
        {
            "title": "Quarterly plan",
            "first_tab_title": "Plan",
            "initial_values": [["Item", "Owner"]],
        },
    )
    assert created["ok"] is True
    assert created["ret"]["spreadsheet_id"] == "created-456"
    assert created["ret"]["first_tab"]["title"] == "Plan"

    monkeypatch.setattr(sheets_proxy, "_authorize", lambda _token: client)
    added = sheets_proxy.execute_google_sheets_operation(
        operation="add_tab",
        access_token="secret-token",
        payload={
            "spreadsheet_ref": "sheet-123",
            "title": "Summary",
            "rows": 100,
            "columns": 8,
        },
    )
    assert added["ret"]["tab"]["sheet_id"] == 22

    updated = sheets_proxy.execute_google_sheets_operation(
        operation="update_tab",
        access_token="secret-token",
        payload={
            "spreadsheet_ref": "sheet-123",
            "sheet_id": 22,
            "title": "Overview",
            "rows": 200,
            "columns": 12,
            "frozen_rows": 1,
        },
    )
    assert updated["ret"]["tab"]["title"] == "Overview"
    assert updated["ret"]["tab"]["frozen_row_count"] == 1

    formatted = sheets_proxy.execute_google_sheets_operation(
        operation="format_range",
        access_token="secret-token",
        payload={
            "spreadsheet_ref": "sheet-123",
            "sheet_id": 22,
            "range": "A1:H1",
            "bold": True,
            "background_color": "#112233",
            "horizontal_alignment": "CENTER",
        },
    )
    assert formatted["ok"] is True
    worksheet = client.spreadsheet.get_worksheet_by_id(22)
    assert worksheet.formatted[0] == "A1:H1"
    assert worksheet.formatted[1]["textFormat"]["bold"] is True

    deleted = sheets_proxy.execute_google_sheets_operation(
        operation="delete_tab",
        access_token="secret-token",
        payload={"spreadsheet_ref": "sheet-123", "sheet_id": 22},
    )
    assert deleted["ok"] is True
    assert client.spreadsheet.deleted_sheet_id == 22


def test_validation_and_provider_auth_errors_are_structured(monkeypatch):
    _, invalid = _execute(
        monkeypatch,
        "read",
        {"spreadsheet_ref": "sheet-123", "ranges": ["Data!A1:Z1000"]},
    )
    assert invalid["ok"] is False
    assert invalid["error"]["code"] == "request_too_large"

    class Unauthorized(RuntimeError):
        status = 401
        code = "UNAUTHENTICATED"

    monkeypatch.setattr(
        sheets_proxy,
        "_authorize",
        lambda _token: (_ for _ in ()).throw(Unauthorized("expired")),
    )
    denied = sheets_proxy.execute_google_sheets_operation(
        operation="describe",
        access_token="secret-token",
        payload={"spreadsheet_ref": "sheet-123"},
    )
    assert denied["ok"] is False
    assert denied["error"]["code"] == "google_sheets_authorization_failed"
    assert denied["error"]["provider_status"] == 401
    assert "secret-token" not in str(denied)


def test_mutating_provider_5xx_reports_unknown_outcome(monkeypatch):
    class ProviderUnavailable(RuntimeError):
        status = 503
        code = "UNAVAILABLE"

    monkeypatch.setattr(
        sheets_proxy,
        "_authorize",
        lambda _token: (_ for _ in ()).throw(ProviderUnavailable("retry later")),
    )

    write_result = sheets_proxy.execute_google_sheets_operation(
        operation="append_rows",
        access_token="secret-token",
        payload={
            "spreadsheet_ref": "sheet-123",
            "range": "Data!A1:B1",
            "rows": [["A", 1]],
        },
    )
    read_result = sheets_proxy.execute_google_sheets_operation(
        operation="describe",
        access_token="secret-token",
        payload={"spreadsheet_ref": "sheet-123"},
    )

    assert write_result["error"]["provider_status"] == 503
    assert write_result["error"]["outcome_unknown"] is True
    assert read_result["error"]["outcome_unknown"] is False
