from __future__ import annotations

from typing import Any

import pytest

from kdcube_ai_app.apps.chat.sdk.integrations.sheets.named_service import (
    ACTION_APPEND_ROWS,
    ACTION_DELETE_TAB,
    SHEETS_CONNECTED_ACCOUNT_REQUIREMENTS,
    SHEETS_GRANT_HINTS,
    SHEETS_READ_CLAIM,
    SHEETS_WRITE_CLAIM,
    SheetsNamedServiceProvider,
    parse_sheets_ref,
    spreadsheet_ref,
    tab_ref,
)
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers import (
    NamedServiceContext,
    NamedServiceRequest,
)
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.types import (
    OBJECT_ACTION,
    OBJECT_DELETE,
    OBJECT_GET,
    OBJECT_SEARCH,
    OBJECT_UPSERT,
)


def _ctx() -> NamedServiceContext:
    return NamedServiceContext(tenant="demo", project="project", user_id="user-1")


class _FakeSheets:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error: dict[str, Any] | None = None

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        if self.error is not None:
            return dict(self.error)
        operation = kwargs["operation"]
        account_id = kwargs.get("account_id") or "account-1"
        if operation == "search":
            return {
                "ok": True,
                "ret": {
                    "account_id": account_id,
                    "items": [
                        {
                            "spreadsheet_id": "sheet-1",
                            "title": "Quarterly plan",
                            "web_url": "https://docs.google.com/spreadsheets/d/sheet-1/edit",
                        }
                    ],
                    "next_cursor": "next-1",
                },
            }
        if operation == "describe":
            return {
                "ok": True,
                "ret": {
                    "account_id": account_id,
                    "spreadsheet_id": "sheet-1",
                    "title": "Quarterly plan",
                    "tabs": [
                        {
                            "sheet_id": 7,
                            "title": "Plan",
                            "index": 0,
                            "row_count": 100,
                            "column_count": 12,
                        }
                    ],
                },
            }
        if operation == "read":
            return {
                "ok": True,
                "ret": {
                    "account_id": account_id,
                    "spreadsheet_id": "sheet-1",
                    "ranges": [
                        {"range": "Plan!A1:B2", "values": [["A", "B"], [1, 2]]}
                    ],
                    "cell_count": 4,
                },
            }
        if operation == "create_spreadsheet":
            return {
                "ok": True,
                "ret": {
                    "account_id": account_id,
                    "spreadsheet_id": "created-1",
                    "title": kwargs["payload"]["title"],
                    "first_tab": {"sheet_id": 0, "title": "Sheet1", "index": 0},
                },
            }
        if operation in {"update_tab", "add_tab"}:
            return {
                "ok": True,
                "ret": {
                    "account_id": account_id,
                    "spreadsheet_id": "sheet-1",
                    "tab": {"sheet_id": 7, "title": "Updated", "index": 0},
                },
            }
        return {
            "ok": True,
            "ret": {
                "account_id": account_id,
                "spreadsheet_id": "sheet-1",
                "operation": operation,
            },
        }


def _provider(fake: _FakeSheets) -> SheetsNamedServiceProvider:
    return SheetsNamedServiceProvider(
        execute_operation=fake.execute,
        bundle_id="kdcube-services@1-0",
    )


def test_sheets_refs_are_provider_neutral_and_stable() -> None:
    spreadsheet = spreadsheet_ref(
        provider="google",
        account_id="account-1",
        spreadsheet_id="sheet-1",
    )
    tab = tab_ref(
        provider="google",
        account_id="account-1",
        spreadsheet_id="sheet-1",
        sheet_id=7,
    )
    assert spreadsheet == "sheets:google:account-1:spreadsheet:sheet-1"
    assert tab == "sheets:google:account-1:spreadsheet:sheet-1:tab:7"
    assert parse_sheets_ref(spreadsheet)["kind"] == "spreadsheet"
    assert parse_sheets_ref(tab) == {
        "ref": tab,
        "provider": "google",
        "account_id": "account-1",
        "spreadsheet_id": "sheet-1",
        "sheet_id": 7,
        "kind": "tab",
    }


def test_write_grant_and_connected_account_claims_are_separate_boundaries() -> None:
    assert SHEETS_GRANT_HINTS["object.upsert"] == [SHEETS_WRITE_CLAIM]
    claims_by_operation = SHEETS_CONNECTED_ACCOUNT_REQUIREMENTS[0][
        "claims_by_operation"
    ]
    assert claims_by_operation["object.upsert"] == [
        SHEETS_READ_CLAIM,
        SHEETS_WRITE_CLAIM,
    ]


@pytest.mark.anyio
async def test_search_returns_named_service_objects() -> None:
    fake = _FakeSheets()
    response = await _provider(fake).dispatch(
        _ctx(),
        NamedServiceRequest(
            operation=OBJECT_SEARCH,
            namespace="sheets",
            query="quarterly",
            limit=5,
            filters={"account_id": "account-1"},
        ),
    )
    assert response.ok is True
    assert response.next_cursor == "next-1"
    assert response.items[0]["ref"] == (
        "sheets:google:account-1:spreadsheet:sheet-1"
    )
    assert response.items[0]["object_kind"] == "sheets.spreadsheet"
    assert fake.calls == [
        {
            "operation": "search",
            "claim": SHEETS_READ_CLAIM,
            "tool_name": "named_services.sheets.object.search",
            "payload": {"query": "quarterly", "limit": 5, "cursor": ""},
            "account_id": "account-1",
        }
    ]


@pytest.mark.anyio
async def test_get_reads_metadata_or_explicit_ranges() -> None:
    fake = _FakeSheets()
    provider = _provider(fake)
    ref = "sheets:google:account-1:spreadsheet:sheet-1"

    metadata = await provider.dispatch(
        _ctx(),
        NamedServiceRequest(operation=OBJECT_GET, namespace="sheets", object_ref=ref),
    )
    assert metadata.ok is True
    assert metadata.object["tabs"][0]["ref"].endswith(":tab:7")

    values = await provider.dispatch(
        _ctx(),
        NamedServiceRequest(
            operation=OBJECT_GET,
            namespace="sheets",
            object_ref=ref,
            filters={"ranges": ["Plan!A1:B2"]},
        ),
    )
    assert values.ok is True
    assert values.object["ranges"][0]["values"] == [["A", "B"], [1, 2]]
    assert [call["operation"] for call in fake.calls] == ["describe", "read"]


@pytest.mark.anyio
async def test_upsert_create_and_tab_update_reuse_google_service() -> None:
    fake = _FakeSheets()
    provider = _provider(fake)

    created = await provider.dispatch(
        _ctx(),
        NamedServiceRequest(
            operation=OBJECT_UPSERT,
            namespace="sheets",
            object={"title": "Created from named services"},
        ),
    )
    assert created.ok is True
    assert created.object_ref == (
        "sheets:google:account-1:spreadsheet:created-1"
    )

    updated = await provider.dispatch(
        _ctx(),
        NamedServiceRequest(
            operation=OBJECT_UPSERT,
            namespace="sheets",
            object_ref="sheets:google:account-1:spreadsheet:sheet-1:tab:7",
            object={"title": "Updated"},
        ),
    )
    assert updated.ok is True
    assert updated.object_ref.endswith(":tab:7")
    assert fake.calls[0]["claim"] == (SHEETS_READ_CLAIM, SHEETS_WRITE_CLAIM)
    assert fake.calls[1]["payload"]["sheet_id"] == 7


@pytest.mark.anyio
async def test_actions_and_delete_require_explicit_refs() -> None:
    fake = _FakeSheets()
    provider = _provider(fake)
    spreadsheet = "sheets:google:account-1:spreadsheet:sheet-1"
    tab = f"{spreadsheet}:tab:7"

    appended = await provider.dispatch(
        _ctx(),
        NamedServiceRequest(
            operation=OBJECT_ACTION,
            namespace="sheets",
            object_ref=spreadsheet,
            action=ACTION_APPEND_ROWS,
            payload={"range": "Plan!A:C", "rows": [[1, 2, 3]]},
        ),
    )
    assert appended.ok is True
    assert fake.calls[-1]["operation"] == ACTION_APPEND_ROWS

    deleted = await provider.dispatch(
        _ctx(),
        NamedServiceRequest(
            operation=OBJECT_DELETE,
            namespace="sheets",
            object_ref=tab,
        ),
    )
    assert deleted.ok is True
    assert fake.calls[-1]["operation"] == ACTION_DELETE_TAB
    assert fake.calls[-1]["payload"]["sheet_id"] == 7
    assert deleted.object_ref == tab

    refused = await provider.dispatch(
        _ctx(),
        NamedServiceRequest(
            operation=OBJECT_DELETE,
            namespace="sheets",
            object_ref=spreadsheet,
        ),
    )
    assert refused.ok is False
    assert refused.error.code == "sheets_spreadsheet_delete_not_supported"


@pytest.mark.anyio
async def test_connected_account_consent_error_is_preserved() -> None:
    fake = _FakeSheets()
    fake.error = {
        "ok": False,
        "error": {
            "code": "needs_connected_account_consent",
            "message": "Approve spreadsheet access.",
        },
        "consent": {
            "reason": "claim_upgrade_required",
            "provider_id": "google",
            "claims": [SHEETS_READ_CLAIM],
            "url": "https://runtime.test/connections",
            "retry_hint": True,
        },
    }
    response = await _provider(fake).dispatch(
        _ctx(),
        NamedServiceRequest(
            operation=OBJECT_SEARCH,
            namespace="sheets",
            query="plan",
        ),
    )
    assert response.ok is False
    assert response.status == 403
    assert response.error.code == "needs_connected_account_consent"
    assert response.error.details["reason"] == "claim_upgrade_required"
    assert response.error.details["claims"] == [SHEETS_READ_CLAIM]
    assert response.error.details["connection_hub_url"] == (
        "https://runtime.test/connections"
    )


@pytest.mark.anyio
async def test_canonical_fields_and_ref_boundaries_cannot_be_overridden() -> None:
    fake = _FakeSheets()

    async def _malicious_search(**kwargs: Any) -> dict[str, Any]:
        fake.calls.append(dict(kwargs))
        return {
            "ok": True,
            "ret": {
                "account_id": "account-1",
                "items": [
                    {
                        "spreadsheet_id": "sheet-1",
                        "ref": "mail:message:not-a-sheet",
                        "object_kind": "wrong.kind",
                        "account_id": "wrong-account",
                    }
                ],
            },
        }

    provider = SheetsNamedServiceProvider(execute_operation=_malicious_search)
    response = await provider.dispatch(
        _ctx(),
        NamedServiceRequest(
            operation=OBJECT_SEARCH,
            namespace="sheets",
            query="",
            filters={"account_id": "account-1"},
        ),
    )
    assert response.ok is True
    assert response.items[0]["ref"] == (
        "sheets:google:account-1:spreadsheet:sheet-1"
    )
    assert response.items[0]["object_kind"] == "sheets.spreadsheet"
    assert response.items[0]["account_id"] == "account-1"


@pytest.mark.anyio
async def test_tab_range_read_and_unknown_provider_fail_before_google_call() -> None:
    fake = _FakeSheets()
    provider = _provider(fake)

    tab_ranges = await provider.dispatch(
        _ctx(),
        NamedServiceRequest(
            operation=OBJECT_GET,
            namespace="sheets",
            object_ref="sheets:google:account-1:spreadsheet:sheet-1:tab:7",
            filters={"ranges": ["Plan!A1:B2"]},
        ),
    )
    assert tab_ranges.ok is False
    assert tab_ranges.error.code == "sheets_spreadsheet_ref_required_for_ranges"

    unsupported = await provider.dispatch(
        _ctx(),
        NamedServiceRequest(
            operation=OBJECT_ACTION,
            namespace="sheets",
            object_ref="sheets:excel:account-1:spreadsheet:sheet-1",
            action=ACTION_APPEND_ROWS,
            payload={"range": "Plan!A:C", "rows": [[1, 2, 3]]},
        ),
    )
    assert unsupported.ok is False
    assert unsupported.error.code == "sheets_provider_not_implemented"
    assert fake.calls == []
