# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.delegated_credentials.oauth.tests.helpers import (
    enable_delegated_client,
    mount_test_oauth_adapter,
)


class _GrantStore:
    def __init__(self, *, kind: str) -> None:
        self.kind = kind
        self.events: list[str] = []

    async def validate_refresh_token(self, token: str):
        if self.kind != "refresh" or token != "caller-token":
            return None
        return {
            "registry_access_id": "access-1",
            "sub": "owner-1",
        }

    async def get_access_grant_record(self, token: str):
        if self.kind != "access" or token != "caller-token":
            return None
        return {
            "registry_access_id": "access-1",
            "credential": {"grantor_subject": "owner-1"},
        }

    async def revoke_refresh_token(self, token: str) -> bool:
        assert token == "caller-token"
        self.events.append("refresh_token")
        return True

    async def revoke_access_grant(self, token: str) -> bool:
        assert token == "caller-token"
        self.events.append("access_token")
        return True


class _CardService:
    def __init__(self, store: _GrantStore, *, result: dict) -> None:
        self.store = store
        self.result = result
        self.calls: list[tuple[dict, str]] = []

    async def revoke_access(self, user, *, access_id: str):
        self.calls.append((dict(user), access_id))
        self.store.events.append("card")
        return dict(self.result)


def _client(
    *, kind: str, card_result: dict
) -> tuple[TestClient, _GrantStore, _CardService]:
    app = FastAPI()
    enable_delegated_client(app)
    store = _GrantStore(kind=kind)
    cards = _CardService(store, result=card_result)
    app.state.oauth_grant_store = store
    app.state.automation_access_factory = lambda: cards
    mount_test_oauth_adapter(app)
    return TestClient(app), store, cards


def test_refresh_disconnect_retires_card_before_token_record() -> None:
    client, store, cards = _client(kind="refresh", card_result={"ok": True})

    response = client.post("/oauth/revoke", data={"token": "caller-token"})

    assert response.status_code == 200
    assert response.json() == {}
    assert cards.calls == [({"user_id": "owner-1"}, "access-1")]
    assert store.events == ["card", "refresh_token"]


def test_access_disconnect_retires_card_before_token_record() -> None:
    client, store, cards = _client(kind="access", card_result={"ok": True})

    response = client.post("/oauth/revoke", data={"token": "caller-token"})

    assert response.status_code == 200
    assert cards.calls == [({"user_id": "owner-1"}, "access-1")]
    assert store.events == ["card", "access_token"]


def test_card_retirement_failure_keeps_token_record_for_retry() -> None:
    client, store, cards = _client(
        kind="refresh",
        card_result={"ok": False, "error": "delegated_cards_unavailable"},
    )

    response = client.post("/oauth/revoke", data={"token": "caller-token"})

    assert response.status_code == 503
    assert response.json()["error"] == "temporarily_unavailable"
    assert cards.calls == [({"user_id": "owner-1"}, "access-1")]
    assert store.events == ["card"]


def test_unknown_token_remains_a_non_probing_success() -> None:
    client, store, cards = _client(kind="unknown", card_result={"ok": True})

    response = client.post("/oauth/revoke", data={"token": "unknown-token"})

    assert response.status_code == 200
    assert response.json() == {}
    assert cards.calls == []
    assert store.events == []
