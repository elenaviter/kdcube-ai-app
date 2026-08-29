"""connection-hub get_token: refresh an expired Google token on demand.

Lives with the connections integration tests and exercises the SDK-owned
Connection Hub provider implementation. Mocks the store + the refresh HTTP call
— no real network, no real secrets."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

# Importing the package registers the built-in providers (slack, google).
import kdcube_ai_app.apps.chat.sdk.integrations.connections  # noqa: F401
from kdcube_ai_app.apps.chat.sdk.integrations.prokura.hub import provider_impl


class _FakeStore:
    """Minimal ConnectionStore stand-in: one google account, expired token."""

    def __init__(self) -> None:
        self.account = {
            "account_id": "google_acc1",
            "provider": "google",
            "app_id": "gmail",
            "has_token": True,
        }
        self.tokens = {
            "google_acc1": {
                "access_token": "ya29.OLD",
                "refresh_token": "rt-keepme",
                "expires_at": int(time.time()) - 10,  # already expired
                "scope": ["openid", "email"],
            }
        }
        self.set_calls: list[tuple[str, dict]] = []

    async def list_accounts_async(self, provider=None):
        return [dict(self.account)] if (provider in (None, "google")) else []

    async def get_tokens_async(self, account_id):
        return dict(self.tokens.get(account_id, {}))

    async def set_tokens_async(self, account_id, tokens):
        self.set_calls.append((account_id, dict(tokens)))
        self.tokens[account_id] = dict(tokens)


@pytest.mark.asyncio
async def test_get_token_refreshes_expired_google_token(monkeypatch):
    store = _FakeStore()

    prov = provider_impl.ConnectionHubProvider(entrypoint=SimpleNamespace(), bundle_id="connection-hub@1-0")
    monkeypatch.setattr(prov, "_store", lambda ctx: (store, "user-1"))

    # Resolve a client app + secret without touching config/secret stores.
    fake_app = SimpleNamespace(app_id="gmail", client_id="cid")
    monkeypatch.setattr(
        provider_impl.connections_apps, "resolve_client_app",
        lambda entrypoint, provider, app_id: fake_app,
    )

    async def _fake_secret(bundle_id, provider, app_id):
        return "sec"

    monkeypatch.setattr(provider_impl.connections_apps, "client_app_secret", _fake_secret)

    async def _fake_refresh(provider_obj, *, refresh_token, client_id, client_secret):
        assert refresh_token == "rt-keepme"
        assert client_id == "cid" and client_secret == "sec"
        # Google returns a new access token + expiry but NO new refresh_token.
        return {"access_token": "ya29.NEW", "expires_at": int(time.time()) + 3599}

    monkeypatch.setattr(provider_impl, "refresh_access_token", _fake_refresh)

    ctx = SimpleNamespace(user_id="user-1", principal_kind="user", principal_id="user-1")
    token = await prov.get_token(ctx, provider="google")

    assert token is not None
    assert token.access_token == "ya29.NEW"               # refreshed value returned
    assert token.refresh_token == "rt-keepme"             # old refresh_token preserved
    # Persisted back to the store with the new access token + preserved refresh_token.
    assert store.set_calls and store.set_calls[-1][0] == "google_acc1"
    persisted = store.set_calls[-1][1]
    assert persisted["access_token"] == "ya29.NEW"
    assert persisted["refresh_token"] == "rt-keepme"


@pytest.mark.asyncio
async def test_get_token_no_refresh_when_not_expired(monkeypatch):
    store = _FakeStore()
    store.tokens["google_acc1"]["expires_at"] = int(time.time()) + 3600  # fresh

    prov = provider_impl.ConnectionHubProvider(entrypoint=SimpleNamespace(), bundle_id="connection-hub@1-0")
    monkeypatch.setattr(prov, "_store", lambda ctx: (store, "user-1"))

    def _boom(*a, **k):
        raise AssertionError("must not refresh a fresh token")

    monkeypatch.setattr(provider_impl, "refresh_access_token", _boom)

    ctx = SimpleNamespace(user_id="user-1", principal_kind="user", principal_id="user-1")
    token = await prov.get_token(ctx, provider="google")
    assert token is not None and token.access_token == "ya29.OLD"
    assert not store.set_calls
