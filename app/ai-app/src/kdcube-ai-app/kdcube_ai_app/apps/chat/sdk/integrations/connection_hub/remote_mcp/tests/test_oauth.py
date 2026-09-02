from __future__ import annotations

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from connection_hub.remote_mcp import (
    AUTH_OAUTH,
    OAUTH_CLIENT_SOURCE_DCR,
    OAUTH_CLIENT_SOURCE_PROVISIONED,
    BundleStorageRemoteMCPOAuthStateStore,
    RemoteMCPCredential,
    RemoteMCPEndpointPolicy,
    RemoteMCPOAuthCredential,
    RemoteMCPOAuthStateError,
)
from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.remote_mcp.oauth import (
    KDCubeRemoteMCPOAuthService,
    OAUTH_CLIENT_MODE_AUTOMATIC,
    OAUTH_CLIENT_MODE_PROVISIONED,
    RemoteMCPOAuthFlowError,
)


class _Secrets:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    async def set(self, *, owner_subject: str, secret_ref: str, value: str):
        self.values[(owner_subject, secret_ref)] = value

    async def get(self, *, owner_subject: str, secret_ref: str):
        return self.values.get((owner_subject, secret_ref))

    async def delete(self, *, owner_subject: str, secret_ref: str):
        self.values.pop((owner_subject, secret_ref), None)


class _Response:
    def __init__(self, status: int, payload=None, *, headers=None) -> None:
        self.status_code = status
        self.headers = dict(headers or {})
        self._raw = (
            payload
            if isinstance(payload, bytes)
            else json.dumps(payload or {}).encode("utf-8")
        )

    async def aread(self):
        return self._raw


class _OAuthHttpClient:
    def __init__(
        self,
        *,
        client_metadata_supported: bool = False,
        registration_supported: bool = True,
        token_auth_methods: tuple[str, ...] = (
            "none",
            "client_secret_post",
        ),
    ) -> None:
        self.calls: list[dict] = []
        self.client_metadata_supported = client_metadata_supported
        self.registration_supported = registration_supported
        self.token_auth_methods = token_auth_methods

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, url, **kwargs):
        self.calls.append({"method": "GET", "url": url, **kwargs})
        if "oauth-protected-resource" in url:
            return _Response(
                200,
                {
                    "resource": "https://mcp.example.test/mcp",
                    "authorization_servers": ["https://auth.example.test"],
                    "scopes_supported": ["records.read"],
                },
            )
        if "oauth-authorization-server" in url:
            metadata = {
                "issuer": "https://auth.example.test",
                "authorization_endpoint": "https://auth.example.test/authorize",
                "token_endpoint": "https://auth.example.test/token",
                "revocation_endpoint": "https://auth.example.test/revoke",
                "scopes_supported": ["records.read", "offline_access"],
                "response_types_supported": ["code"],
                "grant_types_supported": [
                    "authorization_code",
                    "refresh_token",
                ],
                "code_challenge_methods_supported": ["S256"],
                "token_endpoint_auth_methods_supported": list(
                    self.token_auth_methods
                ),
                "authorization_response_iss_parameter_supported": True,
            }
            if self.client_metadata_supported:
                metadata["client_id_metadata_document_supported"] = True
            elif self.registration_supported:
                metadata["registration_endpoint"] = (
                    "https://auth.example.test/register"
                )
            return _Response(
                200,
                metadata,
            )
        return _Response(404)

    async def post(self, url, **kwargs):
        self.calls.append({"method": "POST", "url": url, **kwargs})
        if url == "https://mcp.example.test/mcp":
            return _Response(
                401,
                headers={
                    "www-authenticate": (
                        'Bearer resource_metadata="https://mcp.example.test/'
                        '.well-known/oauth-protected-resource", '
                        'scope="records.read"'
                    )
                },
            )
        if url == "https://auth.example.test/register":
            return _Response(
                201,
                {
                    "client_id": "dynamic-client",
                    "client_secret": "dynamic-secret",
                    "redirect_uris": ["https://hub.example.test/oauth/callback"],
                    "token_endpoint_auth_method": "client_secret_post",
                    "grant_types": ["authorization_code", "refresh_token"],
                    "application_type": "web",
                },
            )
        if url == "https://auth.example.test/token":
            return _Response(
                200,
                {
                    "access_token": "access-value",
                    "refresh_token": "refresh-value",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "scope": "records.read offline_access",
                },
            )
        return _Response(404)


class _ConnectorService:
    def __init__(
        self, *, existing_oauth: RemoteMCPOAuthCredential | None = None
    ) -> None:
        self.calls: list[dict] = []
        self.existing_oauth = existing_oauth or RemoteMCPOAuthCredential(
            access_token="existing-access",
            refresh_token="existing-refresh",
            token_endpoint="https://auth.example.test/token",
            authorization_server="https://auth.example.test",
            client_id="dynamic-client",
            client_secret="dynamic-secret",
            token_endpoint_auth_method="client_secret_post",
            client_source=OAUTH_CLIENT_SOURCE_DCR,
        )

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            connector_id="mcp_0123456789abcdef01234567",
            label=kwargs["label"],
        )

    async def get(self, **kwargs):
        self.calls.append({"method": "get", **kwargs})
        return SimpleNamespace(
            connector_id=kwargs["connector_id"],
            label="Existing records",
            endpoint="https://mcp.example.test/mcp",
            revision=7,
        )

    async def resolve_credential(self, **kwargs):
        self.calls.append({"method": "resolve_credential", **kwargs})
        return RemoteMCPCredential(
            mode=AUTH_OAUTH,
            value=self.existing_oauth.to_json(),
        )

    async def replace_credential(self, **kwargs):
        self.calls.append({"method": "replace_credential", **kwargs})
        return SimpleNamespace(
            connector_id=kwargs["connector_id"],
            label="Existing records",
        )


async def _public_resolver(_host: str, _port: int):
    return ("93.184.216.34",)


@pytest.mark.asyncio
async def test_upstream_oauth_discovers_registers_and_completes_without_returning_secrets(
    tmp_path,
    monkeypatch,
):
    secrets_store = _Secrets()
    connector_service = _ConnectorService()
    state_store = BundleStorageRemoteMCPOAuthStateStore(
        tmp_path, secret_store=secrets_store
    )
    http = _OAuthHttpClient()
    oauth = KDCubeRemoteMCPOAuthService(
        connector_service=connector_service,
        state_store=state_store,
        endpoint_policy=RemoteMCPEndpointPolicy(resolver=_public_resolver),
        connections={},
    )
    monkeypatch.setattr(oauth, "_http_client", lambda: http)

    started = await oauth.start(
        owner_subject="user-1",
        label="Records",
        endpoint="https://mcp.example.test/mcp",
        callback_url="https://hub.example.test/oauth/callback",
        return_hint="https://hub.example.test/settings",
    )

    assert "dynamic-secret" not in repr(started)
    assert "access-value" not in repr(started)
    query = parse_qs(urlsplit(started["authorize_url"]).query)
    assert query["client_id"] == ["dynamic-client"]
    assert query["resource"] == ["https://mcp.example.test/mcp"]
    assert query["scope"] == ["records.read offline_access"]
    assert query["code_challenge_method"] == ["S256"]
    state = query["state"][0]

    completed = await oauth.complete(
        state=state,
        code="authorization-code",
        callback_url="https://hub.example.test/oauth/callback",
        issuer="https://auth.example.test",
    )

    assert completed["connector"].label == "Records"
    assert completed["return_hint"] == "https://hub.example.test/settings"
    assert len(connector_service.calls) == 1
    create = connector_service.calls[0]
    assert create["credential_mode"] == "oauth"
    credential = RemoteMCPOAuthCredential.from_json(create["credential_value"])
    assert credential.access_token == "access-value"
    assert credential.refresh_token == "refresh-value"
    assert credential.client_secret == "dynamic-secret"
    token_call = next(
        call for call in http.calls if call["url"] == "https://auth.example.test/token"
    )
    assert token_call["data"]["client_secret"] == "dynamic-secret"
    assert token_call["data"]["code_verifier"]
    assert secrets_store.values == {}

    with pytest.raises(RemoteMCPOAuthStateError):
        await oauth.complete(
            state=state,
            code="authorization-code",
            callback_url="https://hub.example.test/oauth/callback",
            issuer="https://auth.example.test",
        )


@pytest.mark.asyncio
async def test_upstream_oauth_rejects_callback_issuer_mismatch(
    tmp_path,
    monkeypatch,
):
    secrets_store = _Secrets()
    oauth = KDCubeRemoteMCPOAuthService(
        connector_service=_ConnectorService(),
        state_store=BundleStorageRemoteMCPOAuthStateStore(
            tmp_path, secret_store=secrets_store
        ),
        endpoint_policy=RemoteMCPEndpointPolicy(resolver=_public_resolver),
        connections={},
    )
    http = _OAuthHttpClient()
    monkeypatch.setattr(oauth, "_http_client", lambda: http)
    started = await oauth.start(
        owner_subject="user-1",
        label="Records",
        endpoint="https://mcp.example.test/mcp",
        callback_url="https://hub.example.test/oauth/callback",
    )
    state = parse_qs(urlsplit(started["authorize_url"]).query)["state"][0]

    with pytest.raises(ValueError, match="oauth_callback_issuer_mismatch"):
        await oauth.complete(
            state=state,
            code="authorization-code",
            callback_url="https://hub.example.test/oauth/callback",
            issuer="https://different.example.test",
        )


@pytest.mark.asyncio
async def test_upstream_oauth_uses_client_metadata_url_without_dynamic_registration(
    tmp_path,
    monkeypatch,
):
    oauth = KDCubeRemoteMCPOAuthService(
        connector_service=_ConnectorService(),
        state_store=BundleStorageRemoteMCPOAuthStateStore(
            tmp_path, secret_store=_Secrets()
        ),
        endpoint_policy=RemoteMCPEndpointPolicy(resolver=_public_resolver),
        connections={},
    )
    http = _OAuthHttpClient(client_metadata_supported=True)
    monkeypatch.setattr(oauth, "_http_client", lambda: http)

    started = await oauth.start(
        owner_subject="user-1",
        label="Records",
        endpoint="https://mcp.example.test/mcp",
        callback_url="https://hub.example.test/oauth/callback",
        client_metadata_url="https://hub.example.test/oauth/client-metadata",
    )

    query = parse_qs(urlsplit(started["authorize_url"]).query)
    assert query["client_id"] == [
        "https://hub.example.test/oauth/client-metadata"
    ]
    assert not any(call["url"].endswith("/register") for call in http.calls)
    assert oauth.client_metadata(
        callback_url="https://hub.example.test/oauth/callback"
    )["redirect_uris"] == ["https://hub.example.test/oauth/callback"]


@pytest.mark.asyncio
async def test_upstream_oauth_uses_provisioned_client_without_cimd_or_dcr(
    tmp_path,
    monkeypatch,
):
    secrets_store = _Secrets()
    connector_service = _ConnectorService()
    state_store = BundleStorageRemoteMCPOAuthStateStore(
        tmp_path, secret_store=secrets_store
    )
    oauth = KDCubeRemoteMCPOAuthService(
        connector_service=connector_service,
        state_store=state_store,
        endpoint_policy=RemoteMCPEndpointPolicy(resolver=_public_resolver),
        connections={},
    )
    http = _OAuthHttpClient(
        registration_supported=False,
        token_auth_methods=("client_secret_basic",),
    )
    monkeypatch.setattr(oauth, "_http_client", lambda: http)

    started = await oauth.start(
        owner_subject="user-1",
        label="Provider console records",
        endpoint="https://mcp.example.test/mcp",
        callback_url="https://hub.example.test/oauth/callback",
        oauth_client_mode=OAUTH_CLIENT_MODE_PROVISIONED,
        oauth_client={
            "client_id": "provider-client",
            "client_secret": "provider-secret",
            "token_endpoint_auth_method": "client_secret_basic",
        },
    )

    assert started["oauth_client_source"] == OAUTH_CLIENT_SOURCE_PROVISIONED
    assert "provider-secret" not in repr(started)
    query = parse_qs(urlsplit(started["authorize_url"]).query)
    assert query["client_id"] == ["provider-client"]
    assert not any(call["url"].endswith("/register") for call in http.calls)
    assert "provider-secret" not in "".join(
        path.read_text(encoding="utf-8") for path in state_store.root.glob("*.json")
    )

    await oauth.complete(
        state=query["state"][0],
        code="authorization-code",
        callback_url="https://hub.example.test/oauth/callback",
        issuer="https://auth.example.test",
    )

    created = connector_service.calls[-1]
    credential = RemoteMCPOAuthCredential.from_json(created["credential_value"])
    assert credential.client_id == "provider-client"
    assert credential.client_secret == "provider-secret"
    assert credential.client_source == OAUTH_CLIENT_SOURCE_PROVISIONED
    token_call = next(
        call for call in http.calls if call["url"] == "https://auth.example.test/token"
    )
    assert token_call["headers"]["Authorization"].startswith("Basic ")
    assert "client_secret" not in token_call["data"]
    assert secrets_store.values == {}


@pytest.mark.asyncio
async def test_upstream_oauth_reconnect_reuses_stored_provisioned_client(
    tmp_path,
    monkeypatch,
):
    stored = RemoteMCPOAuthCredential(
        access_token="existing-access",
        refresh_token="existing-refresh",
        token_endpoint="https://auth.example.test/token",
        authorization_server="https://auth.example.test",
        client_id="provider-client",
        client_secret="provider-secret",
        token_endpoint_auth_method="client_secret_post",
        client_source=OAUTH_CLIENT_SOURCE_PROVISIONED,
    )
    connector_service = _ConnectorService(existing_oauth=stored)
    oauth = KDCubeRemoteMCPOAuthService(
        connector_service=connector_service,
        state_store=BundleStorageRemoteMCPOAuthStateStore(
            tmp_path, secret_store=_Secrets()
        ),
        endpoint_policy=RemoteMCPEndpointPolicy(resolver=_public_resolver),
        connections={},
    )
    http = _OAuthHttpClient(registration_supported=False)
    monkeypatch.setattr(oauth, "_http_client", lambda: http)

    started = await oauth.start(
        owner_subject="user-1",
        label="ignored",
        endpoint="https://ignored.example.test/mcp",
        callback_url="https://hub.example.test/oauth/callback",
        connector_id="mcp_0123456789abcdef01234567",
        expected_revision=7,
    )

    query = parse_qs(urlsplit(started["authorize_url"]).query)
    assert query["client_id"] == ["provider-client"]
    assert started["oauth_client_source"] == OAUTH_CLIENT_SOURCE_PROVISIONED
    assert not any(call["url"].endswith("/register") for call in http.calls)


@pytest.mark.asyncio
async def test_upstream_oauth_can_replace_provisioned_client_with_automatic_registration(
    tmp_path,
    monkeypatch,
):
    stored = RemoteMCPOAuthCredential(
        access_token="existing-access",
        refresh_token="existing-refresh",
        token_endpoint="https://auth.example.test/token",
        authorization_server="https://auth.example.test",
        client_id="provider-client",
        client_secret="provider-secret",
        token_endpoint_auth_method="client_secret_post",
        client_source=OAUTH_CLIENT_SOURCE_PROVISIONED,
    )
    oauth = KDCubeRemoteMCPOAuthService(
        connector_service=_ConnectorService(existing_oauth=stored),
        state_store=BundleStorageRemoteMCPOAuthStateStore(
            tmp_path, secret_store=_Secrets()
        ),
        endpoint_policy=RemoteMCPEndpointPolicy(resolver=_public_resolver),
        connections={},
    )
    http = _OAuthHttpClient()
    monkeypatch.setattr(oauth, "_http_client", lambda: http)

    started = await oauth.start(
        owner_subject="user-1",
        label="ignored",
        endpoint="https://ignored.example.test/mcp",
        callback_url="https://hub.example.test/oauth/callback",
        connector_id="mcp_0123456789abcdef01234567",
        expected_revision=7,
        oauth_client_mode=OAUTH_CLIENT_MODE_AUTOMATIC,
    )

    query = parse_qs(urlsplit(started["authorize_url"]).query)
    assert query["client_id"] == ["dynamic-client"]
    assert started["oauth_client_source"] == OAUTH_CLIENT_SOURCE_DCR
    assert any(call["url"].endswith("/register") for call in http.calls)


@pytest.mark.asyncio
async def test_upstream_oauth_rejects_unadvertised_provisioned_auth_method(
    tmp_path,
    monkeypatch,
):
    oauth = KDCubeRemoteMCPOAuthService(
        connector_service=_ConnectorService(),
        state_store=BundleStorageRemoteMCPOAuthStateStore(
            tmp_path, secret_store=_Secrets()
        ),
        endpoint_policy=RemoteMCPEndpointPolicy(resolver=_public_resolver),
        connections={},
    )
    http = _OAuthHttpClient(registration_supported=False)
    monkeypatch.setattr(oauth, "_http_client", lambda: http)

    with pytest.raises(
        RemoteMCPOAuthFlowError,
        match="oauth_provisioned_client_auth_method_unsupported",
    ):
        await oauth.start(
            owner_subject="user-1",
            label="Records",
            endpoint="https://mcp.example.test/mcp",
            callback_url="https://hub.example.test/oauth/callback",
            oauth_client_mode=OAUTH_CLIENT_MODE_PROVISIONED,
            oauth_client={
                "client_id": "provider-client",
                "client_secret": "provider-secret",
                "token_endpoint_auth_method": "client_secret_basic",
            },
        )


@pytest.mark.asyncio
async def test_upstream_oauth_reconnect_replaces_existing_connector_credential(
    tmp_path,
    monkeypatch,
):
    connector_service = _ConnectorService()
    oauth = KDCubeRemoteMCPOAuthService(
        connector_service=connector_service,
        state_store=BundleStorageRemoteMCPOAuthStateStore(
            tmp_path, secret_store=_Secrets()
        ),
        endpoint_policy=RemoteMCPEndpointPolicy(resolver=_public_resolver),
        connections={},
    )
    http = _OAuthHttpClient()
    monkeypatch.setattr(oauth, "_http_client", lambda: http)
    started = await oauth.start(
        owner_subject="user-1",
        label="ignored",
        endpoint="https://ignored.example.test/mcp",
        callback_url="https://hub.example.test/oauth/callback",
        connector_id="mcp_0123456789abcdef01234567",
        expected_revision=7,
    )
    state = parse_qs(urlsplit(started["authorize_url"]).query)["state"][0]

    await oauth.complete(
        state=state,
        code="authorization-code",
        callback_url="https://hub.example.test/oauth/callback",
        issuer="https://auth.example.test",
    )

    replacement = connector_service.calls[-1]
    assert replacement["method"] == "replace_credential"
    assert replacement["connector_id"] == "mcp_0123456789abcdef01234567"
    assert replacement["expected_revision"] == 7
    assert replacement["credential_mode"] == "oauth"
    assert RemoteMCPOAuthCredential.from_json(
        replacement["credential_value"]
    ).access_token == "access-value"
