import asyncio

from kdcube_ai_app.apps.chat.sdk.runtime.mcp.mcp_adapter import MCPServerSpec, PythonSDKMCPAdapter


def test_auth_headers_supports_secret_key(monkeypatch):
    async def get_secret(key, default=None):
        if key == "bundles.react.mcp@2026-03-09.secrets.docs.token":
            return "secret-token"
        return default

    monkeypatch.setattr(
        "kdcube_ai_app.apps.chat.sdk.config.get_secret",
        get_secret,
    )
    adapter = PythonSDKMCPAdapter(
        MCPServerSpec(
            server_id="docs",
            display_name="docs",
            transport="http",
            endpoint="https://mcp.example.com",
            auth_profile={
                "type": "bearer",
                "secret": "bundles.react.mcp@2026-03-09.secrets.docs.token",
            },
        )
    )
    assert asyncio.run(adapter._auth_headers()) == {"Authorization": "Bearer secret-token"}
