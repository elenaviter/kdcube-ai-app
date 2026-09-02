# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Authenticated streamable-HTTP MCP fixture for local acceptance tests."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import uvicorn
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse

from kdcube_ai_app.apps.chat.sdk.runtime.mcp.server import KDCubeMCPServer


def _fixture_version() -> str:
    return str(os.getenv("REMOTE_MCP_FIXTURE_VERSION") or "1").strip() or "1"


def build_app() -> Any:
    version = _fixture_version()
    bearer = str(os.getenv("REMOTE_MCP_FIXTURE_BEARER") or "").strip()
    auth_mode = str(
        os.getenv("REMOTE_MCP_FIXTURE_AUTH") or "bearer"
    ).strip().lower()
    public_base = str(
        os.getenv("REMOTE_MCP_FIXTURE_PUBLIC_BASE")
        or "http://host.docker.internal:8765"
    ).strip().rstrip("/")
    registration_mode = str(
        os.getenv("REMOTE_MCP_FIXTURE_CLIENT_REGISTRATION") or "dcr"
    ).strip().lower()
    access_ttl = max(
        1, int(os.getenv("REMOTE_MCP_FIXTURE_ACCESS_TTL") or "2")
    )
    call_counts = {"search": 0, "delete": 0}
    oauth_counts = {"register": 0, "authorize": 0, "exchange": 0, "refresh": 0, "revoke": 0}
    authorization_codes: dict[str, dict[str, str]] = {}
    access_tokens: dict[str, float] = {}
    refresh_tokens: dict[str, str] = {}
    dynamic_client_id = "fixture-dynamic-client"
    dynamic_client_secret = "fixture-dynamic-secret"
    server = KDCubeMCPServer(
        "Connection Hub acceptance fixture",
        version=version,
        stateless_http=True,
        json_response=True,
    )

    @server.tool(
        name="search",
        description=(
            "Search fixture records"
            if version == "1"
            else "Search fixture records and include matching record details"
        ),
    )
    async def search(query: str) -> dict[str, Any]:
        call_counts["search"] += 1
        return {
            "ok": True,
            "tool": "search",
            "query": query,
            "fixture_version": version,
            "upstream_credential_verified": True,
            "upstream_call_count": call_counts["search"],
        }

    @server.tool(name="delete", description="Delete one fixture record")
    async def delete_record(record_id: str) -> dict[str, Any]:
        call_counts["delete"] += 1
        return {
            "ok": True,
            "tool": "delete",
            "record_id": record_id,
            "fixture_version": version,
            "upstream_credential_verified": True,
            "upstream_call_count": call_counts["delete"],
        }

    mcp_app = server.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
        host="0.0.0.0",
    )

    def oauth_metadata() -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "issuer": public_base,
            "authorization_endpoint": f"{public_base}/authorize",
            "token_endpoint": f"{public_base}/token",
            "revocation_endpoint": f"{public_base}/revoke",
            "scopes_supported": ["records.read", "offline_access"],
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": [
                "none",
                "client_secret_post",
            ],
            "authorization_response_iss_parameter_supported": True,
        }
        if registration_mode == "cimd":
            metadata["client_id_metadata_document_supported"] = True
        else:
            metadata["registration_endpoint"] = f"{public_base}/register"
        return metadata

    async def request_data(scope: Any, receive: Any) -> dict[str, str]:
        raw = (await Request(scope, receive=receive).body()).decode(
            "utf-8", errors="strict"
        )
        return {
            key: values[-1]
            for key, values in parse_qs(raw, keep_blank_values=True).items()
        }

    def issue_tokens(client_id: str) -> dict[str, Any]:
        access_token = f"fixture-access-{secrets.token_urlsafe(18)}"
        refresh_token = f"fixture-refresh-{secrets.token_urlsafe(18)}"
        access_tokens[access_token] = time.time() + access_ttl
        refresh_tokens[refresh_token] = client_id
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "expires_in": access_ttl,
            "scope": "records.read offline_access",
        }

    def valid_client(data: dict[str, str]) -> bool:
        if registration_mode == "cimd":
            client_id = data.get("client_id") or ""
            parsed = urlsplit(client_id)
            return parsed.scheme == "https" and parsed.path not in {"", "/"}
        return (
            data.get("client_id") == dynamic_client_id
            and data.get("client_secret") == dynamic_client_secret
        )

    async def app(scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") == "lifespan":
            await mcp_app(scope, receive, send)
            return
        path = str(scope.get("path") or "")
        if scope.get("type") == "http" and path == "/healthz":
            response = JSONResponse(
                {
                    "ok": True,
                    "version": version,
                    "auth_mode": auth_mode,
                    "oauth": oauth_counts,
                    "active_access_tokens": sum(
                        1 for expiry in access_tokens.values() if expiry > time.time()
                    ),
                    "active_refresh_tokens": len(refresh_tokens),
                }
            )
            await response(scope, receive, send)
            return
        if scope.get("type") == "http" and auth_mode == "oauth":
            if path == "/.well-known/oauth-protected-resource":
                response = JSONResponse(
                    {
                        "resource": f"{public_base}/mcp",
                        "authorization_servers": [public_base],
                        "scopes_supported": ["records.read"],
                    }
                )
                await response(scope, receive, send)
                return
            if path == "/.well-known/oauth-authorization-server":
                response = JSONResponse(oauth_metadata())
                await response(scope, receive, send)
                return
            if path == "/register" and registration_mode == "dcr":
                request = Request(scope, receive=receive)
                try:
                    payload = json.loads(await request.body())
                except (TypeError, ValueError):
                    payload = {}
                redirect_uris = (
                    payload.get("redirect_uris")
                    if isinstance(payload, dict)
                    else None
                )
                if not isinstance(redirect_uris, list) or not redirect_uris:
                    response = JSONResponse(
                        {"error": "invalid_client_metadata"}, status_code=400
                    )
                else:
                    oauth_counts["register"] += 1
                    response = JSONResponse(
                        {
                            "client_id": dynamic_client_id,
                            "client_secret": dynamic_client_secret,
                            "redirect_uris": redirect_uris,
                            "response_types": ["code"],
                            "grant_types": ["authorization_code", "refresh_token"],
                            "token_endpoint_auth_method": "client_secret_post",
                            "application_type": "web",
                        },
                        status_code=201,
                    )
                await response(scope, receive, send)
                return
            if path == "/authorize":
                query = {
                    key: values[-1]
                    for key, values in parse_qs(
                        scope.get("query_string", b"").decode("utf-8"),
                        keep_blank_values=True,
                    ).items()
                }
                redirect_uri = query.get("redirect_uri") or ""
                if (
                    query.get("response_type") != "code"
                    or not redirect_uri
                    or not query.get("state")
                    or not query.get("code_challenge")
                    or query.get("code_challenge_method") != "S256"
                    or (
                        registration_mode == "dcr"
                        and query.get("client_id") != dynamic_client_id
                    )
                ):
                    response = JSONResponse(
                        {"error": "invalid_authorization_request"},
                        status_code=400,
                    )
                    await response(scope, receive, send)
                    return
                code = f"fixture-code-{secrets.token_urlsafe(18)}"
                authorization_codes[code] = {
                    "client_id": query.get("client_id") or "",
                    "redirect_uri": redirect_uri,
                    "code_challenge": query.get("code_challenge") or "",
                }
                oauth_counts["authorize"] += 1
                location = f"{redirect_uri}?{urlencode({'code': code, 'state': query['state'], 'iss': public_base})}"
                response = RedirectResponse(location, status_code=302)
                await response(scope, receive, send)
                return
            if path == "/token":
                data = await request_data(scope, receive)
                grant_type = data.get("grant_type") or ""
                if not valid_client(data):
                    response = JSONResponse(
                        {"error": "invalid_client"}, status_code=401
                    )
                elif grant_type == "authorization_code":
                    record = authorization_codes.pop(data.get("code") or "", None)
                    verifier = data.get("code_verifier") or ""
                    challenge = base64.urlsafe_b64encode(
                        hashlib.sha256(verifier.encode("utf-8")).digest()
                    ).rstrip(b"=").decode("ascii")
                    if (
                        record is None
                        or record["client_id"] != data.get("client_id")
                        or record["redirect_uri"] != data.get("redirect_uri")
                        or record["code_challenge"] != challenge
                    ):
                        response = JSONResponse(
                            {"error": "invalid_grant"}, status_code=400
                        )
                    else:
                        oauth_counts["exchange"] += 1
                        response = JSONResponse(issue_tokens(record["client_id"]))
                elif grant_type == "refresh_token":
                    old_refresh = data.get("refresh_token") or ""
                    client_id = refresh_tokens.pop(old_refresh, "")
                    if not client_id or client_id != data.get("client_id"):
                        response = JSONResponse(
                            {"error": "invalid_grant"}, status_code=400
                        )
                    else:
                        oauth_counts["refresh"] += 1
                        response = JSONResponse(issue_tokens(client_id))
                else:
                    response = JSONResponse(
                        {"error": "unsupported_grant_type"}, status_code=400
                    )
                await response(scope, receive, send)
                return
            if path == "/revoke":
                data = await request_data(scope, receive)
                if not valid_client(data):
                    response = JSONResponse(
                        {"error": "invalid_client"}, status_code=401
                    )
                else:
                    token = data.get("token") or ""
                    access_tokens.pop(token, None)
                    refresh_tokens.pop(token, None)
                    oauth_counts["revoke"] += 1
                    response = JSONResponse({"ok": True})
                await response(scope, receive, send)
                return
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers") or ()
        }
        supplied = str(headers.get("authorization") or "")
        authorized = (
            supplied == f"Bearer {bearer}"
            if auth_mode == "bearer"
            else supplied.startswith("Bearer ")
            and access_tokens.get(supplied.removeprefix("Bearer "), 0)
            > time.time()
        )
        if not authorized:
            response = JSONResponse(
                {"ok": False, "error": "fixture_credential_required"},
                status_code=401,
                headers={
                    "WWW-Authenticate": (
                        f'Bearer resource_metadata="{public_base}/'
                        '.well-known/oauth-protected-resource", '
                        'scope="records.read"'
                    )
                }
                if auth_mode == "oauth"
                else None,
            )
            await response(scope, receive, send)
            return
        await mcp_app(scope, receive, send)

    return app


def main() -> None:
    host = str(os.getenv("REMOTE_MCP_FIXTURE_HOST") or "0.0.0.0")
    port = int(os.getenv("REMOTE_MCP_FIXTURE_PORT") or "8765")
    uvicorn.run(build_app(), host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
