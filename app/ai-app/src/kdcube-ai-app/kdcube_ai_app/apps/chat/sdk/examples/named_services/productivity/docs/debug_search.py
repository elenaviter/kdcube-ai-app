# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Diagnose Google Docs title discovery through KDCube and Google Drive.

The script is deliberately read-only. It separates two boundaries:

1. the governed ``named_services`` MCP call used by an external agent;
2. the SDK Docs search plus raw Drive metadata queries used by the provider.

Run it from IntelliJ with the repository's chat-processor interpreter. Values
come from the Run Configuration environment or a local ``.env`` beside this
file. The local file is ignored by git. See README.md in this directory.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from kdcube_ai_app.apps.chat.sdk.examples.mcp.two_level_client.two_level import (
    result_payload_from_call_tool,
)
from kdcube_ai_app.apps.chat.sdk.integrations.google.docs_proxy import (
    DOCS_MIME_TYPE,
    execute_google_docs_operation,
)
from kdcube_ai_app.apps.chat.sdk.runtime.mcp.client import open_mcp_client


HERE = Path(__file__).resolve().parent
DRIVE_API = "https://www.googleapis.com/drive/v3"
_SENSITIVE_KEYS = {
    "access_token",
    "authorization",
    "bearer",
    "download_token",
    "refresh_token",
    "token",
}


def _load_local_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(HERE / ".env", override=False)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _integer(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name.casefold() in _SENSITIVE_KEYS:
                result[name] = "<redacted>"
            else:
                result[name] = _redact(item)
        return result
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _print_json(label: str, value: Any) -> None:
    print(f"\n--- {label} ---")
    print(json.dumps(_redact(value), ensure_ascii=False, indent=2, default=str))


def _find_items(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        items = value.get("items")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, Mapping)]
        for key in ("ret", "output", "result", "extra"):
            nested = value.get(key)
            found = _find_items(nested)
            if found:
                return found
    return []


@dataclass(frozen=True)
class Settings:
    query: str
    account_id: str
    limit: int
    mcp_url: str
    mcp_bearer: str
    google_access_token: str

    @classmethod
    def from_env(cls) -> "Settings":
        base_url = _clean(os.getenv("KDCUBE_BASE_URL")).rstrip("/")
        tenant = _clean(os.getenv("KDCUBE_TENANT"))
        project = _clean(os.getenv("KDCUBE_PROJECT"))
        mcp_url = _clean(os.getenv("KDCUBE_MCP_URL"))
        if not mcp_url and base_url and tenant and project:
            mcp_url = (
                f"{base_url}/api/integrations/bundles/{tenant}/{project}/"
                "kdcube-services@1-0/public/mcp/named_services"
            )
        return cls(
            query=_clean(os.getenv("KDCUBE_DOCS_QUERY")) or "26_006",
            account_id=_clean(os.getenv("KDCUBE_DOCS_ACCOUNT_ID")),
            limit=max(
                1,
                min(
                    _integer(os.getenv("KDCUBE_DOCS_LIMIT"), default=20),
                    50,
                ),
            ),
            mcp_url=mcp_url,
            mcp_bearer=_clean(os.getenv("KDCUBE_MCP_BEARER")),
            google_access_token=_clean(os.getenv("GOOGLE_ACCESS_TOKEN")),
        )


def _search_params(settings: Settings, *, query: str) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    if settings.account_id:
        filters["account_id"] = settings.account_id
    return {
        "namespace": "docs",
        "query": query,
        "limit": settings.limit,
        "cursor": "",
        "filters_json": json.dumps(filters),
        "provider": "",
    }


async def _call_mcp_tool(
    client: Any, tool: str, params: Mapping[str, Any]
) -> Mapping[str, Any]:
    result = await client.call_tool(tool, dict(params))
    return result_payload_from_call_tool(result)


async def probe_named_services(settings: Settings) -> None:
    print("\n" + "=" * 78)
    print("STAGE 1 - governed KDCube named-services path")
    print("=" * 78)
    if not settings.mcp_url or not settings.mcp_bearer:
        print(
            "SKIPPED: set KDCUBE_MCP_URL (or base/tenant/project) and "
            "KDCUBE_MCP_BEARER."
        )
        return

    print(f"endpoint: {settings.mcp_url}")
    print(f"query: {settings.query!r}")
    print(f"account_id: {settings.account_id or '<auto-select>'}")
    headers = {"Authorization": f"Bearer {settings.mcp_bearer}"}
    async with open_mcp_client(
        transport="streamable-http",
        endpoint=settings.mcp_url,
        headers=headers,
    ) as client:
        schema = await _call_mcp_tool(
            client,
            "named_services_schema",
            {"namespace": "docs", "object_kind": "docs.document"},
        )
        _print_json("named_services_schema", schema)

        exact = await _call_mcp_tool(
            client,
            "named_services_search",
            _search_params(settings, query=settings.query),
        )
        _print_json("named_services_search exact/prefix", exact)

        recent = await _call_mcp_tool(
            client,
            "named_services_search",
            _search_params(settings, query=""),
        )
        _print_json("named_services_search recent supported documents", recent)

        exact_items = _find_items(exact)
        recent_items = _find_items(recent)
        print(
            "\nMCP summary: "
            f"query_items={len(exact_items)} recent_items={len(recent_items)}"
        )


def _drive_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


async def _drive_list(
    client: httpx.AsyncClient,
    *,
    token: str,
    clause: str,
    limit: int,
) -> dict[str, Any]:
    response = await client.get(
        f"{DRIVE_API}/files",
        headers={"Authorization": f"Bearer {token}"},
        params={
            "q": f"trashed = false and {clause}",
            "pageSize": limit,
            "orderBy": "modifiedTime desc",
            "spaces": "drive",
            "corpora": "user",
            "includeItemsFromAllDrives": "true",
            "supportsAllDrives": "true",
            "fields": (
                "nextPageToken,incompleteSearch,files(id,name,mimeType,driveId,"
                "parents,createdTime,modifiedTime,ownedByMe,webViewLink,"
                "capabilities(canCopy),owners(displayName,emailAddress))"
            ),
        },
    )
    try:
        body = response.json()
    except Exception:
        body = {"text": response.text[:1000]}
    if response.status_code >= 400:
        return {"http_status": response.status_code, "body": body}
    return dict(body) if isinstance(body, Mapping) else {"value": body}


async def probe_sdk_and_drive(settings: Settings) -> None:
    print("\n" + "=" * 78)
    print("STAGE 2 - direct SDK search and raw Drive MIME diagnosis")
    print("=" * 78)
    if not settings.google_access_token:
        print(
            "SKIPPED: set GOOGLE_ACCESS_TOKEN to a short-lived token for the "
            "same connected Google account."
        )
        return

    sdk_result = await execute_google_docs_operation(
        operation="search",
        access_token=settings.google_access_token,
        payload={"query": settings.query, "limit": settings.limit},
    )
    _print_json("SDK execute_google_docs_operation(search)", sdk_result)

    escaped = _drive_escape(settings.query)
    async with httpx.AsyncClient(timeout=30.0) as client:
        about_response = await client.get(
            f"{DRIVE_API}/about",
            headers={"Authorization": f"Bearer {settings.google_access_token}"},
            params={"fields": "user(displayName,emailAddress,permissionId)"},
        )
        try:
            about = about_response.json()
        except Exception:
            about = {"text": about_response.text[:1000]}
        _print_json(
            "Drive identity behind GOOGLE_ACCESS_TOKEN",
            {"http_status": about_response.status_code, "body": about},
        )

        exact_any = await _drive_list(
            client,
            token=settings.google_access_token,
            clause=f"name = '{escaped}'",
            limit=settings.limit,
        )
        _print_json("Drive exact title - any MIME type", exact_any)

        exact_native = await _drive_list(
            client,
            token=settings.google_access_token,
            clause=(
                f"name = '{escaped}' and mimeType = '{DOCS_MIME_TYPE}'"
            ),
            limit=settings.limit,
        )
        _print_json("Drive exact title - native Google Docs only", exact_native)

        prefix_any = await _drive_list(
            client,
            token=settings.google_access_token,
            clause=f"name contains '{escaped}'",
            limit=settings.limit,
        )
        _print_json("Drive title prefix - any MIME type", prefix_any)


async def main() -> int:
    _load_local_env()
    settings = Settings.from_env()
    print("Google Docs discovery debugger (read-only)")
    print(f"query={settings.query!r} limit={settings.limit}")
    print("Credentials are loaded but never printed.")

    if not (settings.mcp_bearer or settings.google_access_token):
        print(
            "\nNo diagnostic credential is configured. Set KDCUBE_MCP_BEARER "
            "for stage 1, GOOGLE_ACCESS_TOKEN for stage 2, or both.",
            file=sys.stderr,
        )
        return 2

    try:
        await probe_named_services(settings)
        await probe_sdk_and_drive(settings)
    except Exception as exc:  # noqa: BLE001
        print(f"\nDiagnostic failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
