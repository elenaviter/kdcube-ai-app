# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Live upstream-OAuth acceptance for a user-owned remote MCP connector."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx

from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.remote_mcp.tests.live_acceptance import (
    Fixture,
    _delegated_session,
    _is_error,
    _structured,
)
from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.remote_mcp.tests.live_support import (
    DisposableOwner,
    OwnerOperations,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:5173")
    parser.add_argument("--tenant", default="demo-tenant")
    parser.add_argument("--project", default="demo-project")
    parser.add_argument("--bundle-id", default="connection-hub@1-0")
    parser.add_argument("--cognito-region", default="eu-west-1")
    parser.add_argument("--cognito-pool", default="eu-west-1_JrKKhQUNp")
    parser.add_argument("--cognito-client", default="6lgsqqbpatprt44a4i20hveu6u")
    parser.add_argument("--fixture-container", default="kdcube-remote-mcp-oauth-fixture")
    parser.add_argument("--fixture-image", default="kdcube-chat-proc:latest")
    parser.add_argument("--fixture-port", type=int, default=8765)
    parser.add_argument(
        "--registration-mode",
        choices=("dcr", "cimd", "provisioned"),
        default="dcr",
    )
    parser.add_argument("--source-root", type=Path, default=None)
    return parser.parse_args()


def _browser_url(value: str) -> str:
    return value.replace("http://host.docker.internal:", "http://localhost:", 1)


def _fixture_state(port: int) -> dict[str, Any]:
    response = httpx.get(f"http://localhost:{port}/healthz", timeout=10)
    response.raise_for_status()
    return dict(response.json())


def _assert_no_oauth_secrets(value: Any) -> None:
    secret_keys = {
        "access_token",
        "refresh_token",
        "client_secret",
        "code_verifier",
        "credential_value",
    }

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            assert secret_keys.isdisjoint(node), node.keys()
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(value)


def _complete_browser_flow(authorize_url: str) -> str:
    with httpx.Client(follow_redirects=True, timeout=60) as browser:
        response = browser.get(_browser_url(authorize_url))
        response.raise_for_status()
    body = response.text
    assert "Connection complete" in body, body
    return body


async def _run(args: argparse.Namespace) -> None:
    fixture = Fixture(args, bearer="unused-in-oauth-mode")
    owner_identity = DisposableOwner(
        region=args.cognito_region,
        pool_id=args.cognito_pool,
        client_id=args.cognito_client,
        label="Connection Hub upstream OAuth live acceptance",
    )
    owner: OwnerOperations | None = None
    connector: dict[str, Any] | None = None
    access_id = ""
    try:
        fixture.start(
            1,
            auth_mode="oauth",
            registration_mode=args.registration_mode,
            access_ttl=2,
        )
        owner = OwnerOperations(
            base_url=args.base_url,
            tenant=args.tenant,
            project=args.project,
            bundle_id=args.bundle_id,
            headers=owner_identity.authenticate(),
        )
        owner.wait_ready()

        start_payload: dict[str, Any] = {
            "label": f"OAuth fixture ({args.registration_mode})",
            "endpoint": fixture.endpoint,
            "return_hint": args.base_url,
        }
        if args.registration_mode == "provisioned":
            start_payload.update(
                {
                    "oauth_client_mode": "provisioned",
                    "oauth_client": {
                        "client_id": "fixture-provisioned-client",
                        "client_secret": "fixture-provisioned-secret",
                        "token_endpoint_auth_method": "client_secret_post",
                    },
                }
            )
        started = owner.call(
            "POST",
            "remote_mcp_connector_start_oauth",
            start_payload,
        )
        assert started.get("ok") is True, started
        _assert_no_oauth_secrets(started)
        expected_source = {
            "cimd": "client_metadata_document",
            "dcr": "dynamic_registration",
            "provisioned": "provisioned",
        }[args.registration_mode]
        assert started.get("oauth_client_source") == expected_source, started
        authorize_url = str(started["authorize_url"])
        authorize_query = parse_qs(urlsplit(authorize_url).query)
        if args.registration_mode == "cimd":
            assert authorize_query["client_id"][0].startswith("https://"), (
                authorize_query
            )
            assert "remote_mcp_oauth_client_metadata" in (
                authorize_query["client_id"][0]
            )
        elif args.registration_mode == "provisioned":
            assert authorize_query["client_id"] == [
                "fixture-provisioned-client"
            ], authorize_query
        _complete_browser_flow(authorize_url)

        inventory = owner.call("GET", "remote_mcp_connectors_list")
        assert inventory.get("ok") is True, inventory
        assert len(inventory.get("items") or []) == 1, inventory
        connector = dict(inventory["items"][0])
        assert connector.get("credential_mode") == "oauth"
        assert connector.get("credential_present") is True
        assert "fixture-access" not in json.dumps(connector)
        resource = str(connector["resource"])
        search = next(
            tool for tool in connector["tools"] if tool["name"] == "search"
        )
        print(
            "PASS OAuth discovery, browser authorization, token exchange, "
            "server-side storage, and authenticated MCP discovery"
        )

        card = owner.call(
            "POST",
            "delegated_access_create",
            {
                "label": "OAuth fixture caller",
                "resource_grants": {resource: ["external_mcp:use"]},
                "resource_operations": {resource: [search["name"]]},
                "ttl_seconds": 600,
            },
        )
        assert card.get("ok") is True, card
        delegated_token = str(card["access_token"])
        access_id = str(card["access"]["access_id"])
        async with _delegated_session(owner.proxy_url, delegated_token) as session:
            result = await session.call_tool(
                search["proxy_name"], {"query": "upstream OAuth"}
            )
            assert not _is_error(result), _structured(result)
            assert _structured(result).get("upstream_credential_verified") is True
        before_reconnect = _fixture_state(args.fixture_port)
        assert before_reconnect["oauth"]["refresh"] >= 1, before_reconnect
        if args.registration_mode == "provisioned":
            assert before_reconnect["oauth"]["register"] == 0, before_reconnect
        print("PASS refresh-token rotation before an upstream MCP call")

        reconnect = owner.call(
            "POST",
            "remote_mcp_connector_start_oauth",
            {
                "connector_id": connector["connector_id"],
                "expected_revision": connector["revision"],
                "label": connector["label"],
                "endpoint": connector["endpoint"],
                "return_hint": args.base_url,
            },
        )
        assert reconnect.get("ok") is True, reconnect
        assert reconnect.get("oauth_client_source") == expected_source, reconnect
        _complete_browser_flow(str(reconnect["authorize_url"]))
        inventory = owner.call("GET", "remote_mcp_connectors_list")
        connector = dict(inventory["items"][0])
        assert connector["revision"] >= 2
        reconnected_state = _fixture_state(args.fixture_port)
        assert reconnected_state["oauth"]["revoke"] >= 2, reconnected_state

        async with _delegated_session(owner.proxy_url, delegated_token) as session:
            result = await session.call_tool(
                search["proxy_name"], {"query": "reauthorized connector"}
            )
            assert not _is_error(result), _structured(result)
        print(
            "PASS in-place reauthorization, old upstream grant revocation, "
            "and stable delegated resource authority"
        )

        removed = owner.call(
            "POST",
            "remote_mcp_connector_delete",
            {
                "connector_id": connector["connector_id"],
                "expected_revision": connector["revision"],
            },
        )
        assert removed.get("ok") is True, removed
        connector = None
        time.sleep(3)
        final_state = _fixture_state(args.fixture_port)
        assert final_state["oauth"]["revoke"] >= 4, final_state
        assert final_state["active_access_tokens"] == 0, final_state
        assert final_state["active_refresh_tokens"] == 0, final_state
        print("PASS connector deletion and upstream access/refresh revocation")
    finally:
        if owner is not None:
            if access_id:
                try:
                    owner.call(
                        "POST", "delegated_access_revoke", {"access_id": access_id}
                    )
                except Exception:
                    pass
            if connector is not None:
                try:
                    owner.call(
                        "POST",
                        "remote_mcp_connector_delete",
                        {
                            "connector_id": connector.get("connector_id"),
                            "expected_revision": connector.get("revision"),
                        },
                    )
                except Exception:
                    pass
            owner.close()
        owner_identity.delete()
        fixture.stop()
        print("PASS disposable user and OAuth fixture cleanup")


def main() -> None:
    asyncio.run(_run(_arguments()))


if __name__ == "__main__":
    main()
