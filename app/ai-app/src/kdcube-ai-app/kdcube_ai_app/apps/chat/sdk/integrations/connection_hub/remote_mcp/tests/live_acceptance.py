# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Live acceptance for a user-owned remote MCP connector.

This is an opt-in deployment test. It creates a disposable Cognito user,
starts the authenticated MCP fixture beside the local runtime, exercises the
owner and delegated HTTP boundaries, and removes both in ``finally``.

The interpreter needs ``boto3``, ``httpx``, ``pycognito``, and MCP SDK v2.
AWS credentials must permit temporary-user administration in the selected
Cognito pool. The local Docker runtime must already be staged and healthy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Mapping

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

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
    parser.add_argument(
        "--cognito-client", default="6lgsqqbpatprt44a4i20hveu6u"
    )
    parser.add_argument("--fixture-container", default="kdcube-remote-mcp-fixture")
    parser.add_argument("--fixture-image", default="kdcube-chat-proc:latest")
    parser.add_argument("--fixture-port", type=int, default=8765)
    parser.add_argument("--source-root", type=Path, default=None)
    return parser.parse_args()


def _source_root(explicit: Path | None) -> Path:
    if explicit is not None:
        root = explicit.expanduser().resolve()
        if not (root / "app" / "ai-app").is_dir():
            raise RuntimeError(f"KDCube source root is invalid: {root}")
        return root
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "app" / "ai-app").is_dir():
            return candidate
    raise RuntimeError("Pass --source-root with the KDCube repository root")


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


class Fixture:
    def __init__(self, args: argparse.Namespace, bearer: str) -> None:
        self.args = args
        self.bearer = bearer
        self.source_root = _source_root(args.source_root)

    @property
    def endpoint(self) -> str:
        return f"http://host.docker.internal:{self.args.fixture_port}/mcp"

    def start(
        self,
        version: int,
        *,
        auth_mode: str = "bearer",
        registration_mode: str = "dcr",
        access_ttl: int = 2,
    ) -> None:
        self.stop()
        fixture_path = (
            "/source/app/ai-app/src/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/"
            "integrations/connection_hub/remote_mcp/tests/live_fixture_server.py"
        )
        _docker(
            "run",
            "-d",
            "--rm",
            "--name",
            self.args.fixture_container,
            "-p",
            f"127.0.0.1:{self.args.fixture_port}:8765",
            "-e",
            f"REMOTE_MCP_FIXTURE_BEARER={self.bearer}",
            "-e",
            f"REMOTE_MCP_FIXTURE_VERSION={version}",
            "-e",
            f"REMOTE_MCP_FIXTURE_AUTH={auth_mode}",
            "-e",
            f"REMOTE_MCP_FIXTURE_CLIENT_REGISTRATION={registration_mode}",
            "-e",
            f"REMOTE_MCP_FIXTURE_ACCESS_TTL={access_ttl}",
            "-e",
            (
                "REMOTE_MCP_FIXTURE_PUBLIC_BASE="
                f"http://host.docker.internal:{self.args.fixture_port}"
            ),
            "-v",
            f"{self.source_root}:/source:ro",
            self.args.fixture_image,
            "python",
            fixture_path,
        )
        health_url = f"http://localhost:{self.args.fixture_port}/healthz"
        for _ in range(60):
            try:
                response = httpx.get(health_url, timeout=1)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        raise RuntimeError("remote MCP fixture did not become healthy")

    def stop(self) -> None:
        _docker("rm", "-f", self.args.fixture_container, check=False)


@asynccontextmanager
async def _delegated_session(
    proxy_url: str, token: str
) -> AsyncIterator[ClientSession]:
    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {token}"}, timeout=30
    ) as http_client:
        async with streamable_http_client(
            proxy_url, http_client=http_client
        ) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                yield session


def _structured(result: Any) -> dict[str, Any]:
    value = getattr(result, "structured_content", None)
    if value is None:
        value = getattr(result, "structuredContent", None)
    return dict(value) if isinstance(value, Mapping) else {}


def _is_error(result: Any) -> bool:
    value = getattr(result, "is_error", None)
    if value is None:
        value = getattr(result, "isError", None)
    return bool(value)


def _invocation_meta(invocation_id: str) -> dict[str, str]:
    return {"connection_hub/invocation_id": invocation_id}


async def _run(args: argparse.Namespace) -> None:
    upstream_bearer = secrets.token_urlsafe(32)
    fixture = Fixture(args, upstream_bearer)
    owner_identity = DisposableOwner(
        region=args.cognito_region,
        pool_id=args.cognito_pool,
        client_id=args.cognito_client,
        label="Connection Hub remote MCP live acceptance",
    )
    owner: OwnerOperations | None = None
    connector: dict[str, Any] | None = None
    access_id = ""

    try:
        fixture.start(1)
        owner = OwnerOperations(
            base_url=args.base_url,
            tenant=args.tenant,
            project=args.project,
            bundle_id=args.bundle_id,
            headers=owner_identity.authenticate(),
        )
        owner.wait_ready()

        created = owner.call(
            "POST",
            "remote_mcp_connector_create",
            {
                "label": "Live acceptance fixture",
                "endpoint": fixture.endpoint,
                "credential_mode": "bearer",
                "credential_value": upstream_bearer,
            },
        )
        assert created.get("ok") is True, created
        connector = dict(created["connector"])
        assert upstream_bearer not in json.dumps(connector)
        assert connector.get("credential_present") is True
        resource = str(connector["resource"])
        tools = {row["name"]: row for row in connector["tools"]}
        search = tools["search"]
        delete = tools["delete"]
        print("PASS connector creation, authenticated discovery, and secret redaction")

        card = owner.call(
            "POST",
            "delegated_access_create",
            {
                "label": "Live acceptance caller",
                "resource_grants": {resource: ["external_mcp:use"]},
                "resource_operations": {resource: ["search"]},
                "ttl_seconds": 600,
            },
        )
        assert card.get("ok") is True, card
        delegated_token = str(card.get("access_token") or "")
        access_id = str(card["access"]["access_id"])
        caller_client_id = str(card["access"]["client_id"])
        assert delegated_token
        print("PASS exact resource and operation card creation")

        delete_invocation_id = "live-delete-once"
        async with _delegated_session(owner.proxy_url, delegated_token) as session:
            listed = await session.list_tools()
            assert [tool.name for tool in listed.tools] == [search["proxy_name"]]
            allowed = await session.call_tool(
                search["proxy_name"], {"query": "relay lifecycle"}
            )
            assert not _is_error(allowed), _structured(allowed)
            assert _structured(allowed).get("upstream_credential_verified") is True
            denied = await session.call_tool(
                delete["proxy_name"],
                {"record_id": "fixture-1"},
                meta=_invocation_meta(delete_invocation_id),
            )
            assert _is_error(denied)
            denial = _structured(denied)
            assert denial.get("reason") == "operation_not_consented"
            consent = dict(denial.get("consent") or {})
            assert consent.get("agent_client_id") == caller_client_id
            assert consent.get("access_id") == access_id
            assert consent.get("resource") == resource
            assert consent.get("outer_operation") == "delete"
            assert consent.get("invocation_change_id") == delete_invocation_id
            assert consent.get("available_choices") == [
                "allow_once",
                "allow_always",
            ]
            grant = dict(consent.get("grant") or {})
            assert grant.get("operation") == "delegated_agent_grant_create"
            grant_payload = dict(grant.get("payload") or {})
        print("PASS exact unselected-tool denial with once-or-always recovery")

        once_grant = owner.call(
            "POST",
            "delegated_agent_grant_create",
            {
                **grant_payload,
                "invocation_mode": "once",
            },
        )
        assert once_grant.get("ok") is True, once_grant
        assert once_grant.get("access_id") == access_id
        once_policy = dict(once_grant.get("invocation_policy") or {})
        assert once_policy.get("mode") == "once"
        assert once_policy.get("remaining") == 1

        async with _delegated_session(owner.proxy_url, delegated_token) as session:
            first = await session.call_tool(
                delete["proxy_name"],
                {"record_id": "fixture-1"},
                meta=_invocation_meta(delete_invocation_id),
            )
            assert not _is_error(first), _structured(first)
            first_result = _structured(first)
            assert first_result.get("upstream_call_count") == 1

            replay = await session.call_tool(
                delete["proxy_name"],
                {"record_id": "fixture-1"},
                meta=_invocation_meta(delete_invocation_id),
            )
            assert not _is_error(replay), _structured(replay)
            assert _structured(replay) == first_result

            exhausted = await session.call_tool(
                delete["proxy_name"],
                {"record_id": "fixture-2"},
                meta=_invocation_meta("live-delete-next"),
            )
            assert _is_error(exhausted)
            exhausted_body = _structured(exhausted)
            assert exhausted_body.get("reason") == (
                "delegated_invocation_limit_exhausted"
            )
            assert exhausted_body.get("consent", {}).get("access_id") == access_id
            assert exhausted_body.get("consent", {}).get("available_choices") == [
                "allow_once",
                "allow_always",
            ]
        print("PASS allow once, idempotent replay, and next-invocation denial")

        always = owner.call(
            "POST",
            "delegated_invocation_policy_set",
            {
                "access_id": access_id,
                "resource": resource,
                "operation": "delete",
                "mode": "always",
                "expected_revision": once_policy["revision"],
            },
        )
        assert always.get("ok") is True, always
        assert always.get("policy", {}).get("mode") == "always"
        async with _delegated_session(owner.proxy_url, delegated_token) as session:
            repeated = await session.call_tool(
                delete["proxy_name"],
                {"record_id": "fixture-2"},
                meta=_invocation_meta("live-delete-always"),
            )
            assert not _is_error(repeated), _structured(repeated)
            assert _structured(repeated).get("upstream_call_count") == 2
        print("PASS allow always restores repeated invocation authority")

        changed = owner.call(
            "POST",
            "remote_mcp_connector_set_enabled",
            {
                "connector_id": connector["connector_id"],
                "expected_revision": connector["revision"],
                "enabled": False,
            },
        )
        assert changed.get("ok") is True, changed
        connector = dict(changed["connector"])
        async with _delegated_session(owner.proxy_url, delegated_token) as session:
            denied = await session.call_tool(
                search["proxy_name"], {"query": "disabled connector"}
            )
            assert _is_error(denied)
            assert _structured(denied).get("reason") == "connector_not_active"
        print("PASS connector kill switch on the next invocation")

        changed = owner.call(
            "POST",
            "remote_mcp_connector_set_enabled",
            {
                "connector_id": connector["connector_id"],
                "expected_revision": connector["revision"],
                "enabled": True,
            },
        )
        assert changed.get("ok") is True, changed
        connector = dict(changed["connector"])

        fixture.start(2)
        async with _delegated_session(owner.proxy_url, delegated_token) as session:
            denied = await session.call_tool(
                search["proxy_name"], {"query": "changed descriptor"}
            )
            assert _is_error(denied)
            assert _structured(denied).get("reason") == "operation_descriptor_changed"
        refreshed = owner.call(
            "POST",
            "remote_mcp_connector_refresh",
            {
                "connector_id": connector["connector_id"],
                "expected_revision": connector["revision"],
            },
        )
        assert refreshed.get("ok") is True, refreshed
        connector = dict(refreshed["connector"])
        assert connector.get("descriptor_state") == "drifted"
        accepted = owner.call(
            "POST",
            "remote_mcp_connector_accept_descriptor",
            {
                "connector_id": connector["connector_id"],
                "expected_revision": connector["revision"],
            },
        )
        assert accepted.get("ok") is True, accepted
        connector = dict(accepted["connector"])
        async with _delegated_session(owner.proxy_url, delegated_token) as session:
            allowed = await session.call_tool(
                search["proxy_name"], {"query": "accepted descriptor"}
            )
            assert not _is_error(allowed), _structured(allowed)
        print("PASS descriptor drift denial and explicit owner acceptance")

        revoked = owner.call(
            "POST", "delegated_access_revoke", {"access_id": access_id}
        )
        assert revoked.get("ok") is True, revoked
        access_id = ""
        rejected = httpx.post(
            owner.proxy_url,
            headers={
                "Authorization": f"Bearer {delegated_token}",
                "Accept": "application/json, text/event-stream",
            },
            json={
                "jsonrpc": "2.0",
                "id": "revoked-card-proof",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "connection-hub-live-acceptance",
                        "version": "1",
                    },
                },
            },
            timeout=30,
        )
        assert rejected.status_code in {401, 403, 503}, (
            rejected.status_code,
            rejected.text,
        )
        print("PASS card revocation on the next delegated connection")

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
        listed = owner.call("GET", "remote_mcp_connectors_list")
        assert listed.get("ok") is True and not listed.get("items")
        print("PASS connector deletion and empty owner inventory")
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
        print("PASS disposable user and fixture cleanup")


def main() -> None:
    asyncio.run(_run(_arguments()))


if __name__ == "__main__":
    main()
