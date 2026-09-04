# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Live acceptance for one resident Card with multiple Gateway resources.

This opt-in deployment test creates a disposable Cognito owner and authenticated
Remote MCP fixture. It grants an external fixture and the deployment's managed
Knowledge MCP to one resident ``workspace/main`` Card, then exercises both
through the one aggregate delegated MCP Gateway. Everything owned by the test
is removed in ``finally``.

The interpreter needs ``boto3``, ``httpx``, ``pycognito``, and MCP SDK v2. AWS
credentials must permit temporary-user administration in the selected Cognito
pool. The local Docker runtime must already be staged and healthy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import secrets
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

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
from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.resident_gateway import (
    ConnectionHubResidentCardResolver,
    ConnectionHubResidentGatewayFactsLoader,
    ResidentGatewayEndpoints,
)
from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub._resident_gateway_transport import (
    gateway_tools_from_observation,
    read_resident_gateway_access,
    read_resident_gateway_tools,
)
from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources import (
    AuthoritySource,
    ResidentAgentCeiling,
    ResourceFamilyCeiling,
    ResourceBinding,
    delegated_card_snapshot_from_view,
    resolve_current_resident_runtime_projection,
)
from kdcube_ai_app.apps.chat.sdk.runtime.resident_resources.gateway_adapter import (
    gateway_resident_projection,
)
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers import (
    NamedServiceResponse,
)


APPLICATION = "workspace@2026-03-31-13-36"
AGENT = "main"
CLIENT_ID = f"kdcube-agent:{APPLICATION}:{AGENT}"
MANAGED_KNOWLEDGE_SUFFIX = "/knowledge@1-0/public/mcp/knowledge_managed*"


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
    parser.add_argument(
        "--fixture-container", default="kdcube-resident-gateway-fixture"
    )
    parser.add_argument("--fixture-image", default="kdcube-chat-proc:latest")
    parser.add_argument("--fixture-port", type=int, default=8766)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument(
        "--restart-workdir",
        type=Path,
        default=None,
        help=(
            "Opt in to active-state and revocation durability checks by "
            "restarting this initialized local KDCube workdir twice"
        ),
    )
    parser.add_argument(
        "--restart-kdcube-cli",
        type=Path,
        default=None,
        help="Exact KDCube CLI executable used with --restart-workdir",
    )
    parser.add_argument("--restart-timeout-seconds", type=float, default=180)
    args = parser.parse_args()
    if (args.restart_workdir is None) != (args.restart_kdcube_cli is None):
        parser.error(
            "--restart-workdir and --restart-kdcube-cli must be provided together"
        )
    if args.restart_timeout_seconds <= 0:
        parser.error("--restart-timeout-seconds must be positive")
    return args


def _gateway_urls(owner: OwnerOperations) -> ResidentGatewayEndpoints:
    return ResidentGatewayEndpoints(
        mcp=f"{owner.base}/public/mcp/delegated_mcp_gateway",
        access=f"{owner.base}/public/delegated_mcp_gateway_access?include_requestable=true",
    )


def _tool_payload(tool: Any) -> Mapping[str, Any]:
    if isinstance(tool, Mapping):
        return tool
    dump = getattr(tool, "model_dump", None)
    if callable(dump):
        payload = dump(mode="json", by_alias=True, exclude_none=True)
        if isinstance(payload, Mapping):
            return payload
    raise AssertionError("Gateway returned a tool with no public mapping")


def _gateway_routes(tools: list[Any]) -> dict[tuple[str, str], str]:
    routes: dict[tuple[str, str], str] = {}
    for tool in tools:
        payload = _tool_payload(tool)
        raw_meta = payload.get("_meta") or payload.get("meta") or {}
        meta = raw_meta if isinstance(raw_meta, Mapping) else {}
        raw_route = meta.get("connection_hub") or {}
        route = raw_route if isinstance(raw_route, Mapping) else {}
        resource = str(route.get("resource_id") or "").strip()
        operation = str(route.get("operation") or "").strip()
        name = str(payload.get("name") or "").strip()
        if resource and operation and name:
            key = (resource, operation)
            assert key not in routes, key
            routes[key] = name
    return routes


async def _exercise_gateway(
    endpoints: ResidentGatewayEndpoints,
    token: str,
    *,
    external_resource: str,
    managed_resource: str,
    external_query: str,
) -> None:
    async with _delegated_session(endpoints.mcp, token) as session:
        listed = await session.list_tools()
        routes = _gateway_routes(list(listed.tools))
        assert set(routes) == {
            (external_resource, "search"),
            (managed_resource, "about"),
        }, routes

        external_result = await session.call_tool(
            routes[(external_resource, "search")],
            {"query": external_query},
        )
        assert not _is_error(external_result), _structured(external_result)
        external_envelope = _structured(external_result)
        assert external_envelope.get("structured_content", {}).get(
            "upstream_credential_verified"
        ) is True, external_envelope

        managed_result = await session.call_tool(
            routes[(managed_resource, "about")],
            {},
        )
        assert not _is_error(managed_result), _structured(managed_result)
        managed_envelope = _structured(managed_result)
        assert managed_envelope.get("_meta", {}).get("connection_hub", {}).get(
            "resource_id"
        ) == managed_resource, managed_envelope


def _owner_card(owner: OwnerOperations, *, access_id: str) -> dict[str, Any]:
    listing = owner.call("GET", "delegated_access_list")
    assert listing.get("ok") is True, listing
    matches = [
        dict(item)
        for item in listing.get("items") or []
        if str(item.get("access_id") or "") == access_id
        and str(item.get("client_id") or "") == CLIENT_ID
    ]
    assert len(matches) == 1, matches
    return matches[0]


def _managed_knowledge_option(owner: OwnerOperations) -> dict[str, Any]:
    listing = owner.call("GET", "delegated_access_list")
    assert listing.get("ok") is True, listing
    matches = [
        dict(option)
        for option in listing.get("resources") or []
        if str(option.get("resource") or "").endswith(MANAGED_KNOWLEDGE_SUFFIX)
    ]
    assert len(matches) == 1, matches
    option = matches[0]
    operations = {
        str(item.get("name") or "")
        for item in option.get("operations") or []
        if isinstance(item, Mapping)
    }
    assert "about" in operations, operations
    assert "knowledge:read" in set(option.get("grants") or []), option
    return option


def _set_always(
    owner: OwnerOperations, *, access_id: str, resource: str, operation: str
) -> None:
    result = owner.call(
        "POST",
        "delegated_invocation_policy_set",
        {
            "access_id": access_id,
            "resource": resource,
            "operation": operation,
            "mode": "always",
        },
    )
    assert result.get("ok") is True, result
    assert result.get("policy", {}).get("mode") == "always", result


def _gateway_access(
    endpoints: ResidentGatewayEndpoints, token: str
) -> dict[str, Any]:
    response = httpx.get(
        endpoints.access,
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
        follow_redirects=False,
    )
    response.raise_for_status()
    payload = response.json()
    payload = payload.get("delegated_mcp_gateway_access", payload)
    assert isinstance(payload, Mapping), payload
    assert payload.get("ok") is True, payload
    return dict(payload["access"])


def _managed_knowledge_url(
    args: argparse.Namespace, *, managed_resource: str
) -> str:
    marker = "/api/integrations/bundles/*/*/"
    assert marker in managed_resource, managed_resource
    suffix = managed_resource.split(marker, 1)[1].removesuffix("*")
    return (
        f"{args.base_url.rstrip('/')}/api/integrations/bundles/"
        f"{args.tenant}/{args.project}/{suffix}"
    )


def _workspace_capabilities(
    workspace: OwnerOperations,
    *,
    timeout_seconds: float = 60,
) -> dict[str, Any]:
    """Read the staged Workspace capability view once its bundle is ready."""

    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    last_error = "application_not_ready"
    while time.monotonic() < deadline:
        try:
            result = workspace.call(
                "POST",
                "agent_capabilities",
                {"agent": AGENT},
            )
            if result.get("ok") is True:
                return result
            last_error = str(result.get("error") or "not_ready")
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in {409, 503}:
                raise
            try:
                body = exc.response.json()
                last_error = str(
                    body.get("type") or body.get("error") or "not_ready"
                )
            except ValueError:
                last_error = f"http_{exc.response.status_code}"
        time.sleep(0.25)
    raise RuntimeError(
        f"Workspace did not return agent capabilities: {last_error}"
    )


def _durable_card_snapshot(card: Mapping[str, Any]) -> dict[str, Any]:
    """Return only committed authority whose equality must survive restart."""

    resource_grants = {
        str(resource): sorted(str(grant) for grant in grants)
        for resource, grants in dict(card.get("resource_grants") or {}).items()
    }
    resource_operations = {
        str(resource): sorted(str(operation) for operation in operations)
        for resource, operations in dict(
            card.get("resource_operations") or {}
        ).items()
    }
    policies = sorted(
        (
            json.loads(json.dumps(policy, sort_keys=True))
            for policy in card.get("invocation_policies") or []
            if isinstance(policy, Mapping)
        ),
        key=lambda policy: json.dumps(policy, sort_keys=True),
    )
    return {
        "access_id": str(card.get("access_id") or ""),
        "client_id": str(card.get("client_id") or ""),
        "card_revision": int(card.get("card_revision") or 0),
        "identity_scope": str(card.get("identity_scope") or ""),
        "catalog_version": str(card.get("catalog_version") or ""),
        "created_at": int(card.get("created_at") or 0),
        "expires_at": int(card.get("expires_at") or 0),
        "source": str(card.get("source") or ""),
        "resource_grants": resource_grants,
        "resource_operations": resource_operations,
        "resource_acceptance": json.loads(
            json.dumps(card.get("resource_acceptance") or {}, sort_keys=True)
        ),
        "invocation_policies": policies,
    }


def _run_lifecycle_command(command: list[str]) -> None:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return
    detail = (result.stderr or result.stdout or "no command output").strip()
    raise RuntimeError(
        f"KDCube lifecycle command failed ({result.returncode}): "
        f"{' '.join(command[:2])}: {detail[-4000:]}"
    )


def _restart_runtime(
    args: argparse.Namespace,
    *,
    source_root: Path,
) -> None:
    if args.restart_workdir is None or args.restart_kdcube_cli is None:
        raise RuntimeError("local restart was not configured")
    workdir = args.restart_workdir.expanduser().resolve()
    cli = args.restart_kdcube_cli.expanduser().resolve()
    if not (workdir / "config" / ".env").is_file():
        raise RuntimeError(f"KDCube workdir is not initialized: {workdir}")
    if not cli.is_file():
        raise RuntimeError(f"KDCube CLI does not exist: {cli}")

    common = ["--workdir", str(workdir), "--path", str(source_root)]
    _run_lifecycle_command([str(cli), "stop", *common])
    try:
        _run_lifecycle_command([str(cli), "start", *common])
    except Exception:
        # A second start is a recovery attempt, not a hidden acceptance retry.
        # The original error is retained when recovery succeeds.
        try:
            _run_lifecycle_command([str(cli), "start", *common])
        except Exception:
            pass
        raise


def _wait_after_restart(
    owner: OwnerOperations,
    *,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "runtime_not_ready"
    while time.monotonic() < deadline:
        try:
            owner.wait_ready(
                timeout_seconds=min(5.0, max(1.0, deadline - time.monotonic()))
            )
            return
        except (httpx.HTTPError, RuntimeError) as exc:
            last_error = str(exc)
            time.sleep(0.5)
    raise RuntimeError(f"KDCube did not recover after restart: {last_error}")


def _assert_workspace_projection(
    result: Mapping[str, Any],
    *,
    access_id: str,
    resources: set[str],
    bearer: str,
) -> None:
    catalog = result.get("capabilities")
    assert isinstance(catalog, Mapping), result
    raw_rows = catalog.get("resources")
    assert isinstance(raw_rows, list), catalog
    rows = [dict(row) for row in raw_rows if isinstance(row, Mapping)]
    assert {str(row.get("resource_id") or "") for row in rows} == resources, rows
    assert all(row.get("available") is True for row in rows), rows
    assert {str(row.get("access_id") or "") for row in rows} == {access_id}, rows

    bindings = [
        dict(row.get("binding") or {})
        for row in rows
        if isinstance(row.get("binding"), Mapping)
    ]
    assert len(bindings) == len(rows), rows
    assert {str(binding.get("mode") or "") for binding in bindings} == {
        "gateway"
    }, bindings
    assert len({str(binding.get("server_id") or "") for binding in bindings}) == 1
    assert len({str(binding.get("endpoint") or "") for binding in bindings}) == 1

    direct_delegated = [
        row
        for row in catalog.get("mcp") or []
        if isinstance(row, Mapping)
        and str(row.get("authority_source") or "") == "delegated_card"
    ]
    assert not direct_delegated, direct_delegated
    assert bearer not in json.dumps(result, sort_keys=True)


def _projection_ceiling(
    args: argparse.Namespace, *, managed_resource: str
) -> ResidentAgentCeiling:
    return ResidentAgentCeiling(
        tenant=args.tenant,
        project=args.project,
        application=APPLICATION,
        agent_id=AGENT,
        declared_resource_ids=(managed_resource,),
        resource_families=(
            ResourceFamilyCeiling(
                family_id="user_external_mcp",
                resource_kinds=("remote_mcp",),
                authority_sources=(AuthoritySource.DELEGATED_CARD,),
                transports=("streamable-http",),
                resource_patterns=("urn:connection-hub:remote-mcp:*",),
                allowed_tool_patterns=("*",),
                max_resources=8,
                max_tools_per_resource=64,
            ),
        ),
    )


async def _run(args: argparse.Namespace) -> None:
    upstream_bearer = secrets.token_urlsafe(32)
    fixture_args = argparse.Namespace(**vars(args))
    fixture = Fixture(fixture_args, upstream_bearer)
    owner_identity = DisposableOwner(
        region=args.cognito_region,
        pool_id=args.cognito_pool,
        client_id=args.cognito_client,
        label="Connection Hub resident Gateway live acceptance",
    )
    owner: OwnerOperations | None = None
    workspace: OwnerOperations | None = None
    connector: dict[str, Any] | None = None
    access_id = ""

    try:
        fixture.start(1)
        owner_headers = owner_identity.authenticate()
        owner = OwnerOperations(
            base_url=args.base_url,
            tenant=args.tenant,
            project=args.project,
            bundle_id=args.bundle_id,
            headers=owner_headers,
        )
        workspace = OwnerOperations(
            base_url=args.base_url,
            tenant=args.tenant,
            project=args.project,
            bundle_id=APPLICATION,
            headers=owner_headers,
        )
        owner.wait_ready()

        created = owner.call(
            "POST",
            "remote_mcp_connector_create",
            {
                "label": "Resident Gateway live fixture",
                "endpoint": fixture.endpoint,
                "credential_mode": "bearer",
                "credential_value": upstream_bearer,
            },
        )
        assert created.get("ok") is True, created
        connector = dict(created["connector"])
        assert upstream_bearer not in json.dumps(connector)
        external_resource = str(connector["resource"])
        print("PASS disposable external resource and hidden upstream credential")

        external = owner.call(
            "POST",
            "delegated_agent_grant_create",
            {
                "client_id": CLIENT_ID,
                "label": "Workspace main live acceptance",
                "resource": external_resource,
                "claims": ["external_mcp:use"],
                "resource_operations": {external_resource: ["search"]},
                "ttl_seconds": 600,
            },
        )
        assert external.get("ok") is True, external
        token = str(external.get("access_token") or "")
        access = dict(external["access"])
        access_id = str(access["access_id"])
        assert token
        assert set(access.get("resource_grants") or {}) == {external_resource}
        first_card = _owner_card(owner, access_id=access_id)
        assert first_card.get("stable_identity") is True, first_card
        assert (
            first_card.get("caller_profile", {}).get("application")
            == APPLICATION
        )
        _set_always(
            owner,
            access_id=access_id,
            resource=external_resource,
            operation="search",
        )
        print("PASS first resource on the stable resident Card")

        knowledge_option = _managed_knowledge_option(owner)
        managed_resource = str(knowledge_option["resource"])
        managed = owner.call(
            "POST",
            "delegated_agent_grant_create",
            {
                "client_id": CLIENT_ID,
                "resource": managed_resource,
                "claims": ["knowledge:read"],
                "resource_operations": {managed_resource: ["about"]},
                "ttl_seconds": 600,
            },
        )
        assert managed.get("ok") is True, managed
        assert managed.get("access", {}).get("access_id") == access_id, managed
        merged_token = str(managed.get("access_token") or "")
        assert merged_token
        token = merged_token
        _set_always(
            owner,
            access_id=access_id,
            resource=managed_resource,
            operation="about",
        )

        card = _owner_card(owner, access_id=access_id)
        assert card.get("stable_identity") is True, card
        assert set(card.get("resource_grants") or {}) == {
            external_resource,
            managed_resource,
        }
        print("PASS one stable access_id contains external and managed resources")

        async with _delegated_session(
            _managed_knowledge_url(args, managed_resource=managed_resource),
            token,
        ) as managed_session:
            managed_tools = await managed_session.list_tools()
            assert "about" in {tool.name for tool in managed_tools.tools}
        print("PASS managed Knowledge accepts the same resident Card credential")

        endpoints = _gateway_urls(owner)
        gateway_access = _gateway_access(endpoints, token)
        assert gateway_access.get("caller", {}).get("access_id") == access_id
        assert {
            item["resource_id"] for item in gateway_access.get("resources") or []
        } == {external_resource, managed_resource}
        unavailable = {
            item["resource_id"]: item.get("unavailable_reason")
            for item in gateway_access.get("resources") or []
            if item.get("unavailable_reason")
        }
        assert not unavailable, unavailable

        await _exercise_gateway(
            endpoints,
            token,
            external_resource=external_resource,
            managed_resource=managed_resource,
            external_query="resident gateway live acceptance",
        )
        print("PASS one aggregate Gateway lists and invokes both providers")

        live_card = _owner_card(owner, access_id=access_id)

        async def card_lookup(**kwargs: Any) -> Any:
            assert kwargs["bundle_id"] == args.bundle_id
            request = kwargs["request"]
            assert request["payload"] == {
                "client_id": CLIENT_ID,
                "access_id": access_id,
            }
            response = NamedServiceResponse.coerce(
                owner.call("POST", "named_service", request)
            )
            return SimpleNamespace(value=response)

        resolver = ConnectionHubResidentCardResolver(
            connection_hub_bundle_id=args.bundle_id,
            named_service_caller=card_lookup,
        )
        resolved = await resolver.resolve(
            grantor_subject=str(live_card["grantor_subject"]),
            application=APPLICATION,
            agent_id=AGENT,
        )
        assert resolved is not None
        assert resolved.card.access_id == access_id
        assert {item.resource for item in resolved.card.resources} == {
            external_resource,
            managed_resource,
        }
        card_snapshot = delegated_card_snapshot_from_view(
            resolved.card,
            tenant=args.tenant,
            project=args.project,
        )
        observed_access = await read_resident_gateway_access(
            endpoints.access,
            resolved.access_token,
        )
        observed_tools = await read_resident_gateway_tools(
            endpoints.mcp,
            resolved.access_token,
        )
        observed_gateway_tools = gateway_tools_from_observation(
            observed_tools,
            card=card_snapshot,
            access=observed_access,
        )
        try:
            gateway_resident_projection(
                card=card_snapshot,
                tools=observed_gateway_tools,
                access=observed_access,
                binding=ResourceBinding(
                    mode="gateway",
                    server_id="connection_hub_gateway_live",
                    alias="connection_hub_gateway_live",
                    transport="streamable-http",
                    endpoint=endpoints.mcp,
                ),
            )
        except Exception as exc:
            access_rows = {
                str(item.get("resource_id") or ""): item
                for item in observed_access.get("resources") or []
                if isinstance(item, Mapping)
            }
            comparison = {
                item.resource: {
                    "card_revision": item.current_revision,
                    "card_digest": item.current_digest,
                    "gateway": access_rows.get(item.resource, {}).get(
                        "current_descriptor"
                    ),
                }
                for item in resolved.card.resources
            }
            raise AssertionError(
                {"projection_join_error": str(exc), "descriptors": comparison}
            ) from exc
        loader = ConnectionHubResidentGatewayFactsLoader(
            card_resolver=resolver,
            endpoints=endpoints,
        )
        facts = await loader.load_current(
            ceiling=_projection_ceiling(
                args,
                managed_resource=managed_resource,
            ),
            grantor_subject=str(live_card["grantor_subject"]),
        )
        assert facts.card is not None, facts.card_unavailable_reasons
        assert facts.candidates, facts
        projection = await resolve_current_resident_runtime_projection(
            loader=loader,
            ceiling=_projection_ceiling(args, managed_resource=managed_resource),
            grantor_subject=str(live_card["grantor_subject"]),
        )
        projected_resources = {
            item.resource_id for item in projection.inventory.resources
        }
        assert projected_resources == {
            external_resource,
            managed_resource,
        }, {
            "projected_resources": sorted(projected_resources),
            "inventory": projection.inventory.to_dict(),
        }
        assert len(projection.gateway_plan.connections) == 1
        connection = projection.gateway_plan.connections[0]
        assert connection.access_id == access_id
        assert set(connection.resource_ids) == {
            external_resource,
            managed_resource,
        }
        assert token not in repr(projection)
        print("PASS per-turn projection yields one credential-free Gateway binding")

        workspace_capabilities = _workspace_capabilities(workspace)
        _assert_workspace_projection(
            workspace_capabilities,
            access_id=access_id,
            resources={external_resource, managed_resource},
            bearer=token,
        )
        print(
            "PASS staged Workspace capability view exposes two resources through "
            "one credential-free Gateway binding"
        )

        if args.restart_workdir is not None:
            before_restart = _durable_card_snapshot(live_card)
            _restart_runtime(args, source_root=fixture.source_root)
            _wait_after_restart(
                owner,
                timeout_seconds=args.restart_timeout_seconds,
            )
            restarted_card = _owner_card(owner, access_id=access_id)
            assert _durable_card_snapshot(restarted_card) == before_restart
            restarted_access = _gateway_access(endpoints, token)
            assert restarted_access.get("caller", {}).get("access_id") == access_id
            assert {
                item["resource_id"]
                for item in restarted_access.get("resources") or []
            } == {external_resource, managed_resource}
            await _exercise_gateway(
                endpoints,
                token,
                external_resource=external_resource,
                managed_resource=managed_resource,
                external_query="same bearer after KDCube restart",
            )
            restarted_workspace = _workspace_capabilities(
                workspace,
                timeout_seconds=args.restart_timeout_seconds,
            )
            _assert_workspace_projection(
                restarted_workspace,
                access_id=access_id,
                resources={external_resource, managed_resource},
                bearer=token,
            )
            live_card = restarted_card
            print(
                "PASS restart preserves the Card, policies, bearer, both "
                "providers, and Workspace projection"
            )

        narrowed = owner.call(
            "POST",
            "delegated_access_update",
            {
                "access_id": access_id,
                "resource_grants": {
                    external_resource: ["external_mcp:use"],
                },
                "resource_operations": {external_resource: ["search"]},
                "expected_card_revision": live_card["card_revision"],
            },
        )
        assert narrowed.get("ok") is True, narrowed
        assert narrowed.get("access", {}).get("access_id") == access_id
        narrowed_card = _owner_card(owner, access_id=access_id)
        assert set(narrowed_card.get("resource_grants") or {}) == {
            external_resource
        }

        narrowed_access = _gateway_access(endpoints, token)
        assert {
            item["resource_id"] for item in narrowed_access.get("resources") or []
        } == {external_resource}
        async with _delegated_session(endpoints.mcp, token) as session:
            relisted = await session.list_tools()
            routes = _gateway_routes(list(relisted.tools))
            assert set(routes) == {(external_resource, "search")}, routes
            still_allowed = await session.call_tool(
                routes[(external_resource, "search")],
                {"query": "same bearer after narrowing"},
            )
            assert not _is_error(still_allowed), _structured(still_allowed)
        print("PASS live narrowing keeps the Card identity and removes one resource")

        revoked_access_id = access_id
        revoked = owner.call(
            "POST", "delegated_access_revoke", {"access_id": access_id}
        )
        assert revoked.get("ok") is True, revoked
        access_id = ""
        rejected = httpx.get(
            endpoints.access,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
            follow_redirects=False,
        )
        assert rejected.status_code in {401, 403, 503}, (
            rejected.status_code,
            rejected.text,
        )
        print("PASS Card revocation blocks the aggregate Gateway")

        if args.restart_workdir is not None:
            _restart_runtime(args, source_root=fixture.source_root)
            _wait_after_restart(
                owner,
                timeout_seconds=args.restart_timeout_seconds,
            )
            rejected_after_restart = httpx.get(
                endpoints.access,
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
                follow_redirects=False,
            )
            assert rejected_after_restart.status_code in {401, 403, 503}, (
                rejected_after_restart.status_code,
                rejected_after_restart.text,
            )
            listing_after_restart = owner.call("GET", "delegated_access_list")
            assert listing_after_restart.get("ok") is True, listing_after_restart
            assert not any(
                str(item.get("access_id") or "") == revoked_access_id
                for item in listing_after_restart.get("items") or []
                if isinstance(item, Mapping)
            ), listing_after_restart
            print("PASS Card revocation remains effective after restart")
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
        if workspace is not None:
            workspace.close()
        owner_identity.delete()
        fixture.stop()
        print("PASS disposable resident Card, owner, connector, and fixture cleanup")


def main() -> None:
    asyncio.run(_run(_arguments()))


if __name__ == "__main__":
    main()
