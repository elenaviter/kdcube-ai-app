# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
#
# ── mcp_bridge.py ── delegated MCP for a foreign-runtime agent, runtime-neutral ──
#
# Turn an agent's declared `kind: mcp` tool connections into the standard MCP
# server map — ``{server_id: {url, transport, headers}}`` — for THIS turn's
# user, AS this agent. The agent is a "Delegated By KDCube" entity keyed by
# ``application`` + ``agent_id``, so consent is per-agent: a `delegated: true`
# connection is served with the token the user's per-agent grant already bound
# (read through the Connection Hub named service); when the user has not
# consented, the connection is DROPPED (recorded in ``drop_sink`` as
# ``consent_pending``) instead of any blind call. Those drops are the cue to
# raise the house consent CARD (``announce_connect_required``): the same demand
# + chat event a KDCube-MCP 403 raises, so any wrapped runtime gets the chat's
# real grant affordance instead of prose.
#
# This module STOPS at the server map. The binding step — server map → runtime
# tools/config — is the caller's, per runtime: e.g.
# ``kdcube_ai_app.apps.chat.sdk.frameworks.langchain.mcp.load_mcp_tools_from_server_map``
# binds it as LangChain tools for a LangGraph agent, and a Claude Code adapter
# injects it into the generated ``.mcp.json``. No langchain/langgraph is
# imported here.
#
# ACCOUNTING (the honest rule — "marked = counted"): binding a tool via MCP does
# not by itself make it accounted; a tool whose KDCube-side implementation runs
# a marked model call IS metered, a plain lookup is not.

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_mcp import (
    DROP_CONSENT_PENDING,
    connection_resource,
    delegated_client_id_for_agent,
    resolve_mcp_server_map,
)

LOGGER = logging.getLogger("kdcube.foreign_runtime.mcp_bridge")

__all__ = [
    "current_turn_user_sub",
    "agent_grant_bearer_provider",
    "resolve_turn_mcp",
    "connect_required_outcome",
    "announce_connect_required",
    "load_mcp_server_instructions_safe",
]


def current_turn_user_sub(entrypoint: Any) -> str:
    """This turn's user subject, resolved from the BOUND turn context at build
    time — the accounting context (bound around execute_core), else the comm.
    Used to mint the per-user delegated MCP bearer without threading identity
    through the build signatures. Empty when no user is bound (a delegated MCP
    connection then resolves to nothing — no unauthenticated call)."""
    try:
        from kdcube_ai_app.infra.accounting import _get_context
        sub = str((_get_context().to_dict() or {}).get("user_id") or "").strip()
        if sub:
            return sub
    except Exception:
        pass
    # `entrypoint.comm` is a property that BUILDS the communicator and raises
    # when no turn task is bound (e.g. a build outside a turn) — guard the side
    # effect.
    try:
        return str(getattr(entrypoint.comm, "user_id", "") or "").strip()
    except Exception:
        return ""


def agent_grant_bearer_provider(entrypoint: Any, agent_client_id: str):
    """A bearer provider that reads THIS agent's consented per-agent grant token
    from the Connection Hub named service (`agent_grant.get_token`) for the turn's
    user. Returns None on any absence/failure (consent pending / caller unbound /
    hub unreachable) so the delegated connection is dropped and surfaces as a
    consent demand — never a blind call, never a failed build."""
    async def _provider(conn: Mapping[str, Any], user_sub: str) -> Optional[str]:
        del user_sub  # the hub resolves the grantor from the bound turn context
        resource = connection_resource(conn)
        if not resource:
            return None
        try:
            from kdcube_ai_app.apps.chat.sdk.infra.bundle_operations import call_bundle_named_service
            from kdcube_ai_app.apps.chat.sdk.solutions.connections.connection_edges import (
                connection_hub_bundle_id_from_entrypoint,
            )
            from kdcube_ai_app.apps.chat.sdk.solutions.connections.contract import (
                NAMESPACE, AGENT_GRANT_GET_TOKEN,
            )
            from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers import (
                NamedServiceResponse,
            )
            result = await call_bundle_named_service(
                bundle_id=connection_hub_bundle_id_from_entrypoint(entrypoint),
                request={
                    "namespace": NAMESPACE,
                    "operation": AGENT_GRANT_GET_TOKEN,
                    "payload": {"client_id": agent_client_id, "resource": resource},
                },
            )
            value = getattr(result, "value", None)
            response = NamedServiceResponse.coerce(value) if value is not None else None
        except Exception:
            LOGGER.info(
                "[foreign-runtime] agent-grant token lookup failed for %s; "
                "treating as consent-pending.", resource, exc_info=True,
            )
            return None
        if response is None or not response.ok or not response.attrs.get("has_token"):
            return None
        token = str((response.object or {}).get("access_token") or "").strip()
        return token or None
    return _provider


async def resolve_turn_mcp(
    entrypoint: Any,
    connections: List[Dict[str, Any]],
    *,
    agent_id: str,
    application: str,
    user_sub: Optional[str] = None,
    drop_sink: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """The MCP server map for THIS turn's user, AS this agent.

    Wraps ``resolve_mcp_server_map`` with the agent's delegated-client identity
    (``delegated_client_id_for_agent(application, agent_id)`` — consent grants
    and the bound token are keyed by it, so consent is per-agent and the entity
    is listable/revocable in Connection Hub) and the agent-grant bearer
    provider. Non-``mcp`` connections are skipped by the resolver; the caller
    applies any user tool opt-outs by narrowing ``connections`` first.

    ``user_sub``: the turn's user subject; when None it is resolved from the
    bound turn context (``current_turn_user_sub``). ``drop_sink``: pass a dict
    to learn WHY a delegated connection was omitted (the ``DROP_*`` reasons —
    ``consent_pending`` is the caller's cue to shape a connect-required
    outcome). The drop happens BEFORE any server contact. Never raises for a
    consent-shaped absence; the map simply omits the connection."""
    client_id = delegated_client_id_for_agent(application, agent_id)
    sub = user_sub if user_sub is not None else current_turn_user_sub(entrypoint)
    return await resolve_mcp_server_map(
        connections or [],
        user_sub=sub,
        client_id=client_id,
        bearer_provider=agent_grant_bearer_provider(entrypoint, client_id),
        drop_sink=drop_sink,
    )


def connect_required_outcome(
    drop_sink: Optional[Mapping[str, str]],
    *,
    connection_hub_url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Shape the structured connect-required payload from ``consent_pending``
    drops, or None when nothing is pending.

    ``{"status": "connect_required", "connections": [server ids],
    "connection_hub_url": ...}`` — the server ids whose delegated connection was
    dropped because the user has not granted THIS agent; other drop reasons
    (``no_user``, ``provider_error``, ``mint_error``) are operational, not
    consentable, and are excluded. Pure shaping, no side effects: use it to
    learn WHICH connections are pending (e.g. to name them in the turn's
    answer). ``connection_hub_url`` stays a caller-supplied passthrough for
    non-chat callers that relay a link themselves; a chat lane leaves it unset
    and calls ``announce_connect_required`` — the card carries the action and
    the platform builds its deep link."""
    pending = [
        server_id
        for server_id, reason in (drop_sink or {}).items()
        if reason == DROP_CONSENT_PENDING
    ]
    if not pending:
        return None
    return {
        "status": "connect_required",
        "connections": pending,
        "connection_hub_url": connection_hub_url,
    }


def _connection_server_id(conn: Mapping[str, Any]) -> str:
    return str(
        (conn or {}).get("server_id") or (conn or {}).get("server") or (conn or {}).get("name") or ""
    ).strip()


def _connection_claims(conn: Mapping[str, Any]) -> List[str]:
    raw = (conn or {}).get("scopes") or (conn or {}).get("claims") or (conn or {}).get("grants") or []
    if isinstance(raw, str):
        raw = [raw]
    return [str(item).strip() for item in raw if str(item or "").strip()]


def _turn_scope(entrypoint: Any, tenant: str, project: str) -> tuple[str, str]:
    """(tenant, project) for the hub deep link — explicit args first, then the
    bound turn identity, then the entrypoint's settings. Empty is fine: the
    link builder simply yields no URL and the card still acts through its
    structured grant fields."""
    tenant_id = str(tenant or "").strip()
    project_id = str(project or "").strip()
    if not tenant_id or not project_id:
        try:
            from kdcube_ai_app.apps.chat.sdk.runtime.comm_ctx import get_current_user_identity

            identity = get_current_user_identity() or {}
            tenant_id = tenant_id or str(identity.get("tenant_id") or "").strip()
            project_id = project_id or str(identity.get("project_id") or "").strip()
        except Exception:
            pass
    if not tenant_id or not project_id:
        try:
            settings = getattr(entrypoint, "settings", None)
            tenant_id = tenant_id or str(getattr(settings, "TENANT", "") or "").strip()
            project_id = project_id or str(getattr(settings, "PROJECT", "") or "").strip()
        except Exception:
            pass
    return tenant_id, project_id


async def announce_connect_required(
    entrypoint: Any,
    connections: List[Dict[str, Any]],
    drop_sink: Optional[Mapping[str, str]],
    *,
    agent_id: str,
    application: str,
    tenant: str = "",
    project: str = "",
) -> List[Dict[str, Any]]:
    """Raise the house CONSENT CARD for every ``consent_pending`` drop.

    The chat's consent affordance is an EVENT, not prose: each pending
    connection becomes the same ``MCPConsentRequired`` demand a KDCube-MCP 403
    produces, announced through ``announce_agent_consent`` — recorded once per
    conversation, emitted as the ``delegated_to_kdcube.consent`` chat step, and
    rendered by the chat as a card whose action opens the hub's per-agent grant
    flow (Delegated by KDCube) pre-targeted at THIS agent's client id, the
    connection's resource, and its claims.

    The deep link is the PLATFORM's (``connection_hub_grant_url`` over the
    deployment's public base URL); a caller never hand-writes a hub path. When
    the deployment publishes no public base URL the link is simply absent — the
    card still acts, because the scene-contract path builds the hub view from
    the demand's structured fields.

    Returns one descriptor per announced demand
    (``{server_id, tool_name, resource, claims}``); ``[]`` when nothing is
    pending. Best effort — an announce failure never raises into the turn."""
    pending_ids = {
        server_id
        for server_id, reason in (drop_sink or {}).items()
        if reason == DROP_CONSENT_PENDING
    }
    if not pending_ids:
        return []
    try:
        from kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_credentials.consent_denial import (
            connection_hub_grant_url,
        )
        from kdcube_ai_app.apps.chat.sdk.solutions.connections.mcp_consent import (
            announce_agent_consent,
            mcp_consent_from_denial,
        )
    except Exception:
        LOGGER.warning(
            "[foreign-runtime] consent machinery unavailable; connect-required raised no card "
            "(connections=%s)", sorted(pending_ids), exc_info=True,
        )
        return []
    client_id = delegated_client_id_for_agent(application, agent_id)
    tenant_id, project_id = _turn_scope(entrypoint, tenant, project)
    hub_bundle_id = ""
    try:
        from kdcube_ai_app.apps.chat.sdk.solutions.connections.connection_edges import (
            connection_hub_bundle_id_from_entrypoint,
        )

        hub_bundle_id = str(connection_hub_bundle_id_from_entrypoint(entrypoint) or "").strip()
    except Exception:
        hub_bundle_id = ""
    announced: List[Dict[str, Any]] = []
    for conn in connections or []:
        server_id = _connection_server_id(conn)
        if server_id not in pending_ids:
            continue
        resource = connection_resource(conn)
        claims = _connection_claims(conn)
        tool_name = str(conn.get("alias") or conn.get("name") or server_id).strip() or server_id
        hub_url = ""
        try:
            hub_url = connection_hub_grant_url(
                tenant=tenant_id,
                project=project_id,
                client_id=client_id,
                resource=resource,
                claims=claims,
                **({"hub_bundle_id": hub_bundle_id} if hub_bundle_id else {}),
            )
        except Exception:
            LOGGER.info(
                "[foreign-runtime] hub deep link unavailable for %s; the card acts through its "
                "structured grant fields", server_id, exc_info=True,
            )
        consent = mcp_consent_from_denial(
            {"status": 403, "reason": "authority_mismatch"},
            resource=resource,
            claims=claims,
            connection_hub_url=hub_url,
            tool_name=tool_name,
            agent_client_id=client_id,
        )
        await announce_agent_consent(consent)
        announced.append({
            "server_id": server_id,
            "tool_name": tool_name,
            "resource": resource,
            "claims": claims,
        })
    missing = pending_ids - {item["server_id"] for item in announced}
    if missing:
        LOGGER.warning(
            "[foreign-runtime] consent_pending drops with no matching connection declaration: %s "
            "— no card raised for them", sorted(missing),
        )
    return announced


async def load_mcp_server_instructions_safe(
    server_map: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    """Each MCP server's operating guide (its initialize-result ``instructions``)
    — the text MCP-native clients surface to their model — as
    ``{server_id: instructions}``; servers publishing none are absent.

    Wraps ``frameworks.langchain.mcp.load_mcp_server_instructions``: despite its
    module home, that helper is framework-neutral in implementation (it reads
    the negotiated session via the official MCP SDK through
    ``runtime/mcp/client``; the module's langchain imports are function-local
    and never touched by this path). Deferred import + fail-open: any absence
    or failure yields ``{}``, never raises."""
    try:
        from kdcube_ai_app.apps.chat.sdk.frameworks.langchain.mcp import (
            load_mcp_server_instructions,
        )

        return dict(await load_mcp_server_instructions(server_map) or {})
    except Exception:
        LOGGER.info(
            "[foreign-runtime] mcp server-instructions fetch failed (non-fatal)",
            exc_info=True,
        )
        return {}
