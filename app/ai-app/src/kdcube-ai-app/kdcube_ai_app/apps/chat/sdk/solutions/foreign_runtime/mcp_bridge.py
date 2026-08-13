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
# ``consent_pending``) instead of any blind call — the caller shapes a
# connect-required outcome from those drops.
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
    consentable, and are excluded. Pure shaping: the hub URL is caller-provided
    (this function does no URL discovery)."""
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
