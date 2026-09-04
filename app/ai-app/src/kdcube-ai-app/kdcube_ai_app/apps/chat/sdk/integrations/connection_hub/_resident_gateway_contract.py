# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Exact resident Card resolution and processor-local Gateway coordinates."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlsplit

from connection_hub.contract import AGENT_GRANT_GET_TOKEN, NAMESPACE
from connection_hub.delegated_credentials.cards.identity import (
    ResidentCallerProfile,
)
from connection_hub.delegated_credentials.cards.read_model import DelegatedCardView

from kdcube_ai_app.apps.chat.sdk.infra.bundle_operations import (
    call_bundle_named_service,
)
from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.connection_edges import (
    DEFAULT_CONNECTION_HUB_BUNDLE_ID,
)
from kdcube_ai_app.apps.chat.sdk.integrations.connection_hub.delegated_mcp import (
    kdcube_runtime_local_base,
)
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers import (
    NamedServiceResponse,
)


_ACCESS_OPERATION = "delegated_mcp_gateway_access"
_MCP_ALIAS = "delegated_mcp_gateway"
_MAX_TOKEN_CHARS = 16_384

NamedServiceCaller = Callable[..., Awaitable[Any]]


class ResidentGatewayHostError(RuntimeError):
    """A trusted host observation failed with one secret-safe reason."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def clean(value: Any) -> str:
    return str(value or "").strip()


def safe_bearer(value: Any) -> str:
    token = clean(value)
    if (
        not token
        or len(token) > _MAX_TOKEN_CHARS
        or "\r" in token
        or "\n" in token
    ):
        raise ResidentGatewayHostError("resident_gateway_credential_invalid")
    return token


def mapping(value: Any, reason: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ResidentGatewayHostError(reason)
    return value


def _origin(value: str) -> str:
    selected = clean(value).rstrip("/")
    parsed = urlsplit(selected)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ResidentGatewayHostError("resident_gateway_runtime_origin_invalid")
    return selected


@dataclass(frozen=True)
class ResidentGatewayEndpoints:
    mcp: str
    access: str


def resident_gateway_endpoints(
    *,
    tenant: str,
    project: str,
    connection_hub_bundle_id: str = DEFAULT_CONNECTION_HUB_BUNDLE_ID,
    runtime_origin: str = "",
) -> ResidentGatewayEndpoints:
    """Build the same Gateway's processor-local MCP and read-model routes."""

    coordinates = tuple(
        clean(value) for value in (tenant, project, connection_hub_bundle_id)
    )
    if any(not value for value in coordinates):
        raise ResidentGatewayHostError("resident_gateway_coordinates_invalid")
    base = _origin(runtime_origin or kdcube_runtime_local_base())
    tenant_part, project_part, bundle_part = (
        quote(value, safe="") for value in coordinates
    )
    prefix = (
        f"{base}/api/integrations/bundles/{tenant_part}/{project_part}/"
        f"{bundle_part}/public"
    )
    return ResidentGatewayEndpoints(
        mcp=f"{prefix}/mcp/{_MCP_ALIAS}",
        access=f"{prefix}/{_ACCESS_OPERATION}?include_requestable=true",
    )


@dataclass(frozen=True, repr=False)
class ResidentCardCredential:
    access_token: str
    card: DelegatedCardView

    def __repr__(self) -> str:
        return (
            "ResidentCardCredential("
            f"access_id={self.card.access_id!r}, "
            f"card_revision={self.card.card_revision!r})"
        )


class ConnectionHubResidentCardResolver:
    """Resolve the one stable Card of a request-bound resident profile."""

    def __init__(
        self,
        *,
        connection_hub_bundle_id: str,
        named_service_caller: NamedServiceCaller = call_bundle_named_service,
    ) -> None:
        bundle_id = clean(connection_hub_bundle_id)
        if not bundle_id:
            raise ResidentGatewayHostError("resident_gateway_bundle_id_invalid")
        self._bundle_id = bundle_id
        self._named_service_caller = named_service_caller

    async def resolve(
        self,
        *,
        grantor_subject: str,
        application: str,
        agent_id: str,
        expected_access_id: str = "",
    ) -> ResidentCardCredential | None:
        try:
            profile = ResidentCallerProfile(
                grantor_subject=grantor_subject,
                application=application,
                agent_id=agent_id,
            )
        except ValueError:
            raise ResidentGatewayHostError("resident_gateway_profile_invalid") from None
        expected = clean(expected_access_id)
        if expected and expected != profile.access_id:
            return None
        try:
            result = await self._named_service_caller(
                bundle_id=self._bundle_id,
                request={
                    "namespace": NAMESPACE,
                    "operation": AGENT_GRANT_GET_TOKEN,
                    "payload": {
                        "client_id": profile.client_id,
                        "access_id": profile.access_id,
                    },
                },
            )
            value = getattr(result, "value", None)
            response = (
                NamedServiceResponse.coerce(value) if value is not None else None
            )
        except Exception:
            raise ResidentGatewayHostError(
                "resident_gateway_credential_unavailable"
            ) from None
        if response is None or not response.ok:
            raise ResidentGatewayHostError("resident_gateway_credential_unavailable")
        if not response.attrs.get("has_token"):
            return None
        payload = mapping(
            response.object,
            "resident_gateway_credential_response_invalid",
        )
        try:
            card = DelegatedCardView.from_dict(
                mapping(payload.get("card"), "resident_gateway_card_invalid")
            )
        except ResidentGatewayHostError:
            raise
        except Exception:
            raise ResidentGatewayHostError("resident_gateway_card_invalid") from None
        if (
            card.profile != profile
            or card.access_id != profile.access_id
            or card.client_id != profile.client_id
            or card.grantor_subject != profile.grantor_subject
            or clean(payload.get("access_id")) != profile.access_id
            or clean(payload.get("client_id")) != profile.client_id
            or (clean(payload.get("identity_scope")) or "grantor")
            != card.identity_scope
            or int(payload.get("card_revision") or 0) != card.card_revision
        ):
            raise ResidentGatewayHostError("resident_gateway_credential_mismatch")
        return ResidentCardCredential(
            access_token=safe_bearer(payload.get("access_token")),
            card=card,
        )


__all__ = [
    "ConnectionHubResidentCardResolver",
    "ResidentCardCredential",
    "ResidentGatewayEndpoints",
    "ResidentGatewayHostError",
    "clean",
    "mapping",
    "resident_gateway_endpoints",
    "safe_bearer",
]
