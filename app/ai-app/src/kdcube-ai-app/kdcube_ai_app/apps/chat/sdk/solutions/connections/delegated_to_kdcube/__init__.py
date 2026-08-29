# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Lazy compatibility surface for connected-account delegation."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_BASE = "kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_to_kdcube"

# Package import has always registered the built-in adapters. Preserve that
# contract while the provider implementations are owned by Prokura.
import_module(f"{_BASE}.providers")

_EXPORTS: dict[str, str] = {}
for name in (
    "DelegatedToKdcubeAdapter",
    "adapter",
    "list_adapters",
    "register_adapter",
    "resolve_adapter",
):
    _EXPORTS[name] = f"{_BASE}.adapters"
for name in ("DelegatedToKdcubeBroker", "broker_for_user"):
    _EXPORTS[name] = f"{_BASE}.broker"
_EXPORTS["DelegatedToKdcubeClient"] = f"{_BASE}.client"
for name in (
    "CONNECTIONS_CONFIG_KEY",
    "DELEGATED_TO_KDCUBE_CONFIG_KEY",
    "delegated_to_kdcube_config",
    "delegated_to_kdcube_config_from_entrypoint",
):
    _EXPORTS[name] = f"{_BASE}.config"
for name in (
    "CONNECTION_HUB_BUNDLE_ID",
    "CREDENTIAL_ACTIVE",
    "CREDENTIAL_EXPIRES_SOON",
    "CREDENTIAL_MISSING",
    "CREDENTIAL_RECONNECT_REQUIRED",
    "CREDENTIAL_REFRESHABLE",
    "CREDENTIAL_REVOKED",
    "REASON_ACCOUNT_REQUIRED",
    "REASON_AGENT_ACCOUNT_BINDING_REQUIRED",
    "REASON_AGENT_GRANT_REQUIRED",
    "REASON_CLAIM_UPGRADE_REQUIRED",
    "REASON_CONNECT_REQUIRED",
    "REASON_RECONNECT_REQUIRED",
    "STATUS_CONNECTED",
    "STATUS_REVOKED",
    "USER_ACTIONABLE_REASONS",
    "ClaimResolution",
    "account_choice",
    "ConnectedAccount",
    "ConnectorApp",
    "CredentialHandle",
    "IntegrationProvider",
    "ProviderClaim",
    "ToolClaimPolicy",
    "ToolClaimRequirement",
    "DelegatedToKdcubeConfig",
):
    _EXPORTS[name] = f"{_BASE}.models"
for name in (
    "MemoryOAuthStateStore",
    "OAuthStateStore",
    "RedisOAuthStateStore",
    "consume_oauth_state",
    "create_oauth_state",
    "peek_state_payload",
    "sign_state",
    "state_digest",
    "verify_state",
):
    _EXPORTS[name] = f"{_BASE}.oauth"
for name in ("DelegatedToKdcubeOperations", "operations_for_user"):
    _EXPORTS[name] = f"{_BASE}.operations"
for name in (
    "CONSENT_NEEDED_CODE",
    "PREFLIGHT_SCHEMA",
    "connected_account_consent_payload",
    "consent_action_message",
    "preflight_tool_claim_policies",
    "unavailable_tools_by_provider",
    "unavailable_tools_message",
):
    _EXPORTS[name] = f"{_BASE}.preflight"
for name in ("DelegatedToKdcubeStore", "account_id_for", "credential_id_for"):
    _EXPORTS[name] = f"{_BASE}.store"

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if not module_name:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
