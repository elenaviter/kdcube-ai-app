# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Lazy compatibility surface for delegated identity cards."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_BASE = "kdcube_ai_app.apps.chat.sdk.solutions.connections.delegated_credentials.cards"
_EXPORTS = {
    "CARD_STATE_ACTIVE": f"{_BASE}.model",
    "CARD_STATE_REVOKED": f"{_BASE}.model",
    "CardAuthority": f"{_BASE}.model",
    "CardCredentialHandles": f"{_BASE}.model",
    "CardCurrentPointer": f"{_BASE}.model",
    "CardRecordError": f"{_BASE}.model",
    "NamedServiceSelection": f"{_BASE}.model",
    "card_revision_name": f"{_BASE}.model",
    "CardCacheEntry": f"{_BASE}.cache",
    "DelegatedCardRuntimeCache": f"{_BASE}.cache",
    "CardUnavailable": f"{_BASE}.resolver",
    "DelegatedCardResolver": f"{_BASE}.resolver",
    "CardCommitFailed": f"{_BASE}.service",
    "CardConflict": f"{_BASE}.service",
    "DelegatedCardService": f"{_BASE}.service",
    "BundleStorageDelegatedCardStore": f"{_BASE}.store",
    "CardStorageError": f"{_BASE}.store",
    "DelegatedCardStore": f"{_BASE}.store",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if not module_name:
        raise AttributeError(name)
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
