# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Which connector app serves a provider - the GUARDED SERVICE decides.

A provider (slack, google, ...) may configure several connector apps (OAuth
client registrations) under ``connections.delegated_to_kdcube.providers.
<provider>.connector_apps``. Which one an auth scenario uses is never a user
pick: the service that is guarded declares it, one per provider type, in its
own named-services config block::

    named_services:
      connector_apps:
        slack: slack-demo
        google: gmail

The declaration is a request-scoped policy snapshot, so it rides the platform
portable context (``bundle_call_context``) - it survives async tasks,
subprocesses, and the detached agent/exec runtimes exactly like the rest of
the guaranteed context (see docs/runtime/cross-runtime-context-README.md). A
consumer that binds no declaration (the workspace agent's native gate) reads
empty, which means provider-wide: any connector app's account qualifies.
"""

from __future__ import annotations

from typing import Mapping

from kdcube_ai_app.apps.chat.sdk.runtime.comm_ctx import (
    get_current_bundle_call_context,
    update_current_bundle_call_context,
)

# The key under which the guarded service's declaration lives in
# bundle_call_context. Bundle-owned, JSON-safe, small - a request-scoped
# policy snapshot the portable spec carries across runtime boundaries.
_BCC_KEY = "connector_apps"


def set_service_connector_apps(mapping: Mapping[str, str] | None) -> None:
    """Bind the guarded service's provider -> connector-app declaration into
    the portable context for this execution (and any runtime it crosses). An
    empty/None mapping clears to provider-wide."""
    cleaned = {
        str(provider or "").strip(): str(app_id or "").strip()
        for provider, app_id in dict(mapping or {}).items()
        if str(provider or "").strip() and str(app_id or "").strip()
    }
    update_current_bundle_call_context({_BCC_KEY: cleaned})


def resolve_connector_app_id(provider_id: str) -> str:
    """The connector app to use for ``provider_id`` in the current request.

    One rule: the guarded service's declaration carried in the portable
    context, or empty. Empty means provider-wide - the broker then accepts any
    connector app's account."""
    provider = str(provider_id or "").strip()
    if not provider:
        return ""
    declaration = get_current_bundle_call_context().get(_BCC_KEY)
    if not isinstance(declaration, Mapping):
        return ""
    return str(declaration.get(provider) or "").strip()


__all__ = ["resolve_connector_app_id", "set_service_connector_apps"]
