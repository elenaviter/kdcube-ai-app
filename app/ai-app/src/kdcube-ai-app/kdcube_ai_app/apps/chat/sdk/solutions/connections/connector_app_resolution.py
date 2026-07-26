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
        slack: demo
        google: gmail
      namespaces:
        ...

The named-services bridge binds that declaration per request; realm
integrations resolve through :func:`resolve_connector_app_id` with their
declared default as fallback. The platform layer below (broker, store, OAuth
start) is already multi-connector capable - an empty connector id means
provider-wide matching.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Mapping

# default=None (not {}): a mutable contextvar default is shared across contexts
# and does not survive a reconstructed/copied context (the detached agent
# runtime), where `.get()` then yields None. resolve reads defensively.
_SERVICE_CONNECTOR_APPS: ContextVar[dict[str, str] | None] = ContextVar(
    "kdcube_service_connector_apps", default=None
)


def set_service_connector_apps(mapping: Mapping[str, str] | None) -> None:
    """Bind the guarded service's provider -> connector-app declaration for
    this request context. An empty/None mapping clears to defaults."""
    cleaned = {
        str(provider or "").strip(): str(app_id or "").strip()
        for provider, app_id in dict(mapping or {}).items()
        if str(provider or "").strip() and str(app_id or "").strip()
    }
    _SERVICE_CONNECTOR_APPS.set(cleaned)


def resolve_connector_app_id(provider_id: str) -> str:
    """The connector app to use for ``provider_id`` in the current request.

    One rule: the guarded service's bound declaration, or empty. Empty means
    provider-wide - the broker then accepts any connector app's account."""
    provider = str(provider_id or "").strip()
    if not provider:
        return ""
    return (_SERVICE_CONNECTOR_APPS.get() or {}).get(provider, "")


__all__ = ["resolve_connector_app_id", "set_service_connector_apps"]
