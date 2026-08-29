# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter

"""Request-scoped public base URL for the named-services MCP surface.

Named-service providers run downstream of the MCP bridge through a generic
transport that carries no HTTP request. But a provider that needs to hand the
external client an absolute out-of-band URL (e.g. a binary file download link)
needs the public origin the client actually connected to. The bridge captures it
from the live request into this contextvar at the start of request handling, and
the provider's URL factory reads it back — same async context, no transport
change. Falls back to an empty string (provider then delivers inline) when unset.
"""

from __future__ import annotations

import contextvars

from prokura.mcp_metadata import (
    _request_public_base_url,
)

_PUBLIC_BASE_URL: contextvars.ContextVar[str] = contextvars.ContextVar(
    "kdcube_named_services_public_base_url", default=""
)


def set_public_base_url_from_request(request) -> None:
    try:
        base = _request_public_base_url(request)
    except Exception:
        base = ""
    _PUBLIC_BASE_URL.set(str(base or "").rstrip("/"))


def get_public_base_url() -> str:
    return _PUBLIC_BASE_URL.get()


__all__ = ["set_public_base_url_from_request", "get_public_base_url"]
