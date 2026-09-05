from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from kdcube_cli.management.errors import ManagementCliError


def _text(value: Any, *, maximum: int) -> str:
    candidate = str(value or "").strip()
    if not candidate or len(candidate) > maximum:
        raise ManagementCliError(
            "management_url_invalid",
            "The KDCube management URL is invalid.",
        )
    return candidate


def _is_loopback(hostname: str) -> bool:
    lowered = hostname.rstrip(".").lower()
    if lowered == "localhost" or lowered.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(lowered).is_loopback
    except ValueError:
        return False


def validate_web_url(
    value: Any,
    *,
    code: str,
    allow_query: bool = True,
    loopback_only: bool = False,
) -> str:
    """Validate an HTTPS URL, permitting HTTP only for loopback targets."""

    try:
        raw = _text(value, maximum=8192)
        parsed = urlsplit(raw)
        _ = parsed.port
    except (ManagementCliError, ValueError):
        raise ManagementCliError(
            code,
            "The KDCube management service published an invalid URL.",
        ) from None
    hostname = parsed.hostname or ""
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or (not allow_query and parsed.query)
    ):
        raise ManagementCliError(
            code,
            "The KDCube management service published an invalid URL.",
        )
    if parsed.scheme.lower() == "http" and not _is_loopback(hostname):
        raise ManagementCliError(
            code,
            "KDCube management requires HTTPS except on this device.",
        )
    if loopback_only and not _is_loopback(hostname):
        raise ManagementCliError(
            code,
            "The KDCube management callback must resolve to this device.",
        )
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc,
            parsed.path or "/",
            parsed.query if allow_query else "",
            "",
        )
    )


__all__ = ["validate_web_url"]
