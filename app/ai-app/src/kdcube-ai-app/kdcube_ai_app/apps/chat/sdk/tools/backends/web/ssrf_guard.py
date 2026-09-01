# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
# Ported from https://github.com/kdcube/kdcube/pull/1 by Yaroslav Kytsia
# (NavigationGuard core); reputation providers intentionally left out.

"""SSRF guard: the address-level layer of the web egress path.

The egress filter (``allowlist.py``) decides which NAMES the operator
permits. This module decides which ADDRESSES are ever connectable,
regardless of name: loopback, private networks, link-local (including
cloud metadata at 169.254.169.254), CGNAT, multicast, reserved ranges,
and metadata-style hostnames are refused before any connection.

Two enforcement points use it:

- a per-URL pre-check in the fetch core (catches IP-literal URLs and
  gives a clean ``denied_by_ssrf_guard`` result without burning a
  connection), and
- ``GuardedResolver``, installed on the fetcher's aiohttp connector,
  which validates every DNS answer at connect time — so redirect targets
  and DNS-rebinding answers are checked with the exact IPs the
  connection would use.

Default ON. Operators who genuinely need internal fetches (an intranet
deployment) disable it with ``filter.ssrf_guard: false`` in config.yaml
or ``WEB_SSRF_GUARD=off`` — and then own that trade.

Verdicts for hostnames are cached in-process for a short TTL only:
long-lived ALLOWED verdicts would reopen the rebinding window the
resolver exists to close.
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import urlparse

SSRF_GUARD_ENV = "WEB_SSRF_GUARD"

_VERDICT_TTL_SECONDS = 60.0


class ReasonCode(str, Enum):
    ALLOWED = "ALLOWED"
    BLOCKED_LOCALHOST = "BLOCKED_LOCALHOST"
    BLOCKED_PRIVATE_IP = "BLOCKED_PRIVATE_IP"
    BLOCKED_LINK_LOCAL = "BLOCKED_LINK_LOCAL"
    BLOCKED_HOSTNAME = "BLOCKED_HOSTNAME"
    RESOLUTION_FAILED = "RESOLUTION_FAILED"
    INVALID_URL = "INVALID_URL"


@dataclass
class Verdict:
    allowed: bool
    reason: ReasonCode
    details: Dict[str, Any] = field(default_factory=dict)


BLOCKED_HOSTNAMES = {"localhost", "metadata.google.internal"}
_BLOCKED_HOST_SUFFIXES = (".localhost", ".local", ".internal")


def enabled() -> bool:
    value = (os.environ.get(SSRF_GUARD_ENV) or "").strip().lower()
    return value not in ("off", "false", "0", "no", "disabled")


def normalize_hostname(hostname: str) -> str:
    if not hostname:
        return ""
    normalized = hostname.strip().lower().rstrip(".")
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    return normalized


def _ip_violation(ip: Union[ipaddress.IPv4Address, ipaddress.IPv6Address]) -> Optional[ReasonCode]:
    if ip.is_loopback:
        return ReasonCode.BLOCKED_LOCALHOST
    if ip.is_link_local:
        return ReasonCode.BLOCKED_LINK_LOCAL

    if isinstance(ip, ipaddress.IPv6Address):
        mapped = ip.ipv4_mapped
        if mapped is not None:
            return _ip_violation(mapped)
        if ip.is_private:
            return ReasonCode.BLOCKED_PRIVATE_IP
        if str(ip).startswith("fec0:"):  # deprecated site-local
            return ReasonCode.BLOCKED_PRIVATE_IP
        if ip == ipaddress.IPv6Address("::"):
            return ReasonCode.BLOCKED_PRIVATE_IP

    if isinstance(ip, ipaddress.IPv4Address):
        if str(ip).startswith("0."):
            return ReasonCode.BLOCKED_PRIVATE_IP
        if ip.is_private:
            return ReasonCode.BLOCKED_PRIVATE_IP
        if ip in ipaddress.IPv4Network("100.64.0.0/10"):  # CGNAT
            return ReasonCode.BLOCKED_PRIVATE_IP
        if ip.is_multicast or ip.is_reserved:
            return ReasonCode.BLOCKED_PRIVATE_IP

    return None


def is_blocked_hostname(hostname: str, extra_blocked: Optional[Iterable[str]] = None) -> bool:
    normalized = normalize_hostname(hostname)
    if not normalized:
        return False
    if normalized in BLOCKED_HOSTNAMES:
        return True
    if extra_blocked and normalized in {normalize_hostname(e) for e in extra_blocked}:
        return True
    return normalized.endswith(_BLOCKED_HOST_SUFFIXES)


_verdict_cache: Dict[str, Tuple[float, Verdict]] = {}


def _cache_get(host: str) -> Optional[Verdict]:
    entry = _verdict_cache.get(host)
    if entry and (time.monotonic() - entry[0]) < _VERDICT_TTL_SECONDS:
        return entry[1]
    return None


def _cache_put(host: str, verdict: Verdict) -> None:
    if len(_verdict_cache) > 2048:
        _verdict_cache.clear()
    _verdict_cache[host] = (time.monotonic(), verdict)


async def check_host(hostname: str) -> Verdict:
    """Verdict for one hostname or IP literal (no scheme, no path)."""
    host = normalize_hostname(hostname)
    if not host:
        return Verdict(False, ReasonCode.INVALID_URL, {"error": "empty hostname"})

    if is_blocked_hostname(host):
        return Verdict(False, ReasonCode.BLOCKED_HOSTNAME, {"hostname": host})

    try:
        ip_obj = ipaddress.ip_address(host)
    except ValueError:
        ip_obj = None
    if ip_obj is not None:
        violation = _ip_violation(ip_obj)
        if violation:
            return Verdict(False, violation, {"ip": str(ip_obj), "input_type": "ip_literal"})
        return Verdict(True, ReasonCode.ALLOWED, {"ip": str(ip_obj), "input_type": "ip_literal"})

    cached = _cache_get(host)
    if cached is not None:
        return cached

    try:
        loop = asyncio.get_running_loop()
        addr_info = await loop.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        resolved = {sockaddr[0] for *_ignored, sockaddr in addr_info}
    except Exception as e:
        return Verdict(False, ReasonCode.RESOLUTION_FAILED, {"hostname": host, "error": str(e)})
    if not resolved:
        return Verdict(False, ReasonCode.RESOLUTION_FAILED, {"hostname": host})

    for ip_str in resolved:
        try:
            violation = _ip_violation(ipaddress.ip_address(ip_str))
        except ValueError:
            continue
        if violation:
            verdict = Verdict(False, violation, {"hostname": host, "blocked_ip": ip_str})
            _cache_put(host, verdict)
            return verdict

    verdict = Verdict(True, ReasonCode.ALLOWED, {"hostname": host, "resolved_ips": sorted(resolved)})
    _cache_put(host, verdict)
    return verdict


async def check_url(url: str) -> Verdict:
    """Verdict for a full URL (the fetch core's per-URL pre-check)."""
    try:
        hostname = urlparse(url).hostname if "://" in url else url
    except Exception as e:
        return Verdict(False, ReasonCode.INVALID_URL, {"error": str(e)})
    if not hostname:
        return Verdict(False, ReasonCode.INVALID_URL, {"error": "no hostname", "url": url})
    return await check_host(hostname)


def deny_text(verdict: Verdict) -> str:
    """One line for in-band denial results."""
    what = verdict.details.get("blocked_ip") or verdict.details.get("ip") or verdict.details.get("hostname")
    return (
        f"refused by the SSRF guard ({verdict.reason.value}: {what}); private, "
        "loopback, link-local, and metadata addresses are not fetchable. The "
        "operator can disable the guard (filter.ssrf_guard: false) for "
        "internal-network deployments."
    )


def guarded_connector():
    """A TCPConnector whose resolver validates every DNS answer at connect
    time — redirect targets and rebinding answers included. Returns None
    when the guard is disabled (caller uses aiohttp defaults)."""
    if not enabled():
        return None

    import aiohttp
    from aiohttp.abc import AbstractResolver
    from aiohttp.resolver import DefaultResolver

    class GuardedResolver(AbstractResolver):
        def __init__(self) -> None:
            self._inner = DefaultResolver()

        async def resolve(self, host: str, port: int = 0, family: int = socket.AF_INET) -> List[Dict[str, Any]]:
            if is_blocked_hostname(host):
                raise OSError(f"SSRF guard: hostname '{host}' is blocked")
            infos = await self._inner.resolve(host, port, family)
            for info in infos:
                ip_str = info.get("host")
                try:
                    violation = _ip_violation(ipaddress.ip_address(ip_str))
                except (ValueError, TypeError):
                    continue
                if violation:
                    raise OSError(
                        f"SSRF guard: '{host}' resolves to {ip_str} ({violation.value})"
                    )
            return infos

        async def close(self) -> None:
            await self._inner.close()

    return aiohttp.TCPConnector(resolver=GuardedResolver())
