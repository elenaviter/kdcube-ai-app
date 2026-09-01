# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
# Test cases descend from https://github.com/kdcube/kdcube/pull/1
# (Yaroslav Kytsia); adapted to the ported module.

import asyncio
import ipaddress

import pytest

from kdcube_ai_app.apps.chat.sdk.tools.backends.web import ssrf_guard
from kdcube_ai_app.apps.chat.sdk.tools.backends.web.ssrf_guard import (
    ReasonCode,
    _ip_violation,
    check_url,
    is_blocked_hostname,
)


@pytest.fixture(autouse=True)
def _fresh_guard(monkeypatch):
    monkeypatch.delenv("WEB_SSRF_GUARD", raising=False)
    ssrf_guard._verdict_cache.clear()
    yield
    ssrf_guard._verdict_cache.clear()


@pytest.mark.parametrize(
    "ip,expected",
    [
        ("127.0.0.1", ReasonCode.BLOCKED_LOCALHOST),
        ("::1", ReasonCode.BLOCKED_LOCALHOST),
        ("10.1.2.3", ReasonCode.BLOCKED_PRIVATE_IP),
        ("172.16.0.1", ReasonCode.BLOCKED_PRIVATE_IP),
        ("192.168.1.1", ReasonCode.BLOCKED_PRIVATE_IP),
        ("169.254.169.254", ReasonCode.BLOCKED_LINK_LOCAL),  # cloud metadata
        ("100.64.0.1", ReasonCode.BLOCKED_PRIVATE_IP),  # CGNAT
        ("0.0.0.0", ReasonCode.BLOCKED_PRIVATE_IP),
        ("224.0.0.1", ReasonCode.BLOCKED_PRIVATE_IP),  # multicast
        ("240.0.0.1", ReasonCode.BLOCKED_PRIVATE_IP),  # reserved
        ("fe80::1", ReasonCode.BLOCKED_LINK_LOCAL),
        ("fd00::1", ReasonCode.BLOCKED_PRIVATE_IP),  # unique local
        ("::", ReasonCode.BLOCKED_PRIVATE_IP),
        ("::ffff:127.0.0.1", ReasonCode.BLOCKED_LOCALHOST),  # v4-mapped
        ("::ffff:10.0.0.1", ReasonCode.BLOCKED_PRIVATE_IP),
    ],
)
def test_blocked_ip_ranges(ip, expected):
    assert _ip_violation(ipaddress.ip_address(ip)) == expected


@pytest.mark.parametrize("ip", ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"])
def test_public_ips_pass(ip):
    assert _ip_violation(ipaddress.ip_address(ip)) is None


def test_blocked_hostnames():
    assert is_blocked_hostname("localhost")
    assert is_blocked_hostname("LOCALHOST.")
    assert is_blocked_hostname("metadata.google.internal")
    assert is_blocked_hostname("foo.internal")
    assert is_blocked_hostname("printer.local")
    assert is_blocked_hostname("sub.localhost")
    assert not is_blocked_hostname("example.org")
    assert not is_blocked_hostname("internal.example.org")


def test_check_url_ip_literal_denied():
    verdict = asyncio.run(check_url("http://169.254.169.254/latest/meta-data/"))
    assert not verdict.allowed
    assert verdict.reason == ReasonCode.BLOCKED_LINK_LOCAL
    text = ssrf_guard.deny_text(verdict)
    assert "169.254.169.254" in text and "SSRF" in text


def test_check_url_resolved_private_denied(monkeypatch):
    async def _fake_getaddrinfo(host, *args, **kwargs):
        return [(None, None, None, None, ("10.0.0.5", 0))]

    class _Loop:
        getaddrinfo = staticmethod(_fake_getaddrinfo)

    monkeypatch.setattr(ssrf_guard.asyncio, "get_running_loop", lambda: _Loop())
    verdict = asyncio.run(check_url("https://rebinder.example/x"))
    assert not verdict.allowed
    assert verdict.reason == ReasonCode.BLOCKED_PRIVATE_IP
    assert verdict.details["blocked_ip"] == "10.0.0.5"


def test_check_url_resolved_public_allowed(monkeypatch):
    async def _fake_getaddrinfo(host, *args, **kwargs):
        return [(None, None, None, None, ("93.184.216.34", 0))]

    class _Loop:
        getaddrinfo = staticmethod(_fake_getaddrinfo)

    monkeypatch.setattr(ssrf_guard.asyncio, "get_running_loop", lambda: _Loop())
    verdict = asyncio.run(check_url("https://example.org/"))
    assert verdict.allowed and verdict.reason == ReasonCode.ALLOWED


def test_enabled_env_switch(monkeypatch):
    assert ssrf_guard.enabled()  # default on
    for off in ("off", "false", "0", "no"):
        monkeypatch.setenv("WEB_SSRF_GUARD", off)
        assert not ssrf_guard.enabled()
    monkeypatch.setenv("WEB_SSRF_GUARD", "True")
    assert ssrf_guard.enabled()


def test_guarded_connector_off_returns_none(monkeypatch):
    monkeypatch.setenv("WEB_SSRF_GUARD", "off")
    assert ssrf_guard.guarded_connector() is None


def test_guarded_resolver_rejects_private_answers(monkeypatch):
    aiohttp = pytest.importorskip("aiohttp")

    async def _run():
        connector = ssrf_guard.guarded_connector()
        assert connector is not None
        resolver = connector._resolver
        try:
            async def _fake_inner(host, port=0, family=0):
                return [{"host": "192.168.1.10", "port": port}]

            resolver._inner.resolve = _fake_inner
            with pytest.raises(OSError) as err:
                await resolver.resolve("rebinder.example", 443)
            assert "SSRF guard" in str(err.value)

            with pytest.raises(OSError):
                await resolver.resolve("metadata.google.internal", 80)
        finally:
            await connector.close()

    asyncio.run(_run())


def test_fetch_core_precheck_denies_before_any_connection():
    from kdcube_ai_app.apps.chat.sdk.tools.backends.web.fetch_backends import (
        _fetch_urls_core,
    )

    out = asyncio.run(
        _fetch_urls_core(
            urls=["http://127.0.0.1:8080/admin", "http://169.254.169.254/latest/"],
            max_content_length=-1,
            use_archive_fallback=False,
            extraction_mode="custom",
            max_concurrent=2,
            include_binary_base64=False,
            include_content_blocks=False,
        )
    )
    assert set(out) == {"http://127.0.0.1:8080/admin", "http://169.254.169.254/latest/"}
    for row in out.values():
        assert row["status"] == "denied_by_ssrf_guard"
        assert "SSRF" in row["error"]
