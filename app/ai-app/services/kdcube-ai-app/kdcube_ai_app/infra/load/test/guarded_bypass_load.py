# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

"""
Guarded vs bypass throttling load test.

Goal:
- Guarded endpoint should produce 429 under burst load.
- Bypass endpoint should NOT produce 429 under the same load.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import hashlib
from urllib.parse import urlparse

import httpx


DEFAULT_BASE_URL = os.getenv("CHAT_BASE_URL", "http://localhost:8010")
DEFAULT_GUARDED_ENDPOINT = "/api/cb/resources/by-rn"
DEFAULT_BYPASS_ENDPOINT = "/api/admin/control-plane/webhooks/stripe"


@dataclass
class LoadUser:
    token: str
    role: str
    user_id: Optional[str] = None
    username: Optional[str] = None


def _load_idp_users(path: str) -> Dict[str, Dict]:
    with open(path, "r") as f:
        return json.load(f)


def _group_tokens(users: Dict[str, Dict]) -> Dict[str, List[LoadUser]]:
    groups: Dict[str, List[LoadUser]] = {"admin": [], "registered": [], "paid": []}
    for token, user in (users or {}).items():
        roles = set(user.get("roles") or [])
        entry = LoadUser(
            token=token,
            role="registered",
            user_id=user.get("sub") or user.get("user_id"),
            username=user.get("username"),
        )
        if "kdcube:role:super-admin" in roles:
            entry.role = "admin"
            groups["admin"].append(entry)
        elif "kdcube:role:paid" in roles:
            entry.role = "paid"
            groups["paid"].append(entry)
        else:
            groups["registered"].append(entry)
    return groups


def _select_user(groups: Dict[str, List[LoadUser]], role: str) -> Optional[LoadUser]:
    if role == "anonymous":
        return LoadUser(token="", role="anonymous")
    return (groups.get(role) or [None])[0]


def _load_tenant_project() -> Tuple[Optional[str], Optional[str]]:
    gateway_cfg = os.getenv("GATEWAY_CONFIG_JSON")
    if gateway_cfg:
        try:
            data = json.loads(gateway_cfg)
            return data.get("tenant") or data.get("tenant_id"), data.get("project") or data.get("project_id")
        except Exception:
            return None, None
    return os.getenv("TENANT_ID"), os.getenv("PROJECT_ID")


def _normalize_url(base_url: str, endpoint: str) -> str:
    if endpoint.startswith("http"):
        return endpoint
    base = base_url.rstrip("/")
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"
    return f"{base}{endpoint}"

def _path_from_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        return parsed.path or url
    except Exception:
        return url


def _default_guarded_payload(tenant: Optional[str], project: Optional[str], owner_id: str) -> Dict[str, str]:
    tenant = tenant or "tenant"
    project = project or "project"
    conv_id = f"guarded-{uuid.uuid4().hex[:6]}"
    turn_id = f"turn-{uuid.uuid4().hex[:6]}"
    message_id = "msg-1"
    rn = f"ef:{tenant}:{project}:chatbot:message:{owner_id}:{conv_id}:{turn_id}:assistant:{message_id}"
    return {"rn": rn}

async def _fetch_gateway_config(client: httpx.AsyncClient, base_url: str, admin_token: str) -> Optional[Dict]:
    if not admin_token:
        return None
    try:
        resp = await client.get(
            f"{base_url.rstrip('/')}/monitoring/system",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None

def _select_component_payload(payload: object, component: str) -> Optional[object]:
    if isinstance(payload, dict):
        if any(k in payload for k in ("ingress", "proc", "processor", "worker")):
            return payload.get(component)
    return payload

def _path_candidates(path: str) -> List[str]:
    if not path:
        return [path]
    clean = path if path.startswith("/") else f"/{path}"
    segments = [seg for seg in clean.split("/") if seg]
    if not segments:
        return ["/"]
    candidates: List[str] = [clean]
    for i in range(1, len(segments)):
        candidates.append("/" + "/".join(segments[i:]))
    seen = set()
    unique: List[str] = []
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique

def _patterns_match(path: str, patterns: List[str]) -> bool:
    import re
    candidates = _path_candidates(path)
    for pat in patterns or []:
        try:
            if any(re.match(pat, candidate) for candidate in candidates):
                return True
        except Exception:
            continue
    return False

def _get_role_limits(rate_limits: Dict[str, object], component: str, role: str) -> Tuple[Optional[int], Optional[int]]:
    comp_payload = _select_component_payload(rate_limits, component)
    roles = comp_payload.get("roles") if isinstance(comp_payload, dict) else comp_payload
    if not isinstance(roles, dict):
        return None, None
    role_cfg = roles.get(role)
    if not isinstance(role_cfg, dict):
        return None, None
    return role_cfg.get("hourly"), role_cfg.get("burst")


async def _send_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    user: LoadUser,
    headers_override: Optional[Dict[str, str]],
    payload: Optional[Dict[str, object]],
) -> Tuple[int, int, Optional[str]]:
    headers: Dict[str, str] = {}
    if user.role in {"registered", "admin", "paid"} and user.token:
        headers["Authorization"] = f"Bearer {user.token}"
    if user.role == "anonymous":
        fake_ip = f"192.168.1.{hash(url) % 254 + 1}"
        headers.update({"X-Forwarded-For": fake_ip, "User-Agent": "GuardedBypassTest"})
    if headers_override:
        headers.update(headers_override)

    t0 = time.monotonic()
    try:
        resp = await client.request(method.upper(), url, json=payload, headers=headers)
        ms = int((time.monotonic() - t0) * 1000)
        if resp.status_code >= 400:
            try:
                return resp.status_code, ms, resp.text
            except Exception:
                return resp.status_code, ms, None
        return resp.status_code, ms, None
    except Exception as e:
        ms = int((time.monotonic() - t0) * 1000)
        return 0, ms, str(e)


def _summarize(name: str, results: List[Tuple[int, int, Optional[str]]]) -> None:
    codes: Dict[int, int] = {}
    for code, _, _ in results:
        codes[code] = codes.get(code, 0) + 1
    total = len(results)
    rate_limited = codes.get(429, 0)
    print(f"\n{name}")
    print(f"  total={total} 429={rate_limited}")
    for code in sorted(codes.keys()):
        print(f"  {code}: {codes[code]}")


async def run_load(args: argparse.Namespace) -> int:
    tenant, project = _load_tenant_project()
    idp_path = args.idp_path or os.getenv("IDP_DB_PATH")
    groups: Dict[str, List[LoadUser]] = {}
    if idp_path and os.path.exists(idp_path):
        groups = _group_tokens(_load_idp_users(idp_path))

    if args.user_role == "anonymous":
        user = LoadUser(token="", role="anonymous")
    else:
        user = _select_user(groups, args.user_role)
        if not user:
            print("No user available for role. Check IDP_DB_PATH or choose --user-role anonymous.")
            return 1

    anon_headers = None
    anon_fingerprint = None
    if args.user_role == "anonymous":
        anon_ip = args.anon_ip
        anon_ua = args.anon_ua
        anon_headers = {
            "X-Forwarded-For": anon_ip,
            "User-Agent": anon_ua,
        }
        anon_fingerprint = hashlib.sha256(f"{anon_ip}:{anon_ua}".encode()).hexdigest()[:16]
        print(f"[info] anonymous fingerprint={anon_fingerprint}")

    owner_id = user.user_id or user.username or anon_fingerprint or "user"
    guarded_payload = json.loads(args.guarded_payload) if args.guarded_payload else _default_guarded_payload(tenant, project, owner_id)
    bypass_payload = json.loads(args.bypass_payload) if args.bypass_payload else {}
    guarded_headers = json.loads(args.guarded_headers) if args.guarded_headers else None
    bypass_headers = json.loads(args.bypass_headers) if args.bypass_headers else None
    if anon_headers:
        guarded_headers = {**anon_headers, **(guarded_headers or {})}
        bypass_headers = {**anon_headers, **(bypass_headers or {})}

    base_url = args.base_url.rstrip("/")
    guarded_url = _normalize_url(base_url, args.guarded_endpoint)
    bypass_url = _normalize_url(base_url, args.bypass_endpoint)

    timeout = httpx.Timeout(args.timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Try to validate config expectations (use any available token).
        admin_user = (groups.get("admin") or [None])[0] if groups else None
        reg_user = (groups.get("registered") or [None])[0] if groups else None
        paid_user = (groups.get("paid") or [None])[0] if groups else None
        monitor_token = (admin_user or reg_user or paid_user or None)
        monitor_token_val = monitor_token.token if monitor_token else ""
        gw_status = await _fetch_gateway_config(client, base_url, monitor_token_val)
        raw_cfg = gw_status.get("gateway_config_raw") if isinstance(gw_status, dict) else None
        if isinstance(raw_cfg, dict):
            guarded_payload_cfg = _select_component_payload(raw_cfg.get("guarded_rest_patterns"), "ingress")
            bypass_payload_cfg = _select_component_payload(raw_cfg.get("bypass_throttling_patterns"), "ingress")
            guarded_patterns = guarded_payload_cfg if isinstance(guarded_payload_cfg, list) else []
            bypass_patterns = bypass_payload_cfg if isinstance(bypass_payload_cfg, list) else []
            guarded_path = _path_from_url(guarded_url)
            bypass_path = _path_from_url(bypass_url)
            if guarded_patterns and not _patterns_match(guarded_path, guarded_patterns):
                print(f"[warn] guarded endpoint not matched by guarded_rest_patterns: {guarded_path}")
            if bypass_patterns and not _patterns_match(bypass_path, bypass_patterns):
                print(f"[warn] bypass endpoint not matched by bypass_throttling_patterns: {bypass_path}")
            if not bypass_patterns:
                print("[warn] bypass_throttling_patterns is empty for ingress; bypass test will 429.")
            hourly, burst = _get_role_limits(raw_cfg.get("rate_limits", {}), "ingress", args.user_role)
            if burst and args.requests <= int(burst):
                print(f"[warn] requests({args.requests}) <= burst({burst}); may not see 429 on guarded endpoint.")
        else:
            if monitor_token_val:
                print("[warn] could not fetch gateway config from /monitoring/system; skipping pattern checks.")
            else:
                print("[warn] no auth token available for /monitoring/system; skipping pattern checks.")

        sem = asyncio.Semaphore(max(1, args.concurrency))

        async def _burst(url: str, method: str, payload: Optional[Dict[str, object]], headers: Optional[Dict[str, str]]):
            tasks = []
            results: List[Tuple[int, int, Optional[str]]] = []

            async def _one():
                async with sem:
                    res = await _send_request(client, method, url, user, headers, payload)
                    results.append(res)

            for _ in range(args.requests):
                tasks.append(asyncio.create_task(_one()))
            await asyncio.gather(*tasks, return_exceptions=True)
            return results

        print("Guarded vs Bypass load test")
        print(f"  guarded={guarded_url} method={args.guarded_method}")
        print(f"  bypass={bypass_url} method={args.bypass_method}")
        print(f"  requests={args.requests} concurrency={args.concurrency} role={args.user_role}")

        guarded_results = await _burst(guarded_url, args.guarded_method, guarded_payload, guarded_headers)
        bypass_results = await _burst(bypass_url, args.bypass_method, bypass_payload, bypass_headers)

        _summarize("Guarded endpoint results", guarded_results)
        _summarize("Bypass endpoint results", bypass_results)

        if all(code == 404 for code, _, _ in guarded_results) and all(code != 429 for code, _, _ in guarded_results):
            print("[hint] Guarded endpoint returned only 404. Ensure it exists and is included in guarded_rest_patterns.")
        if any(code == 429 for code, _, _ in bypass_results):
            print("[hint] Bypass endpoint returned 429. Ensure it is included in bypass_throttling_patterns.")

    print("\nExpected: guarded has 429; bypass has 0 429 (other 4xx/5xx can still happen).")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Guarded vs bypass throttling load test")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--idp-path", default=os.getenv("IDP_DB_PATH"))
    p.add_argument("--user-role", default="anonymous", choices=["anonymous", "registered", "admin", "paid"])
    p.add_argument("--guarded-endpoint", default=DEFAULT_GUARDED_ENDPOINT)
    p.add_argument("--bypass-endpoint", default=DEFAULT_BYPASS_ENDPOINT)
    p.add_argument("--guarded-method", default="POST")
    p.add_argument("--bypass-method", default="POST")
    p.add_argument("--requests", type=int, default=50)
    p.add_argument("--concurrency", type=int, default=10)
    p.add_argument("--guarded-payload", default="")
    p.add_argument("--bypass-payload", default="")
    p.add_argument("--guarded-headers", default="")
    p.add_argument("--bypass-headers", default='{"Stripe-Signature":"t=0,v1=fake"}')
    p.add_argument("--anon-ip", default="198.51.100.42", help="Anonymous client IP for fingerprint")
    p.add_argument("--anon-ua", default="GuardedBypassTest/1.0", help="Anonymous user-agent for fingerprint")
    p.add_argument("--timeout", type=float, default=30.0)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return asyncio.run(run_load(args))


if __name__ == "__main__":
    raise SystemExit(main())
