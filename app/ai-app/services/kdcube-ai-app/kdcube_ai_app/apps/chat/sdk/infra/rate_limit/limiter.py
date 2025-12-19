# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# sdk/rate_limit/limiter.py
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime, timedelta, timezone

from redis.asyncio import Redis

from kdcube_ai_app.apps.chat.sdk.infra.rate_limit.policy import QuotaPolicy


# --------- helpers (keys / time) ---------
def _k(ns: str, bundle: str, subject: str, *parts: str) -> str:
    """
    Build Redis key for user rate limiting.

    Format: {namespace}:{bundle}:{subject}:{parts}
    Example: kdcube:rl:kdcube.codegen.orchestrator:tenant-a:project-x:user123:locks
    """
    return ":".join([ns, bundle, subject, *parts])

def _ymd(dt: datetime) -> str:  return dt.strftime("%Y%m%d")
def _ym(dt: datetime) -> str:   return dt.strftime("%Y%m")
def _ymdh(dt: datetime) -> str: return dt.strftime("%Y%m%d%H")

def _eod(dt: datetime) -> int:
    end = datetime(dt.year, dt.month, dt.day, tzinfo=timezone.utc) + timedelta(days=1)
    return int(end.timestamp())

def _eom(dt: datetime) -> int:
    if dt.month == 12:
        nxt = datetime(dt.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        nxt = datetime(dt.year, dt.month + 1, 1, tzinfo=timezone.utc)
    return int(nxt.timestamp())

def _eoh(dt: datetime) -> int:
    end = datetime(dt.year, dt.month, dt.day, dt.hour, tzinfo=timezone.utc) + timedelta(hours=1)
    return int(end.timestamp())

def _strs(*items) -> list[str]:
    return [str(x) for x in items]

# --------- Lua scripts ---------
# ZSET lock with per-member expiry
# KEYS[1] = locks_zset
# ARGV = [now_ts, lock_id, max_concurrent, expire_ts]
_LUA_TRY_LOCK = r"""
local z = KEYS[1]
local now = tonumber(ARGV[1])
local lock_id = ARGV[2]
local maxc = tonumber(ARGV[3])
local exp  = tonumber(ARGV[4])

-- purge expired holders
redis.call('ZREMRANGEBYSCORE', z, '-inf', now)

local current = redis.call('ZCARD', z)
if current >= maxc then
  return {0, current, maxc}
end

redis.call('ZADD', z, exp, lock_id)
redis.call('EXPIREAT', z, exp)  -- clean up if idle
return {1, current + 1, maxc}
"""

# Atomic commit: +1 request, +tokens into hour/day/month, write last_turn_*, release lock
# KEYS: d_reqs, m_reqs, t_reqs, h_toks, d_toks, m_toks, last_tok, last_at, locks_zset
# ARGV: inc_req, inc_tokens, exp_day, exp_mon, exp_hour, now_ts, lock_id
_LUA_COMMIT = r"""
local d_reqs = KEYS[1]
local m_reqs = KEYS[2]
local t_reqs = KEYS[3]
local h_toks = KEYS[4]
local d_toks = KEYS[5]
local m_toks = KEYS[6]
local last_t = KEYS[7]
local last_a = KEYS[8]
local locks  = KEYS[9]

local inc_req  = tonumber(ARGV[1])
local inc_tok  = tonumber(ARGV[2])
local exp_day  = tonumber(ARGV[3])
local exp_mon  = tonumber(ARGV[4])
local exp_hour = tonumber(ARGV[5])
local now_ts   = tonumber(ARGV[6])
local lock_id  = ARGV[7]

if inc_req > 0 then
  redis.call('INCRBY', d_reqs, inc_req); redis.call('EXPIREAT', d_reqs, exp_day)
  redis.call('INCRBY', m_reqs, inc_req); redis.call('EXPIREAT', m_reqs, exp_mon)
  redis.call('INCRBY', t_reqs, inc_req)
end

if inc_tok > 0 then
  redis.call('INCRBY', h_toks, inc_tok); redis.call('EXPIREAT', h_toks, exp_hour)
  redis.call('INCRBY', d_toks, inc_tok); redis.call('EXPIREAT', d_toks, exp_day)
  redis.call('INCRBY', m_toks, inc_tok); redis.call('EXPIREAT', m_toks, exp_mon)
end

redis.call('SET', last_t, tostring(inc_tok))
redis.call('SET', last_a, tostring(now_ts))

if lock_id and lock_id ~= '' then
  redis.call('ZREM', locks, lock_id)
end
return 1
"""


# --------- Tier Balance Helpers ---------
def _merge_policy_with_tier_balance(
        base_policy: QuotaPolicy,
        tier_balance: Optional['UserTierBalance']
) -> QuotaPolicy:
    """
    Apply tier balance OVERRIDE (not additive).

    If tier_balance exists and has tier override: Use override limits where set, else fall back to base tier.

    Args:
        base_policy: Base tier policy
        tier_balance: User's tier balance (contains tier override + lifetime budget)

    Returns:
        Merged QuotaPolicy with overrides applied
    """
    if not tier_balance or not tier_balance.has_tier_override():
        return base_policy

    # Check if tier override is expired
    if tier_balance.is_expired():
        return base_policy

    # Apply OVERRIDE semantics (tier_balance fields override base_policy)
    return QuotaPolicy(
        max_concurrent=(
            tier_balance.max_concurrent
            if tier_balance.max_concurrent is not None
            else base_policy.max_concurrent
        ),
        requests_per_day=(
            tier_balance.requests_per_day
            if tier_balance.requests_per_day is not None
            else base_policy.requests_per_day
        ),
        requests_per_month=(
            tier_balance.requests_per_month
            if tier_balance.requests_per_month is not None
            else base_policy.requests_per_month
        ),
        total_requests=(
            tier_balance.total_requests
            if tier_balance.total_requests is not None
            else base_policy.total_requests
        ),
        tokens_per_hour=(
            tier_balance.tokens_per_hour
            if tier_balance.tokens_per_hour is not None
            else base_policy.tokens_per_hour
        ),
        tokens_per_day=(
            tier_balance.tokens_per_day
            if tier_balance.tokens_per_day is not None
            else base_policy.tokens_per_day
        ),
        tokens_per_month=(
            tier_balance.tokens_per_month
            if tier_balance.tokens_per_month is not None
            else base_policy.tokens_per_month
        ),
    )

# --------- API ---------
@dataclass
class AdmitResult:
    allowed: bool
    reason: Optional[str]
    lock_id: Optional[str]
    # snapshot after admission (remaining or current readings)
    snapshot: Dict[str, int]     # {req_day, req_month, req_total, tok_hour, tok_day, tok_month, in_flight}
    # tier balance info (for transparency)
    used_tier_override: bool = False
    effective_policy: Optional[Dict[str, Any]] = None  # Merged policy used for admission


class RateLimiter:
    """
    Redis-backed, atomic admission & accounting for user-level rate limiting.

    Supports tier balance - tier overrides and lifetime token budgets purchased or granted to users
    above their base policy limits.

    Tracks:
      - Concurrency via ZSET (+ per-holder expiry)
      - Request quotas: daily / monthly / total
      - Token budgets: hour / day / month (post-paid; checked at admit based on *previous* commits)

    Redis Keys (bundle-scoped with namespace prefix):
      kdcube:rl:{bundle}:{subject}:locks
      kdcube:rl:{bundle}:{subject}:reqs:day:{YYYYMMDD}
      kdcube:rl:{bundle}:{subject}:reqs:month:{YYYYMM}
      kdcube:rl:{bundle}:{subject}:reqs:total
      kdcube:rl:{bundle}:{subject}:toks:hour:{YYYYMMDDHH}
      kdcube:rl:{bundle}:{subject}:toks:day:{YYYYMMDD}
      kdcube:rl:{bundle}:{subject}:toks:month:{YYYYMM}
      kdcube:rl:{bundle}:{subject}:last_turn_tokens
      kdcube:rl:{bundle}:{subject}:last_turn_at

    Where:
      - bundle = Bundle ID (e.g., "kdcube.codegen.orchestrator")
      - subject = {tenant}:{project}:{user_id} or {tenant}:{project}:{user_id}:{session_id}

    Example:
      kdcube:rl:kdcube.codegen.orchestrator:tenant-a:project-x:user123:locks
      kdcube:rl:kdcube.codegen.orchestrator:tenant-a:project-x:user123:reqs:day:20250515
      kdcube:rl:kdcube.codegen.orchestrator:tenant-a:project-x:user123:toks:hour:2025051514
    """

    def __init__(
        self,
        redis: Redis,
        *,
        namespace: str = "kdcube:rl",
        tier_balance_manager: Optional['TierBalanceManager'] = None,
    ):
        """
        Initialize RateLimiter.

        Args:
            redis: Redis client
            namespace: Namespace prefix (default: "kdcube:rl")
            tier_balance_manager: Manager for querying user tier balances
        """
        self.r = redis
        self.ns = namespace
        self.tier_balance_manager = tier_balance_manager

    async def admit(
        self,
        *,
        bundle_id: str,
        subject_id: str,
        policy: QuotaPolicy,
        lock_id: str,
        lock_ttl_sec: int = 120,
        now: Optional[datetime] = None,
    ) -> AdmitResult:
        """
        Check request & token quotas (based on *already committed* usage),
        then (if allowed) acquire a concurrency slot.

        If tier_balance_manager is configured, fetches and applies tier overrides
        purchased or granted to the user.

        Args:
            bundle_id: Bundle identifier
            subject_id: Subject (format: {tenant}:{project}:{user_id})
            policy: Base QuotaPolicy with limits
            lock_id: Unique lock identifier (usually turn_id)
            lock_ttl_sec: Lock TTL in seconds (default: 120)
            now: Current time (for testing)

        Returns:
            AdmitResult with allowed status, snapshot, and tier balance info
        """
        now = (now or datetime.utcnow()).replace(tzinfo=timezone.utc)
        ymd, ym, ymdh = _ymd(now), _ym(now), _ymdh(now)

        # Parse subject_id to get tenant, project, user_id
        subject_parts = subject_id.split(":")
        tenant = subject_parts[0] if len(subject_parts) > 0 else None
        project = subject_parts[1] if len(subject_parts) > 1 else None
        user_id = subject_parts[2] if len(subject_parts) > 2 else None

        # Fetch tier balance and merge with base policy
        tier_balance = None
        used_tier_override = False
        effective_policy = policy

        if self.tier_balance_manager and tenant and project and user_id:
            try:
                tier_balance = await self.tier_balance_manager.get_user_tier_balance(
                    tenant=tenant,
                    project=project,
                    user_id=user_id,
                )

                if tier_balance and tier_balance.has_tier_override():
                    effective_policy = _merge_policy_with_tier_balance(policy, tier_balance)
                    used_tier_override = True
            except Exception as e:
                # Log but don't fail admission on tier balance errors
                import logging
                logging.warning(f"Failed to fetch tier balance for {subject_id}: {e}")

        # ---- Build keys using namespace prefix
        k_locks = _k(self.ns, bundle_id, subject_id, "locks")

        k_req_d = _k(self.ns, bundle_id, subject_id, "reqs:day", ymd)
        k_req_m = _k(self.ns, bundle_id, subject_id, "reqs:month", ym)
        k_req_t = _k(self.ns, bundle_id, subject_id, "reqs:total")

        k_tok_h = _k(self.ns, bundle_id, subject_id, "toks:hour", ymdh)
        k_tok_d = _k(self.ns, bundle_id, subject_id, "toks:day", ymd)
        k_tok_m = _k(self.ns, bundle_id, subject_id, "toks:month", ym)

        # ---- read current counters
        vals = await self.r.mget(k_req_d, k_req_m, k_req_t, k_tok_h, k_tok_d, k_tok_m)
        req_d = int(vals[0] or 0); req_m = int(vals[1] or 0); req_t = int(vals[2] or 0)
        tok_h = int(vals[3] or 0); tok_d = int(vals[4] or 0); tok_m = int(vals[5] or 0)

        # ---- policy checks using EFFECTIVE policy (base + tier override)
        violations = []
        if effective_policy.requests_per_day   is not None and req_d >= effective_policy.requests_per_day:   violations.append("requests_per_day")
        if effective_policy.requests_per_month is not None and req_m >= effective_policy.requests_per_month: violations.append("requests_per_month")
        if effective_policy.total_requests     is not None and req_t >= effective_policy.total_requests:     violations.append("total_requests")
        if effective_policy.tokens_per_hour    is not None and tok_h >= effective_policy.tokens_per_hour:    violations.append("tokens_per_hour")
        if effective_policy.tokens_per_day     is not None and tok_d >= effective_policy.tokens_per_day:     violations.append("tokens_per_day")
        if effective_policy.tokens_per_month   is not None and tok_m >= effective_policy.tokens_per_month:   violations.append("tokens_per_month")

        if violations:
            return AdmitResult(
                allowed=False,
                reason="|".join(violations),
                lock_id=None,
                snapshot={
                    "req_day": req_d, "req_month": req_m, "req_total": req_t,
                    "tok_hour": tok_h, "tok_day": tok_d, "tok_month": tok_m,
                    "in_flight": 0,
                },
                used_tier_override=used_tier_override,
                effective_policy=asdict(effective_policy) if used_tier_override else None,
            )

        # ---- concurrency lock (if configured)
        in_flight = 0
        if effective_policy.max_concurrent and effective_policy.max_concurrent > 0:
            res = await self.r.eval(
                _LUA_TRY_LOCK,
                1,
                *_strs(k_locks),
                *_strs(
                    int(now.timestamp()),                      # now (secs)
                    lock_id,                                   # member id
                    int(effective_policy.max_concurrent),      # max (using effective policy!)
                    int(now.timestamp()) + int(lock_ttl_sec),  # expire (secs)
                )
            )

            ok = bool(int(res[0]))
            in_flight = int(res[1]) if ok else int(res[1])  # res[1]=current after purge
            if not ok:
                return AdmitResult(
                    allowed=False,
                    reason="concurrency",
                    lock_id=None,
                    snapshot={
                        "req_day": req_d, "req_month": req_m, "req_total": req_t,
                        "tok_hour": tok_h, "tok_day": tok_d, "tok_month": tok_m,
                        "in_flight": in_flight,
                    },
                    used_tier_override=used_tier_override,
                    effective_policy=asdict(effective_policy) if used_tier_override else None,
                )

        return AdmitResult(
            allowed=True,
            reason=None,
            lock_id=lock_id,
            snapshot={
                "req_day": req_d, "req_month": req_m, "req_total": req_t,
                "tok_hour": tok_h, "tok_day": tok_d, "tok_month": tok_m,
                "in_flight": in_flight,
            },
            used_tier_override=used_tier_override,
            effective_policy=asdict(effective_policy) if used_tier_override else None,
        )

    async def commit(
        self,
        *,
        bundle_id: str,
        subject_id: str,
        tokens: int,
        lock_id: Optional[str],
        now: Optional[datetime] = None,
    ) -> None:
        """
        End-of-turn/accounting commit:
          - +1 request (day/month/total)
          - +tokens (hour/day/month)
          - last_turn_tokens / last_turn_at
          - release concurrency (if lock_id provided)

        Args:
            bundle_id: Bundle identifier
            subject_id: Subject (format: {tenant}:{project}:{user_id})
            tokens: Number of tokens to commit
            lock_id: Lock identifier to release
            now: Current time (for testing)
        """
        now = (now or datetime.utcnow()).replace(tzinfo=timezone.utc)
        ymd, ym, ymdh = _ymd(now), _ym(now), _ymdh(now)

        k_req_d = _k(self.ns, bundle_id, subject_id, "reqs:day", ymd)
        k_req_m = _k(self.ns, bundle_id, subject_id, "reqs:month", ym)
        k_req_t = _k(self.ns, bundle_id, subject_id, "reqs:total")

        k_tok_h = _k(self.ns, bundle_id, subject_id, "toks:hour", ymdh)
        k_tok_d = _k(self.ns, bundle_id, subject_id, "toks:day", ymd)
        k_tok_m = _k(self.ns, bundle_id, subject_id, "toks:month", ym)

        k_last_t = _k(self.ns, bundle_id, subject_id, "last_turn_tokens")
        k_last_a = _k(self.ns, bundle_id, subject_id, "last_turn_at")
        k_locks  = _k(self.ns, bundle_id, subject_id, "locks")

        await self.r.eval(
            _LUA_COMMIT,
            9,
            *_strs(
                k_req_d, k_req_m, k_req_t,
                k_tok_h, k_tok_d, k_tok_m,
                k_last_t, k_last_a, k_locks,
            ),
            *_strs(
                1,                      # +1 request
                int(tokens or 0),       # +tokens
                _eod(now),              # day EXPIREAT
                _eom(now),              # month EXPIREAT
                _eoh(now),              # hour EXPIREAT
                int(now.timestamp()),   # last_at
                lock_id or "",          # release this member
            ),
        )

    async def release(self, *, bundle_id: str, subject_id: str, lock_id: str) -> int:
        """
        Force-release a concurrency slot (use in error/abort paths).

        Args:
            bundle_id: Bundle identifier
            subject_id: Subject (format: {tenant}:{project}:{user_id})
            lock_id: Lock identifier to release

        Returns:
            Number of locks removed (0 or 1)
        """
        k_locks = _k(self.ns, bundle_id, subject_id, "locks")
        return int(await self.r.zrem(k_locks, lock_id))

    async def breakdown(
            self,
            *,
            tenant: str,
            project: str,
            user_id: str,
            bundle_ids: Optional[List[str]] = None,
            now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get usage breakdown for a user across bundles.

        Args:
            tenant: Tenant ID
            project: Project ID
            user_id: User ID
            bundle_ids: List of bundle IDs, or ["*"] for all bundles
            now: Current time (for testing)

        Returns:
            {
                "bundles": {
                    "bundle_id": {
                        "requests_today": int,
                        "requests_this_month": int,
                        "requests_total": int,
                        "tokens_today": int,
                        "tokens_this_month": int,
                        "concurrent": int,
                    }
                },
                "totals": {
                    "requests_today": int,
                    "requests_this_month": int,
                    "requests_total": int,
                    "tokens_today": int,
                    "tokens_this_month": int,
                }
            }
        """
        now = (now or datetime.utcnow()).replace(tzinfo=timezone.utc)
        ymd = _ymd(now)
        ym = _ym(now)

        subject_id = subject_id_of(tenant, project, user_id)

        # Find all bundles for this user if not specified
        if not bundle_ids or (len(bundle_ids) == 1 and bundle_ids[0] == "*"):
            # Scan Redis for all bundle keys
            pattern = f"{self.ns}:*:{subject_id}:reqs:total"
            cursor = 0
            found_bundles = set()

            while True:
                cursor, keys = await self.r.scan(cursor, match=pattern, count=100)
                for key in keys:
                    # Extract bundle_id from key pattern: kdcube:rl:{bundle}:{subject}:reqs:total
                    parts = key.decode().split(":")
                    if len(parts) >= 3:
                        bundle_id = parts[2]  # Index 2 is bundle_id
                        found_bundles.add(bundle_id)

                if cursor == 0:
                    break

            bundle_ids = list(found_bundles)

        if not bundle_ids:
            return {"bundles": {}, "totals": {
                "requests_today": 0,
                "requests_this_month": 0,
                "requests_total": 0,
                "tokens_today": 0,
                "tokens_this_month": 0,
            }}

        # Collect usage for each bundle
        bundles = {}
        totals = {
            "requests_today": 0,
            "requests_this_month": 0,
            "requests_total": 0,
            "tokens_today": 0,
            "tokens_this_month": 0,
        }

        for bundle_id in bundle_ids:
            k_req_d = _k(self.ns, bundle_id, subject_id, "reqs:day", ymd)
            k_req_m = _k(self.ns, bundle_id, subject_id, "reqs:month", ym)
            k_req_t = _k(self.ns, bundle_id, subject_id, "reqs:total")
            k_tok_d = _k(self.ns, bundle_id, subject_id, "toks:day", ymd)
            k_tok_m = _k(self.ns, bundle_id, subject_id, "toks:month", ym)
            k_locks = _k(self.ns, bundle_id, subject_id, "locks")

            vals = await self.r.mget(k_req_d, k_req_m, k_req_t, k_tok_d, k_tok_m)
            req_d = int(vals[0] or 0)
            req_m = int(vals[1] or 0)
            req_t = int(vals[2] or 0)
            tok_d = int(vals[3] or 0)
            tok_m = int(vals[4] or 0)
            concurrent = await self.r.zcard(k_locks)

            bundles[bundle_id] = {
                "requests_today": req_d,
                "requests_this_month": req_m,
                "requests_total": req_t,
                "tokens_today": tok_d,
                "tokens_this_month": tok_m,
                "concurrent": concurrent,
            }

            # Aggregate totals
            totals["requests_today"] += req_d
            totals["requests_this_month"] += req_m
            totals["requests_total"] += req_t
            totals["tokens_today"] += tok_d
            totals["tokens_this_month"] += tok_m

        return {"bundles": bundles, "totals": totals}

def subject_id_of(tenant: str, project: str, user_id: str, session_id: Optional[str] = None) -> str:
    """
    Build subject ID from tenant, project, and user.

    Format: {tenant}:{project}:{user_id} or {tenant}:{project}:{user_id}:{session_id}

    Args:
        tenant: Tenant ID
        project: Project ID
        user_id: User ID
        session_id: Optional session ID for session-level limiting

    Returns:
        Subject ID string

    Examples:
        >>> subject_id_of("tenant-a", "project-x", "user123")
        "tenant-a:project-x:user123"

        >>> subject_id_of("tenant-a", "project-x", "user123", "session456")
        "tenant-a:project-x:user123:session456"
    """
    return f"{tenant}:{project}:{user_id}" if not session_id else f"{tenant}:{project}:{user_id}:{session_id}"


@dataclass(frozen=True)
class QuotaInsight:
    """
    Infra-level view of quotas & current usage.

    Pure numbers / machine-friendly; no UI strings.
    """
    limits: Dict[str, Optional[int]]
    remaining: Dict[str, Optional[int]]
    violations: List[str]
    messages_remaining: Optional[int]
    retry_after_sec: Optional[int]
    retry_scope: Optional[str]   # "hour" | "day" | "month" | None
    used_tier_override: bool = False  # Whether tier override was applied


def _remaining_from_policy(policy: QuotaPolicy, snapshot: Dict[str, int]) -> Dict[str, Optional[int]]:
    """
    Compute remaining budget for each quota dimension from policy + snapshot.

    snapshot keys:
      req_day, req_month, req_total, tok_hour, tok_day, tok_month, in_flight
    """
    def rem(limit: Optional[int], used: int) -> Optional[int]:
        if limit is None:
            return None
        return max(limit - int(used or 0), 0)

    return {
        "requests_per_day": rem(policy.requests_per_day, snapshot.get("req_day", 0)),
        "requests_per_month": rem(policy.requests_per_month, snapshot.get("req_month", 0)),
        "total_requests": rem(policy.total_requests, snapshot.get("req_total", 0)),
        "tokens_per_hour": rem(policy.tokens_per_hour, snapshot.get("tok_hour", 0)),
        "tokens_per_day": rem(policy.tokens_per_day, snapshot.get("tok_day", 0)),
        "tokens_per_month": rem(policy.tokens_per_month, snapshot.get("tok_month", 0)),
    }


def _messages_remaining_from_remaining(remaining: Dict[str, Optional[int]]) -> Optional[int]:
    """
    Single "messages remaining" number.

    We take the minimum across all *request* quotas that are configured:
      - daily
      - monthly
      - total_requests (if used)

    That's the tightest bound on "how many more requests can I safely send?".
    """
    candidates = [
        remaining.get("requests_per_day"),
        remaining.get("requests_per_month"),
        remaining.get("total_requests"),
    ]
    candidates = [v for v in candidates if v is not None]
    if not candidates:
        return None
    return min(candidates)


def _retry_after_from_violations(violations: List[str], *, now: Optional[datetime] = None) -> Tuple[Optional[int], Optional[str]]:
    """
    Given violated quota names (matching the strings from RateLimiter.admit),
    compute TTL until the user is allowed again.

    If multiple windows are violated (e.g. tokens_per_hour + tokens_per_day),
    you must wait for *all* of them, so we take the MAX TTL.

    Returns: (retry_after_sec, scope) where scope ∈ {"hour","day","month"} or None.
    """
    if not violations:
        return None, None

    now = (now or datetime.utcnow()).replace(tzinfo=timezone.utc)
    now_ts = int(now.timestamp())
    candidates: List[Tuple[str, int]] = []

    for v in violations:
        if v in ("requests_per_day", "tokens_per_day"):
            ttl = max(_eod(now) - now_ts, 0)
            candidates.append(("day", ttl))
        elif v in ("requests_per_month", "tokens_per_month"):
            ttl = max(_eom(now) - now_ts, 0)
            candidates.append(("month", ttl))
        elif v == "tokens_per_hour":
            ttl = max(_eoh(now) - now_ts, 0)
            candidates.append(("hour", ttl))
        # total_requests has no reset; concurrency is not a quota window → ignore

    if not candidates:
        return None, None

    scope, ttl = max(candidates, key=lambda it: it[1])
    return ttl, scope


def compute_quota_insight(
        *,
        policy: QuotaPolicy,
        snapshot: Dict[str, int],
        reason: Optional[str],
        user_budget_tokens: Optional[int] = None,  # User's purchased token balance
        used_tier_override: bool = False,
        now: Optional[datetime] = None,
) -> QuotaInsight:
    """
    Compute quota insight considering BOTH tier limits AND user token budget.

    Early warning should trigger when user is close to running out of EITHER:
    - Request quotas (tier)
    - Token budget (tier + purchased)

    Args:
        policy: Effective QuotaPolicy (already merged with tier override if applicable)
        snapshot: Current usage snapshot
        reason: Violation reason if denied
        user_budget_tokens: User's purchased lifetime token balance
        used_tier_override: Whether tier override was applied
        now: Current time (for testing)

    Returns:
        QuotaInsight with limits, remaining, violations, and recommendations
    """
    limits = asdict(policy)
    remaining = _remaining_from_policy(policy, snapshot)
    violations: List[str] = (reason or "").split("|") if reason else []

    retry_after_sec, retry_scope = _retry_after_from_violations(violations, now=now)

    # Calculate messages_remaining from REQUEST quotas
    request_remaining = _messages_remaining_from_remaining(remaining)

    # Calculate messages_remaining from TOKEN BUDGET
    # Tier tokens remaining
    tier_token_remaining = remaining.get("tokens_per_month")

    # Total available tokens = tier + user budget
    total_token_remaining = tier_token_remaining
    if total_token_remaining is not None and user_budget_tokens is not None:
        total_token_remaining = total_token_remaining + user_budget_tokens
    elif total_token_remaining is None and user_budget_tokens is not None:
        total_token_remaining = user_budget_tokens

    # Estimate messages from token budget (assuming ~100K tokens per request)
    token_based_messages = None
    if total_token_remaining is not None:
        token_based_messages = max(total_token_remaining // 100_000, 0)

    # Final messages_remaining is the MINIMUM (most restrictive)
    messages_remaining = request_remaining
    if token_based_messages is not None:
        if messages_remaining is None:
            messages_remaining = token_based_messages
        else:
            messages_remaining = min(messages_remaining, token_based_messages)

    return QuotaInsight(
        limits=limits,
        remaining=remaining,
        violations=violations,
        messages_remaining=messages_remaining,
        retry_after_sec=retry_after_sec,
        retry_scope=retry_scope,
        used_tier_override=used_tier_override,
    )