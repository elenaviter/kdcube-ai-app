# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# sdk/rate_limit/budget_limiter.py

"""
Application-level budget limiter with tenant/project isolation.

Tracks USD spending per provider (anthropic, openai, brave, etc.)
across ALL users of a bundle within a specific tenant/project.

Keys follow the pattern: {tenant}:{project}:kdcube:budget:{bundle}:{provider}:*
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from redis.asyncio import Redis

from kdcube_ai_app.apps.chat.sdk.infra.rate_limit.policy import ProviderBudgetPolicy, BudgetInsight

logger = logging.getLogger(__name__)


# --------- Time helpers ---------
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


# --------- Lua script for atomic spend commit ---------
_LUA_COMMIT_SPEND = r"""
local h_spend = KEYS[1]
local d_spend = KEYS[2]
local m_spend = KEYS[3]
local last_u  = KEYS[4]
local last_a  = KEYS[5]

local inc_usd  = tonumber(ARGV[1])
local exp_hour = tonumber(ARGV[2])
local exp_day  = tonumber(ARGV[3])
local exp_mon  = tonumber(ARGV[4])
local now_ts   = tonumber(ARGV[5])

if inc_usd > 0 then
  -- Increment as integer cents to avoid float precision issues
  local inc_cents = math.floor(inc_usd * 100 + 0.5)
  
  redis.call('INCRBY', h_spend, inc_cents); redis.call('EXPIREAT', h_spend, exp_hour)
  redis.call('INCRBY', d_spend, inc_cents); redis.call('EXPIREAT', d_spend, exp_day)
  redis.call('INCRBY', m_spend, inc_cents); redis.call('EXPIREAT', m_spend, exp_mon)
  
  redis.call('SET', last_u, tostring(inc_cents))
  redis.call('SET', last_a, tostring(now_ts))
end

return 1
"""


class BudgetLimiter:
    """
    Application-level budget tracking per provider with tenant/project isolation.

    Keys (with tenant/project prefix):
      {tenant}:{project}:kdcube:budget:{bundle}:{provider}:spend:hour:{YYYYMMDDHH}   (cents)
      {tenant}:{project}:kdcube:budget:{bundle}:{provider}:spend:day:{YYYYMMDD}      (cents)
      {tenant}:{project}:kdcube:budget:{bundle}:{provider}:spend:month:{YYYYMM}      (cents)
      {tenant}:{project}:kdcube:budget:{bundle}:{provider}:last_spend_usd            (cents)
      {tenant}:{project}:kdcube:budget:{bundle}:{provider}:last_spend_at             (timestamp)
    """

    def __init__(
            self,
            redis: Redis,
            *,
            tenant: str,
            project: str,
            namespace: str = "kdcube:budget"
    ):
        self.r = redis
        self.tenant = tenant
        self.project = project
        self.ns = namespace

    def _k(self, bundle: str, provider: str, *parts: str) -> str:
        """
        Build Redis key with tenant/project prefix.

        Format: {tenant}:{project}:kdcube:budget:{bundle}:{provider}:{parts}
        """
        base = f"{self.ns}:{bundle}:{provider}"
        if parts:
            base = f"{base}:{':'.join(parts)}"
        return f"{self.tenant}:{self.project}:{base}"

    async def check_budget(
            self,
            *,
            bundle_id: str,
            provider: str,
            policy: ProviderBudgetPolicy,
            now: Optional[datetime] = None,
    ) -> BudgetInsight:
        """
        Check if provider budget allows more spending.

        Args:
            bundle_id: Bundle identifier
            provider: Provider name (anthropic, openai, brave, etc.)
            policy: Budget policy for this provider
            now: Current time (for testing)

        Returns:
            BudgetInsight with allowed status and remaining budgets
        """
        now = (now or datetime.utcnow()).replace(tzinfo=timezone.utc)
        ymd, ym, ymdh = _ymd(now), _ym(now), _ymdh(now)

        # Build keys with tenant/project prefix
        k_spend_h = self._k(bundle_id, provider, "spend:hour", ymdh)
        k_spend_d = self._k(bundle_id, provider, "spend:day", ymd)
        k_spend_m = self._k(bundle_id, provider, "spend:month", ym)

        # Read current spend (stored as cents)
        vals = await self.r.mget(k_spend_h, k_spend_d, k_spend_m)

        # Convert cents to USD
        spend_h = int(vals[0] or 0) / 100.0
        spend_d = int(vals[1] or 0) / 100.0
        spend_m = int(vals[2] or 0) / 100.0

        # Check violations
        violations = []
        if policy.usd_per_hour is not None and spend_h >= policy.usd_per_hour:
            violations.append("usd_per_hour")
        if policy.usd_per_day is not None and spend_d >= policy.usd_per_day:
            violations.append("usd_per_day")
        if policy.usd_per_month is not None and spend_m >= policy.usd_per_month:
            violations.append("usd_per_month")

        # Calculate remaining
        def rem(limit: Optional[float], used: float) -> Optional[float]:
            if limit is None:
                return None
            return max(float(limit) - used, 0.0)

        remaining = {
            "usd_per_hour": rem(policy.usd_per_hour, spend_h),
            "usd_per_day": rem(policy.usd_per_day, spend_d),
            "usd_per_month": rem(policy.usd_per_month, spend_m),
        }

        # Calculate retry_after
        retry_after_sec, retry_scope = self._compute_retry_after(violations, now=now)

        return BudgetInsight(
            provider=provider,
            limits={
                "usd_per_hour": policy.usd_per_hour,
                "usd_per_day": policy.usd_per_day,
                "usd_per_month": policy.usd_per_month,
            },
            remaining=remaining,
            violations=violations,
            retry_after_sec=retry_after_sec,
            retry_scope=retry_scope,
        )

    async def commit(
            self,
            *,
            bundle_id: str,
            provider: str,
            spent_usd: float,
            now: Optional[datetime] = None,
    ) -> None:
        """
        Atomically record actual USD spend for a provider.

        Args:
            bundle_id: Bundle identifier
            provider: Provider name
            spent_usd: Actual USD spent (from calculate_turn_costs)
            now: Current time (for testing)
        """
        if spent_usd <= 0:
            return

        now = (now or datetime.utcnow()).replace(tzinfo=timezone.utc)
        ymd, ym, ymdh = _ymd(now), _ym(now), _ymdh(now)

        k_spend_h = self._k(bundle_id, provider, "spend:hour", ymdh)
        k_spend_d = self._k(bundle_id, provider, "spend:day", ymd)
        k_spend_m = self._k(bundle_id, provider, "spend:month", ym)
        k_last_u  = self._k(bundle_id, provider, "last_spend_usd")
        k_last_a  = self._k(bundle_id, provider, "last_spend_at")

        await self.r.eval(
            _LUA_COMMIT_SPEND,
            5,
            *_strs(k_spend_h, k_spend_d, k_spend_m, k_last_u, k_last_a),
            *_strs(
                float(spent_usd),
                _eoh(now),
                _eod(now),
                _eom(now),
                int(now.timestamp()),
            ),
        )

        logger.debug(
            f"Budget committed: tenant={self.tenant}, project={self.project}, "
            f"bundle={bundle_id}, provider={provider}, spent=${spent_usd:.6f}"
        )

    async def get_current_spend(
            self,
            *,
            bundle_id: str,
            provider: str,
            now: Optional[datetime] = None,
    ) -> dict[str, float]:
        """
        Get current spending for a provider (for monitoring/debugging).

        Returns:
            {"hour": 1.23, "day": 45.67, "month": 890.12}
        """
        now = (now or datetime.utcnow()).replace(tzinfo=timezone.utc)
        ymd, ym, ymdh = _ymd(now), _ym(now), _ymdh(now)

        k_spend_h = self._k(bundle_id, provider, "spend:hour", ymdh)
        k_spend_d = self._k(bundle_id, provider, "spend:day", ymd)
        k_spend_m = self._k(bundle_id, provider, "spend:month", ym)

        vals = await self.r.mget(k_spend_h, k_spend_d, k_spend_m)

        return {
            "hour": int(vals[0] or 0) / 100.0,
            "day": int(vals[1] or 0) / 100.0,
            "month": int(vals[2] or 0) / 100.0,
        }

    def _compute_retry_after(
            self,
            violations: list[str],
            *,
            now: Optional[datetime] = None,
    ) -> tuple[Optional[int], Optional[str]]:
        """
        Compute retry_after_sec and scope from violations.

        Returns: (retry_after_sec, scope) where scope in {"hour", "day", "month"} or None
        """
        if not violations:
            return None, None

        now = (now or datetime.utcnow()).replace(tzinfo=timezone.utc)
        now_ts = int(now.timestamp())
        candidates: list[tuple[str, int]] = []

        for v in violations:
            if v == "usd_per_hour":
                ttl = max(_eoh(now) - now_ts, 0)
                candidates.append(("hour", ttl))
            elif v == "usd_per_day":
                ttl = max(_eod(now) - now_ts, 0)
                candidates.append(("day", ttl))
            elif v == "usd_per_month":
                ttl = max(_eom(now) - now_ts, 0)
                candidates.append(("month", ttl))

        if not candidates:
            return None, None

        # Return longest wait
        scope, ttl = max(candidates, key=lambda it: it[1])
        return ttl, scope