# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# sdk/rate_limit/project_budget_limiter.py
from __future__ import annotations
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from redis.asyncio import Redis
import asyncpg

logger = logging.getLogger(__name__)

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

# Lua script for atomic spend commit
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
  local inc_cents = math.floor(inc_usd * 100 + 0.5)
  redis.call('INCRBY', h_spend, inc_cents); redis.call('EXPIREAT', h_spend, exp_hour)
  redis.call('INCRBY', d_spend, inc_cents); redis.call('EXPIREAT', d_spend, exp_day)
  redis.call('INCRBY', m_spend, inc_cents); redis.call('EXPIREAT', m_spend, exp_mon)
  redis.call('SET', last_u, tostring(inc_cents))
  redis.call('SET', last_a, tostring(now_ts))
end
return 1
"""

class ProjectBudgetLimiter:
    """
    Application-level budget with PostgreSQL balance tracking.

    Two-tier system:
    1. Redis: Per-bundle spending tracking (hour/day/month)
    2. PostgreSQL: Global app budget balance (deducted on commit)

     1. Application Budget BALANCE (money tracking)
│      - get_app_budget_balance()
│      - topup_app_budget()
│      - commit() → deducts from balance
│      - Table: tenant_project_budget
    Tracks actual MONEY available
    Table: tenant_project_budget
    Example: "We have $5,432.19 remaining"
    topup_app_budget(
        usd_amount=1000.0  # ADD MONEY
    )
    commit(
        spent_usd=12.50  # DEDUCT MONEY
    )

    │ 2. Spending Analytics (Redis)
    │  - get_spending_by_bundle()
    """

    CONTROL_PLANE_SCHEMA = "kdcube_control_plane"

    def __init__(self, redis: Redis, pg_pool: asyncpg.Pool, *, tenant: str, project: str, namespace: str = "kdcube:budget"):
        self.r = redis
        self.pg_pool = pg_pool
        self.tenant = tenant
        self.project = project
        self.ns = namespace

    def _k(self, bundle: str, provider: str, *parts: str) -> str:
        """Redis key: {tenant}:{project}:kdcube:budget:{bundle}:{provider}:{parts}"""
        base = f"{self.ns}:{bundle}:{provider}"
        if parts:
            base = f"{base}:{':'.join(parts)}"
        return f"{self.tenant}:{self.project}:{base}"

    async def commit(self, *, bundle_id: str, provider: str, spent_usd: float, now: Optional[datetime] = None) -> None:
        """
        Commit spending:
        1. Track in Redis (per-bundle breakdown)
        2. Deduct from PostgreSQL app budget (atomic)
        """
        if spent_usd <= 0:
            return

        now = (now or datetime.utcnow()).replace(tzinfo=timezone.utc)
        ymd, ym, ymdh = _ymd(now), _ym(now), _ymdh(now)

        # 1. Track in Redis
        k_spend_h = self._k(bundle_id, provider, "spend:hour", ymdh)
        k_spend_d = self._k(bundle_id, provider, "spend:day", ymd)
        k_spend_m = self._k(bundle_id, provider, "spend:month", ym)
        k_last_u  = self._k(bundle_id, provider, "last_spend_usd")
        k_last_a  = self._k(bundle_id, provider, "last_spend_at")

        await self.r.eval(
            _LUA_COMMIT_SPEND, 5,
            *_strs(k_spend_h, k_spend_d, k_spend_m, k_last_u, k_last_a),
            *_strs(float(spent_usd), _eoh(now), _eod(now), _eom(now), int(now.timestamp())),
        )

        # 2. Deduct from app budget (PostgreSQL, atomic)
        spent_cents = int(spent_usd * 100)
        async with self.pg_pool.acquire() as conn:
            result = await conn.fetchrow(f"""
                UPDATE {self.CONTROL_PLANE_SCHEMA}.tenant_project_budget
                SET 
                    balance_cents = balance_cents - $3,
                    lifetime_spent_cents = lifetime_spent_cents + $3,
                    updated_at = NOW()
                WHERE tenant = $1 AND project = $2
                RETURNING balance_cents
            """, self.tenant, self.project, spent_cents)

            if not result:
                # First spend - initialize budget row
                await conn.execute(f"""
                    INSERT INTO {self.CONTROL_PLANE_SCHEMA}.tenant_project_budget (tenant, project, balance_cents, lifetime_spent_cents)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (tenant, project) DO UPDATE SET
                        balance_cents = {self.CONTROL_PLANE_SCHEMA}.tenant_project_budget.balance_cents - $3,
                        lifetime_spent_cents = {self.CONTROL_PLANE_SCHEMA}.tenant_project_budget.lifetime_spent_cents + $4,
                        updated_at = NOW()
                """, self.tenant, self.project, -spent_cents, spent_cents)

        logger.debug(f"Budget committed: {self.tenant}/{self.project}/{bundle_id}/{provider}: ${spent_usd:.6f}")

    async def get_app_budget_balance(self) -> Dict[str, Any]:
        """Get current app budget balance from PostgreSQL."""
        async with self.pg_pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                SELECT balance_cents, lifetime_added_cents, lifetime_spent_cents
                FROM {self.CONTROL_PLANE_SCHEMA}.tenant_project_budget
                WHERE tenant = $1 AND project = $2
            """, self.tenant, self.project)

        if not row:
            return {"balance_usd": 0, "lifetime_added_usd": 0, "lifetime_spent_usd": 0}

        return {
            "balance_usd": row['balance_cents'] / 100.0,
            "lifetime_added_usd": row['lifetime_added_cents'] / 100.0,
            "lifetime_spent_usd": row['lifetime_spent_cents'] / 100.0,
        }

    async def topup_app_budget(self, *, usd_amount: float, notes: Optional[str] = None) -> Dict[str, float]:
        """Add funds to app budget (company owners, subscriptions)."""
        cents = int(usd_amount * 100)
        async with self.pg_pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                INSERT INTO {self.CONTROL_PLANE_SCHEMA}.tenant_project_budget (tenant, project, balance_cents, lifetime_added_cents, notes)
                VALUES ($1, $2, $3, $3, $4)
                ON CONFLICT (tenant, project) DO UPDATE SET
                    balance_cents = {self.CONTROL_PLANE_SCHEMA}.tenant_project_budget.balance_cents + $3,
                    lifetime_added_cents = {self.CONTROL_PLANE_SCHEMA}.tenant_project_budget.lifetime_added_cents + $3,
                    notes = $4,
                    updated_at = NOW()
                RETURNING balance_cents, lifetime_added_cents
            """, self.tenant, self.project, cents, notes)

        return {
            "new_balance_usd": row['balance_cents'] / 100.0,
            "lifetime_added_usd": row['lifetime_added_cents'] / 100.0,
        }

    async def get_spending_by_bundle(self, *, bundle_ids: Optional[list[str]] = None, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Aggregate spending across bundles from Redis."""
        now = (now or datetime.utcnow()).replace(tzinfo=timezone.utc)
        ymd, ym, ymdh = _ymd(now), _ym(now), _ymdh(now)

        if not bundle_ids:
            # Scan for all bundles
            pattern = f"{self.tenant}:{self.project}:{self.ns}:*:spend:month:{ym}"
            cursor = 0
            found_keys = []
            while True:
                cursor, keys = await self.r.scan(cursor, match=pattern, count=100)
                found_keys.extend(keys)
                if cursor == 0:
                    break

            # Extract unique bundle:provider pairs
            bundle_providers = set()
            for key in found_keys:
                parts = key.decode().split(":")
                # Format: {tenant}:{project}:kdcube:budget:{bundle}:{provider}:spend:month:{ym}
                if len(parts) >= 7:
                    bundle_id = parts[4]
                    provider = parts[5]
                    bundle_providers.add((bundle_id, provider))

            bundle_ids = list(set(bp[0] for bp in bundle_providers))

        # Aggregate per bundle
        bundles = {}
        total_hour = 0
        total_day = 0
        total_month = 0

        for bundle_id in bundle_ids:
            # Scan providers for this bundle
            pattern = f"{self.tenant}:{self.project}:{self.ns}:{bundle_id}:*:spend:month:{ym}"
            cursor = 0
            providers = set()
            while True:
                cursor, keys = await self.r.scan(cursor, match=pattern, count=100)
                for key in keys:
                    parts = key.decode().split(":")
                    if len(parts) >= 7:
                        providers.add(parts[5])
                if cursor == 0:
                    break

            bundle_total_h = 0
            bundle_total_d = 0
            bundle_total_m = 0

            for provider in providers:
                k_h = self._k(bundle_id, provider, "spend:hour", ymdh)
                k_d = self._k(bundle_id, provider, "spend:day", ymd)
                k_m = self._k(bundle_id, provider, "spend:month", ym)

                vals = await self.r.mget(k_h, k_d, k_m)
                h = int(vals[0] or 0) / 100.0
                d = int(vals[1] or 0) / 100.0
                m = int(vals[2] or 0) / 100.0

                bundle_total_h += h
                bundle_total_d += d
                bundle_total_m += m

            bundles[bundle_id] = {
                "hour": bundle_total_h,
                "day": bundle_total_d,
                "month": bundle_total_m,
            }

            total_hour += bundle_total_h
            total_day += bundle_total_d
            total_month += bundle_total_m

        return {
            "bundles": bundles,
            "totals": {
                "hour": total_hour,
                "day": total_day,
                "month": total_month,
            }
        }

    async def get_current_spend(self, *, bundle_id: str, provider: str, now: Optional[datetime] = None) -> dict[str, float]:
        """Get spending for specific bundle+provider."""
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