# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# apps/chat/sdk/infra/economics/tier_balance.py

"""
Tier Balance Manager

Manages user tier balance:
1. Tier overrides (temporary tier upgrades via admin grants)
2. Lifetime budget (purchased tokens that deplete on use)

Table: user_tier_balance (formerly user_quota_replenishment)

CRITICAL: Uses COALESCE for partial updates - only updates fields you provide!
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Optional, List

import asyncpg
from redis.asyncio import Redis

from kdcube_ai_app.infra.namespaces import REDIS

logger = logging.getLogger(__name__)


@dataclass
class UserTierBalance:
    """
    User's tier balance - stores both tier override and lifetime budget.

    Tier Override: Temporary tier upgrade (expires)
      - Example: Grant 100 req/day for 7 days (trial)
      - Fields: requests_per_day, tokens_per_month, expires_at

    Lifetime Budget: Purchased tokens (depletes)
      - Example: Buy $10 → 666,667 tokens
      - Fields: lifetime_tokens_purchased, lifetime_tokens_consumed
    """
    # Identification
    tenant: str
    project: str
    user_id: str

    # Tier Override Limits (NULL = use base tier)
    max_concurrent: Optional[int] = None
    requests_per_day: Optional[int] = None
    requests_per_month: Optional[int] = None
    total_requests: Optional[int] = None
    tokens_per_hour: Optional[int] = None
    tokens_per_day: Optional[int] = None
    tokens_per_month: Optional[int] = None

    # Metadata
    expires_at: Optional[datetime] = None
    purchase_id: Optional[str] = None
    purchase_amount_usd: Optional[float] = None
    purchase_notes: Optional[str] = None

    # Lifetime tokens (only for purchases)
    lifetime_tokens_purchased: Optional[int] = None
    lifetime_tokens_consumed: Optional[int] = None

    active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def is_expired(self) -> bool:
        """Check if tier override has expired."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) >= self.expires_at

    def is_valid(self) -> bool:
        """Check if tier override is valid and active."""
        return self.active and not self.is_expired()

    def has_tier_override(self) -> bool:
        """Check if user has tier override (any quota limit set)."""
        return any([
            self.max_concurrent is not None,
            self.requests_per_day is not None,
            self.requests_per_month is not None,
            self.total_requests is not None,
            self.tokens_per_hour is not None,
            self.tokens_per_day is not None,
            self.tokens_per_month is not None,
        ])

    def has_lifetime_budget(self) -> bool:
        """Check if user has purchased lifetime budget."""
        return (
            self.purchase_amount_usd is not None
            and self.purchase_amount_usd > 0
            and self.lifetime_tokens_purchased is not None
            and self.lifetime_tokens_purchased > 0
        )

@dataclass
class TokenReservation:
    tenant: str
    project: str
    user_id: str
    reservation_id: str
    tokens_reserved: int
    status: str
    expires_at: datetime
    bundle_id: Optional[str] = None
    tokens_used: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    committed_at: Optional[datetime] = None
    released_at: Optional[datetime] = None
    notes: Optional[str] = None


class TierBalanceManager:
    """
    Manager for user tier balance.

    Operations:
    - update_user_tier_budget() - Set/update tier override (partial updates supported!)
    - add_lifetime_tokens() - Add purchased tokens to lifetime budget
    - deduct_lifetime_tokens() - Deduct tokens from lifetime budget on usage
    - get_user_tier_balance() - Get current tier balance (cached)
    """

    CONTROL_PLANE_SCHEMA = "kdcube_control_plane"
    TABLE_NAME = "user_tier_balance"
    RESERVATIONS_TABLE = "user_token_reservations"

    def __init__(
            self,
            pg_pool: Optional[asyncpg.Pool] = None,
            redis: Optional[Redis] = None,
            *,
            cache_ttl: int = 10,
            cache_namespace: str = REDIS.ECONOMICS.TIER_BALANCE_CACHE,
    ):
        self._pg_pool: Optional[asyncpg.Pool] = pg_pool
        self._redis: Optional[Redis] = redis
        self.cache_ttl = cache_ttl
        self.cache_ns = cache_namespace

        from kdcube_ai_app.apps.chat.sdk.config import get_settings
        self._settings = get_settings()

        self._owns_pool = pg_pool is None
        self._owns_redis = redis is None

    async def init(self, *, redis_url: Optional[str] = None):
        """Initialize connections if not provided."""
        if not self._pg_pool:
            self._pg_pool = await asyncpg.create_pool(
                host=self._settings.PGHOST,
                port=self._settings.PGPORT,
                user=self._settings.PGUSER,
                password=self._settings.PGPASSWORD,
                database=self._settings.PGDATABASE,
                ssl=self._settings.PGSSL,
            )
            self._owns_pool = True

        if not self._redis and redis_url:
            self._redis = Redis.from_url(redis_url)
            self._owns_redis = True

    async def close(self):
        """Close connections."""
        if self._owns_pool and self._pg_pool:
            await self._pg_pool.close()
        if self._owns_redis and self._redis:
            await self._redis.close()

    def _cache_key(self, tenant: str, project: str, user_id: str) -> str:
        """Build Redis cache key."""
        return f"{self.cache_ns}:{tenant}:{project}:{user_id}"

    async def get_user_tier_balance(
            self, *, tenant: str, project: str, user_id: str
    ) -> Optional[UserTierBalance]:
        """
        Get user's tier balance (Redis cache → PostgreSQL).

        Returns None if user has no tier override or lifetime budget.
        """
        # Check Redis cache
        if self._redis:
            cache_key = self._cache_key(tenant, project, user_id)
            try:
                cached = await self._redis.get(cache_key)
                if cached:
                    raw = cached.decode() if isinstance(cached, (bytes, bytearray)) else str(cached)
                    data = json.loads(raw)
                    if data:
                        for k in ['expires_at', 'created_at', 'updated_at']:
                            if data.get(k):
                                data[k] = datetime.fromisoformat(data[k])
                        balance = UserTierBalance(**data)
                        return balance if balance.is_valid() else None
            except Exception as e:
                logger.warning(f"Redis read error: {e}")

        # Query PostgreSQL
        if not self._pg_pool:
            return None

        async with self._pg_pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                SELECT * FROM {self.CONTROL_PLANE_SCHEMA}.{self.TABLE_NAME}
                WHERE tenant = $1 AND project = $2 AND user_id = $3 AND active = TRUE
                LIMIT 1
            """, tenant, project, user_id)

        if row:
            balance = UserTierBalance(**dict(row))
            # Cache in Redis
            if self._redis:
                try:
                    data = asdict(balance)
                    for k in ['expires_at', 'created_at', 'updated_at']:
                        if data.get(k):
                            data[k] = data[k].isoformat()
                    await self._redis.setex(
                        self._cache_key(tenant, project, user_id),
                        self.cache_ttl,
                        json.dumps(data)
                    )
                except Exception:
                    pass
            return balance if balance.is_valid() else None
        return None

    async def update_user_tier_budget(
            self,
            *,
            tenant: str,
            project: str,
            user_id: str,
            # Tier override fields (PARTIAL UPDATES SUPPORTED!)
            max_concurrent: Optional[int] = None,
            requests_per_day: Optional[int] = None,
            requests_per_month: Optional[int] = None,
            total_requests: Optional[int] = None,
            tokens_per_hour: Optional[int] = None,
            tokens_per_day: Optional[int] = None,
            tokens_per_month: Optional[int] = None,
            # Metadata
            expires_at: Optional[datetime] = None,
            purchase_id: Optional[str] = None,
            purchase_amount_usd: Optional[float] = None,
            purchase_notes: Optional[str] = None,
    ) -> UserTierBalance:
        """
        Update user's tier budget (tier override).

        IMPORTANT: This supports PARTIAL UPDATES!

        Example 1 - Create new tier override:
            update_user_tier_budget(
                user_id="john",
                requests_per_day=100,
                tokens_per_month=300_000_000,
                expires_at=datetime(..., days=7)
            )
            # Creates: requests_per_day=100, tokens_per_month=300M

        Example 2 - Partial update (only change requests):
            update_user_tier_budget(
                user_id="john",
                requests_per_day=200  # Only update this!
                # tokens_per_month not provided → stays 300M
            )
            # Updates: requests_per_day=200, tokens_per_month=300M (unchanged!)

        Example 3 - Extend expiry:
            update_user_tier_budget(
                user_id="john",
                expires_at=datetime(..., days=14)
                # All quotas stay the same, only expiry changes
            )
        """
        async with self._pg_pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                INSERT INTO {self.CONTROL_PLANE_SCHEMA}.{self.TABLE_NAME} (
                    tenant, project, user_id,
                    max_concurrent, requests_per_day, requests_per_month, total_requests,
                    tokens_per_hour, tokens_per_day, tokens_per_month,
                    expires_at, purchase_id, purchase_amount_usd, purchase_notes
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                ON CONFLICT (tenant, project, user_id)
                DO UPDATE SET
                    -- PARTIAL UPDATE: Only update fields that are provided (not NULL)
                    -- COALESCE keeps existing value if new value is NULL
                    max_concurrent = COALESCE(
                        EXCLUDED.max_concurrent, 
                        {self.CONTROL_PLANE_SCHEMA}.{self.TABLE_NAME}.max_concurrent
                    ),
                    requests_per_day = COALESCE(
                        EXCLUDED.requests_per_day, 
                        {self.CONTROL_PLANE_SCHEMA}.{self.TABLE_NAME}.requests_per_day
                    ),
                    requests_per_month = COALESCE(
                        EXCLUDED.requests_per_month, 
                        {self.CONTROL_PLANE_SCHEMA}.{self.TABLE_NAME}.requests_per_month
                    ),
                    total_requests = COALESCE(
                        EXCLUDED.total_requests, 
                        {self.CONTROL_PLANE_SCHEMA}.{self.TABLE_NAME}.total_requests
                    ),
                    tokens_per_hour = COALESCE(
                        EXCLUDED.tokens_per_hour, 
                        {self.CONTROL_PLANE_SCHEMA}.{self.TABLE_NAME}.tokens_per_hour
                    ),
                    tokens_per_day = COALESCE(
                        EXCLUDED.tokens_per_day, 
                        {self.CONTROL_PLANE_SCHEMA}.{self.TABLE_NAME}.tokens_per_day
                    ),
                    tokens_per_month = COALESCE(
                        EXCLUDED.tokens_per_month, 
                        {self.CONTROL_PLANE_SCHEMA}.{self.TABLE_NAME}.tokens_per_month
                    ),
                    expires_at = COALESCE(
                        EXCLUDED.expires_at, 
                        {self.CONTROL_PLANE_SCHEMA}.{self.TABLE_NAME}.expires_at
                    ),
                    purchase_id = COALESCE(
                        EXCLUDED.purchase_id, 
                        {self.CONTROL_PLANE_SCHEMA}.{self.TABLE_NAME}.purchase_id
                    ),
                    purchase_amount_usd = COALESCE(
                        EXCLUDED.purchase_amount_usd, 
                        {self.CONTROL_PLANE_SCHEMA}.{self.TABLE_NAME}.purchase_amount_usd
                    ),
                    purchase_notes = COALESCE(
                        EXCLUDED.purchase_notes, 
                        {self.CONTROL_PLANE_SCHEMA}.{self.TABLE_NAME}.purchase_notes
                    ),
                    updated_at = NOW()
                RETURNING *
            """, tenant, project, user_id,
                                      max_concurrent, requests_per_day, requests_per_month, total_requests,
                                      tokens_per_hour, tokens_per_day, tokens_per_month,
                                      expires_at, purchase_id, purchase_amount_usd, purchase_notes)

        await self._invalidate_cache(tenant, project, user_id)
        return UserTierBalance(**dict(row))

    async def _invalidate_cache(self, tenant: str, project: str, user_id: str):
        """Invalidate Redis cache."""
        if self._redis:
            try:
                await self._redis.delete(self._cache_key(tenant, project, user_id))
            except Exception:
                pass

    async def deactivate_tier_balance(
            self, *, tenant: str, project: str, user_id: str
    ):
        """
        Deactivate user's tier balance (soft delete).
        Clears BOTH tier override and lifetime budget.
        """
        if not self._pg_pool:
            raise RuntimeError("PostgreSQL pool not initialized")

        async with self._pg_pool.acquire() as conn:
            await conn.execute(f"""
                UPDATE {self.CONTROL_PLANE_SCHEMA}.{self.TABLE_NAME}
                SET active = FALSE, updated_at = NOW()
                WHERE tenant = $1 AND project = $2 AND user_id = $3
            """, tenant, project, user_id)

        await self._invalidate_cache(tenant, project, user_id)

    async def list_user_tier_balances(
            self, *, tenant: str, project: str, user_id: str, include_expired: bool = False
    ) -> List[UserTierBalance]:
        """List all tier balances for a user (should be 0 or 1)."""
        if not self._pg_pool:
            return []

        expired_filter = "" if include_expired else "AND (expires_at IS NULL OR expires_at > NOW())"

        async with self._pg_pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT *
                FROM {self.CONTROL_PLANE_SCHEMA}.{self.TABLE_NAME}
                WHERE tenant = $1 AND project = $2 AND user_id = $3 AND active = TRUE
                {expired_filter}
                ORDER BY created_at DESC
            """, tenant, project, user_id)

        return [UserTierBalance(**dict(row)) for row in rows]

    # =========================================================================
    # Lifetime Token Operations
    # =========================================================================

    async def add_lifetime_tokens(
            self,
            *,
            tenant: str,
            project: str,
            user_id: str,
            tokens: int,
            usd_amount: float,
            purchase_id: Optional[str] = None,
            notes: Optional[str] = None,
            conn: Optional[asyncpg.Connection] = None,
    ) -> UserTierBalance:
        if not self._pg_pool and not conn:
            raise RuntimeError("PostgreSQL pool not initialized")
        if tokens <= 0:
            # no-op, but still return current row if exists
            bal = await self.get_user_tier_balance(tenant=tenant, project=project, user_id=user_id)
            return bal  # type: ignore

        sql = f"""
          INSERT INTO {self.CONTROL_PLANE_SCHEMA}.{self.TABLE_NAME} (
            tenant, project, user_id,
            lifetime_tokens_purchased,
            lifetime_tokens_consumed,
            expires_at,
            purchase_id,
            purchase_amount_usd,
            purchase_notes
          ) VALUES ($1,$2,$3,$4,0,NULL,$5,$6,$7)
          ON CONFLICT (tenant, project, user_id)
          DO UPDATE SET
            lifetime_tokens_purchased = COALESCE({self.CONTROL_PLANE_SCHEMA}.{self.TABLE_NAME}.lifetime_tokens_purchased, 0)
                                     + EXCLUDED.lifetime_tokens_purchased,
            purchase_amount_usd = COALESCE({self.CONTROL_PLANE_SCHEMA}.{self.TABLE_NAME}.purchase_amount_usd, 0)
                                + EXCLUDED.purchase_amount_usd,
            purchase_id = EXCLUDED.purchase_id,
            purchase_notes = EXCLUDED.purchase_notes,
            updated_at = NOW()
          RETURNING *
        """

        if conn:
            row = await conn.fetchrow(sql, tenant, project, user_id, int(tokens), purchase_id, float(usd_amount), notes)
        else:
            async with self._pg_pool.acquire() as c:
                row = await c.fetchrow(sql, tenant, project, user_id, int(tokens), purchase_id, float(usd_amount), notes)

        await self._invalidate_cache(tenant, project, user_id)
        return UserTierBalance(**dict(row))

    async def deduct_lifetime_tokens(
            self, *, tenant: str, project: str, user_id: str, tokens: int
    ) -> int:
        """
        Deduct tokens from lifetime budget.

        Returns overflow tokens (tokens that couldn't be deducted).
        """
        if not self._pg_pool or tokens <= 0:
            return tokens

        async with self._pg_pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                UPDATE {self.CONTROL_PLANE_SCHEMA}.{self.TABLE_NAME}
                SET 
                    lifetime_tokens_consumed = LEAST(
                        COALESCE(lifetime_tokens_consumed, 0) + $4,
                        COALESCE(lifetime_tokens_purchased, 0)
                    ),
                    updated_at = NOW()
                WHERE tenant = $1 AND project = $2 AND user_id = $3
                  AND active = TRUE
                  AND purchase_amount_usd IS NOT NULL
                  AND purchase_amount_usd > 0
                  AND (expires_at IS NULL OR expires_at > NOW())
                RETURNING 
                    COALESCE(lifetime_tokens_purchased, 0) AS purchased,
                    COALESCE(lifetime_tokens_consumed, 0) AS consumed
            """, tenant, project, user_id, tokens)

        if not row:
            return tokens  # No budget, full overflow

        old_consumed = row['consumed'] - tokens
        actually_deducted = min(tokens, row['purchased'] - old_consumed)
        overflow = tokens - actually_deducted

        await self._invalidate_cache(tenant, project, user_id)
        return max(overflow, 0)

    async def get_lifetime_balance(
            self, *, tenant: str, project: str, user_id: str
    ) -> Optional[int]:
        """
        Get remaining AVAILABLE lifetime tokens:
        purchased - consumed - active_reservations
        """
        if not self._pg_pool:
            return None

        async with self._pg_pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                SELECT
                    COALESCE(utb.lifetime_tokens_purchased, 0) AS purchased,
                    COALESCE(utb.lifetime_tokens_consumed, 0) AS consumed,
                    COALESCE(rsv.reserved, 0) AS reserved
                FROM {self.CONTROL_PLANE_SCHEMA}.{self.TABLE_NAME} utb
                LEFT JOIN (
                    SELECT tenant, project, user_id, COALESCE(SUM(tokens_reserved), 0) AS reserved
                    FROM {self.CONTROL_PLANE_SCHEMA}.{self.RESERVATIONS_TABLE}
                    WHERE tenant = $1 AND project = $2 AND user_id = $3
                      AND status = 'reserved'
                      AND expires_at > NOW()
                    GROUP BY tenant, project, user_id
                ) rsv
                ON rsv.tenant = utb.tenant AND rsv.project = utb.project AND rsv.user_id = utb.user_id
                WHERE utb.tenant = $1 AND utb.project = $2 AND utb.user_id = $3
                  AND utb.active = TRUE
                  AND utb.purchase_amount_usd IS NOT NULL
                  AND utb.purchase_amount_usd > 0
                  AND (utb.expires_at IS NULL OR utb.expires_at > NOW())
            """, tenant, project, user_id)

        if not row:
            return None

        remaining = int(row["purchased"]) - int(row["consumed"]) - int(row["reserved"])
        return max(remaining, 0)

    # =========================================================================
    # Lifetime Token Reservations (Personal Credits)
    # =========================================================================

    async def _reserved_sum(
            self, *, conn: asyncpg.Connection, tenant: str, project: str, user_id: str, exclude_reservation_id: Optional[str] = None
    ) -> int:
        exclude_sql = ""
        args = [tenant, project, user_id]
        if exclude_reservation_id:
            args.append(exclude_reservation_id)
            exclude_sql = f"AND reservation_id <> ${len(args)}"

        row = await conn.fetchval(f"""
            SELECT COALESCE(SUM(tokens_reserved), 0)
            FROM {self.CONTROL_PLANE_SCHEMA}.{self.RESERVATIONS_TABLE}
            WHERE tenant = $1 AND project = $2 AND user_id = $3
              AND status = 'reserved'
              AND expires_at > NOW()
              {exclude_sql}
        """, *args)
        return int(row or 0)

    async def reserve_lifetime_tokens(
            self,
            *,
            tenant: str,
            project: str,
            user_id: str,
            reservation_id: str,
            tokens: int,
            ttl_sec: int = 900,
            bundle_id: Optional[str] = None,
            notes: Optional[str] = None,
    ) -> bool:
        """
        Atomically reserve 'tokens' from user's lifetime budget so concurrent in-flight requests
        cannot oversubscribe credits.

        Returns True if reserved, False if insufficient available tokens.
        """
        if not self._pg_pool:
            raise RuntimeError("PostgreSQL pool not initialized")
        if tokens <= 0:
            return True

        exp = datetime.now(timezone.utc) + timedelta(seconds=int(ttl_sec))

        async with self._pg_pool.acquire() as conn:
            async with conn.transaction():
                bal = await conn.fetchrow(f"""
                    SELECT
                        COALESCE(lifetime_tokens_purchased, 0) AS purchased,
                        COALESCE(lifetime_tokens_consumed, 0) AS consumed
                    FROM {self.CONTROL_PLANE_SCHEMA}.{self.TABLE_NAME}
                    WHERE tenant = $1 AND project = $2 AND user_id = $3
                      AND active = TRUE
                      AND purchase_amount_usd IS NOT NULL
                      AND purchase_amount_usd > 0
                      AND (expires_at IS NULL OR expires_at > NOW())
                    FOR UPDATE
                """, tenant, project, user_id)

                if not bal:
                    return False

                purchased = int(bal["purchased"] or 0)
                consumed = int(bal["consumed"] or 0)
                if purchased <= 0:
                    return False

                reserved = await self._reserved_sum(conn=conn, tenant=tenant, project=project, user_id=user_id)
                available = purchased - consumed - reserved
                if available < tokens:
                    return False

                # Idempotent per reservation_id; never shrink an existing reservation (GREATEST)
                await conn.execute(f"""
                    INSERT INTO {self.CONTROL_PLANE_SCHEMA}.{self.RESERVATIONS_TABLE} (
                        tenant, project, user_id,
                        reservation_id, bundle_id,
                        tokens_reserved, status, expires_at, notes
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, 'reserved', $7, $8)
                    ON CONFLICT (tenant, project, user_id, reservation_id)
                    DO UPDATE SET
                        tokens_reserved = GREATEST(
                            EXCLUDED.tokens_reserved,
                            {self.CONTROL_PLANE_SCHEMA}.{self.RESERVATIONS_TABLE}.tokens_reserved
                        ),
                        status = 'reserved',
                        expires_at = EXCLUDED.expires_at,
                        bundle_id = EXCLUDED.bundle_id,
                        notes = EXCLUDED.notes,
                        updated_at = NOW()
                """, tenant, project, user_id, reservation_id, bundle_id, int(tokens), exp, notes)

        # No need to invalidate tier-balance cache (purchased/consumed unchanged),
        # but lifetime "available" reporting uses DB anyway.
        return True

    async def release_lifetime_token_reservation(
            self,
            *,
            tenant: str,
            project: str,
            user_id: str,
            reservation_id: str,
            reason: Optional[str] = None,
    ) -> None:
        """Release an in-flight reservation (abort/error path)."""
        if not self._pg_pool:
            return

        async with self._pg_pool.acquire() as conn:
            await conn.execute(f"""
                UPDATE {self.CONTROL_PLANE_SCHEMA}.{self.RESERVATIONS_TABLE}
                SET status = 'released',
                    released_at = NOW(),
                    notes = COALESCE(notes, '') || CASE WHEN $5 IS NULL THEN '' ELSE (' | ' || $5) END,
                    updated_at = NOW()
                WHERE tenant = $1 AND project = $2 AND user_id = $3 AND reservation_id = $4
                  AND status = 'reserved'
            """, tenant, project, user_id, reservation_id, reason)

    async def commit_reserved_lifetime_tokens(
            self,
            *,
            tenant: str,
            project: str,
            user_id: str,
            reservation_id: str,
            tokens: int,
    ) -> int:
        """
        Commit actual token spend against a reservation.

        Returns overflow tokens (portion that could NOT be covered by personal credits),
        respecting other active reservations.
        """
        if not self._pg_pool or tokens <= 0:
            return tokens

        async with self._pg_pool.acquire() as conn:
            async with conn.transaction():
                bal = await conn.fetchrow(f"""
                    SELECT
                        COALESCE(lifetime_tokens_purchased, 0) AS purchased,
                        COALESCE(lifetime_tokens_consumed, 0) AS consumed
                    FROM {self.CONTROL_PLANE_SCHEMA}.{self.TABLE_NAME}
                    WHERE tenant = $1 AND project = $2 AND user_id = $3
                      AND active = TRUE
                      AND purchase_amount_usd IS NOT NULL
                      AND purchase_amount_usd > 0
                      AND (expires_at IS NULL OR expires_at > NOW())
                    FOR UPDATE
                """, tenant, project, user_id)

                if not bal:
                    # No personal credits -> full overflow
                    await self.release_lifetime_token_reservation(
                        tenant=tenant, project=project, user_id=user_id,
                        reservation_id=reservation_id, reason="commit: no_balance_row"
                    )
                    return tokens

                purchased = int(bal["purchased"] or 0)
                consumed = int(bal["consumed"] or 0)

                # Exclude THIS reservation from the "reserved sum" so it becomes available for commit.
                other_reserved = await self._reserved_sum(
                    conn=conn, tenant=tenant, project=project, user_id=user_id,
                    exclude_reservation_id=reservation_id
                )
                available = purchased - consumed - other_reserved
                if available < 0:
                    available = 0

                consume = min(int(tokens), int(available))

                if consume > 0:
                    await conn.execute(f"""
                        UPDATE {self.CONTROL_PLANE_SCHEMA}.{self.TABLE_NAME}
                        SET lifetime_tokens_consumed = LEAST(
                                COALESCE(lifetime_tokens_consumed, 0) + $4,
                                COALESCE(lifetime_tokens_purchased, 0)
                            ),
                            updated_at = NOW()
                        WHERE tenant = $1 AND project = $2 AND user_id = $3
                    """, tenant, project, user_id, consume)

                await conn.execute(f"""
                    UPDATE {self.CONTROL_PLANE_SCHEMA}.{self.RESERVATIONS_TABLE}
                    SET status = 'committed',
                        tokens_used = $5,
                        committed_at = NOW(),
                        expires_at = NOW(),
                        updated_at = NOW()
                    WHERE tenant = $1 AND project = $2 AND user_id = $3 AND reservation_id = $4
                """, tenant, project, user_id, reservation_id, consume)

        await self._invalidate_cache(tenant, project, user_id)
        return max(int(tokens) - int(consume), 0)

    async def consume_lifetime_tokens(
            self,
            *,
            tenant: str,
            project: str,
            user_id: str,
            tokens: int,
    ) -> int:
        """
        Reservation-aware consumption WITHOUT a reservation_id.
        This will NOT steal tokens that are currently reserved by other in-flight requests.

        Returns overflow tokens.
        """
        if not self._pg_pool or tokens <= 0:
            return tokens

        async with self._pg_pool.acquire() as conn:
            async with conn.transaction():
                bal = await conn.fetchrow(f"""
                    SELECT
                        COALESCE(lifetime_tokens_purchased, 0) AS purchased,
                        COALESCE(lifetime_tokens_consumed, 0) AS consumed
                    FROM {self.CONTROL_PLANE_SCHEMA}.{self.TABLE_NAME}
                    WHERE tenant = $1 AND project = $2 AND user_id = $3
                      AND active = TRUE
                      AND purchase_amount_usd IS NOT NULL
                      AND purchase_amount_usd > 0
                      AND (expires_at IS NULL OR expires_at > NOW())
                    FOR UPDATE
                """, tenant, project, user_id)

                if not bal:
                    return tokens

                purchased = int(bal["purchased"] or 0)
                consumed = int(bal["consumed"] or 0)

                reserved = await self._reserved_sum(conn=conn, tenant=tenant, project=project, user_id=user_id)
                available = purchased - consumed - reserved
                if available < 0:
                    available = 0

                consume = min(int(tokens), int(available))
                if consume > 0:
                    await conn.execute(f"""
                        UPDATE {self.CONTROL_PLANE_SCHEMA}.{self.TABLE_NAME}
                        SET lifetime_tokens_consumed = LEAST(
                                COALESCE(lifetime_tokens_consumed, 0) + $4,
                                COALESCE(lifetime_tokens_purchased, 0)
                            ),
                            updated_at = NOW()
                        WHERE tenant = $1 AND project = $2 AND user_id = $3
                    """, tenant, project, user_id, consume)

        await self._invalidate_cache(tenant, project, user_id)
        return max(int(tokens) - int(consume), 0)