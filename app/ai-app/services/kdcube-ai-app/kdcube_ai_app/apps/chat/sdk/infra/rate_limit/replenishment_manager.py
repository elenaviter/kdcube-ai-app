# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# apps/chat/sdk/infra/rate_limit/replenishment_manager.py

"""
Async Replenishment Manager

Integrates PostgreSQL (source of truth) with Redis (cache) for user quota replenishments.
Automatically invalidates Redis cache on updates.
"""

from __future__ import annotations

import logging
import time
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

import asyncpg
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


@dataclass
class QuotaReplenishment:
    """
    Additional quotas for a user (add to base policy).

    All fields are incremental - they ADD to the base policy limits.
    NULL/None means no additional quota for that dimension.
    """
    # Identification
    tenant: str
    project: str
    user_id: str
    bundle_id: str

    # Additional quotas (add to base policy)
    additional_requests_per_day: Optional[int] = None
    additional_requests_per_month: Optional[int] = None
    additional_total_requests: Optional[int] = None
    additional_tokens_per_hour: Optional[int] = None
    additional_tokens_per_day: Optional[int] = None
    additional_tokens_per_month: Optional[int] = None
    additional_max_concurrent: Optional[int] = None

    # Metadata
    expires_at: Optional[datetime] = None
    purchase_id: Optional[str] = None
    active: bool = True

    def is_expired(self) -> bool:
        """Check if this replenishment has expired."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) >= self.expires_at

    def is_valid(self) -> bool:
        """Check if this replenishment is valid and active."""
        return self.active and not self.is_expired()


class ReplenishmentManager:
    """
    Async manager for user quota replenishments.

    Integrates:
    - PostgreSQL: Source of truth (control_plane.user_quota_replenishment)
    - Redis: Three-level cache (memory → Redis → PostgreSQL)

    Automatically invalidates cache on updates to keep Redis in sync.
    """

    CONTROL_PLANE_SCHEMA = "control_plane"

    def __init__(
            self,
            pg_pool: Optional[asyncpg.Pool] = None,
            redis: Optional[Redis] = None,
            *,
            cache_ttl: int = 10,  # 10 seconds Redis cache
            cache_namespace: str = "kdcube:quota:replenishment",
    ):
        """
        Initialize replenishment manager.

        Args:
            pg_pool: asyncpg connection pool
            redis: Redis client for caching
            cache_ttl: Cache TTL in seconds (default: 10)
            cache_namespace: Redis namespace for cache keys
        """
        self._pg_pool: Optional[asyncpg.Pool] = pg_pool
        self._redis: Optional[Redis] = redis
        self.cache_ttl = cache_ttl
        self.cache_ns = cache_namespace

        # In-memory cache for ultra-fast lookups (process-local)
        self._memory_cache: Dict[str, tuple[Optional[QuotaReplenishment], float]] = {}
        self._memory_cache_ttl = 5  # 5 seconds for memory cache

        # Track if we own the pool/redis (for cleanup)
        self._owns_pool = pg_pool is None
        self._owns_redis = redis is None

    async def init(
            self,
            *,
            pg_host: Optional[str] = None,
            pg_port: Optional[int] = None,
            pg_user: Optional[str] = None,
            pg_password: Optional[str] = None,
            pg_database: Optional[str] = None,
            pg_ssl: Optional[str] = None,
            redis_url: Optional[str] = None,
    ):
        """
        Initialize connections if not provided in constructor.

        Args:
            pg_*: PostgreSQL connection parameters (if pool not provided)
            redis_url: Redis URL (if redis not provided)
        """
        # Initialize PostgreSQL pool if needed
        if not self._pg_pool and pg_host:
            import os
            self._pg_pool = await asyncpg.create_pool(
                host=pg_host or os.environ.get("POSTGRES_HOST"),
                port=pg_port or int(os.environ.get("POSTGRES_PORT", 5432)),
                user=pg_user or os.environ.get("POSTGRES_USER"),
                password=pg_password or os.environ.get("POSTGRES_PASSWORD"),
                database=pg_database or os.environ.get("POSTGRES_DATABASE"),
                ssl=pg_ssl or os.environ.get("POSTGRES_SSL", "disable"),
            )
            self._owns_pool = True

        # Initialize Redis if needed
        if not self._redis and redis_url:
            self._redis = Redis.from_url(redis_url)
            self._owns_redis = True

    async def close(self):
        """Close connections if we own them."""
        if self._owns_pool and self._pg_pool:
            await self._pg_pool.close()
        if self._owns_redis and self._redis:
            await self._redis.close()

    def _cache_key(self, tenant: str, project: str, user_id: str, bundle_id: str) -> str:
        """Build Redis cache key."""
        return f"{self.cache_ns}:{tenant}:{project}:{user_id}:{bundle_id}"

    def _check_memory_cache(
            self, tenant: str, project: str, user_id: str, bundle_id: str
    ) -> Optional[QuotaReplenishment]:
        """Check process-local memory cache."""
        cache_key = self._cache_key(tenant, project, user_id, bundle_id)

        if cache_key in self._memory_cache:
            replenishment, cached_at = self._memory_cache[cache_key]

            # Check if cache is still valid
            if time.time() - cached_at < self._memory_cache_ttl:
                logger.debug(f"Replenishment cache HIT (memory): {user_id}")
                return replenishment
            else:
                # Expired, remove from cache
                del self._memory_cache[cache_key]

        return None

    def _store_memory_cache(
            self, tenant: str, project: str, user_id: str, bundle_id: str,
            replenishment: Optional[QuotaReplenishment]
    ):
        """Store in process-local memory cache."""
        cache_key = self._cache_key(tenant, project, user_id, bundle_id)
        self._memory_cache[cache_key] = (replenishment, time.time())

        # Cleanup old entries (simple LRU with max 1000 entries)
        if len(self._memory_cache) > 1000:
            # Remove oldest 100 entries
            sorted_items = sorted(self._memory_cache.items(), key=lambda x: x[1][1])
            for key, _ in sorted_items[:100]:
                del self._memory_cache[key]

    async def _check_redis_cache(
            self, tenant: str, project: str, user_id: str, bundle_id: str
    ) -> Optional[QuotaReplenishment]:
        """Check Redis cache."""
        if not self._redis:
            return None

        cache_key = self._cache_key(tenant, project, user_id, bundle_id)

        try:
            cached = await self._redis.get(cache_key)
            if cached:
                logger.debug(f"Replenishment cache HIT (Redis): {user_id}")
                data = json.loads(cached.decode())

                # Reconstruct QuotaReplenishment
                if data:
                    # Convert expires_at string back to datetime
                    if data.get('expires_at'):
                        data['expires_at'] = datetime.fromisoformat(data['expires_at'])

                    replenishment = QuotaReplenishment(**data)

                    # Validate before returning
                    if replenishment.is_valid():
                        return replenishment

                # Invalid or expired - return None but keep cached (avoids DB hits)
                return None
        except Exception as e:
            logger.warning(f"Redis cache read error: {e}")

        return None

    async def _store_redis_cache(
            self, tenant: str, project: str, user_id: str, bundle_id: str,
            replenishment: Optional[QuotaReplenishment]
    ):
        """Store in Redis cache."""
        if not self._redis:
            return

        cache_key = self._cache_key(tenant, project, user_id, bundle_id)

        try:
            if replenishment:
                # Convert to dict and handle datetime serialization
                data = asdict(replenishment)
                if data.get('expires_at'):
                    data['expires_at'] = data['expires_at'].isoformat()

                await self._redis.setex(
                    cache_key,
                    self.cache_ttl,
                    json.dumps(data)
                )
            else:
                # Cache "not found" result to avoid repeated DB queries
                await self._redis.setex(cache_key, self.cache_ttl, json.dumps(None))
        except Exception as e:
            logger.warning(f"Redis cache write error: {e}")

    async def _query_database(
            self, tenant: str, project: str, user_id: str, bundle_id: str
    ) -> Optional[QuotaReplenishment]:
        """
        Query PostgreSQL for replenishment.

        Tries bundle-specific first, then global (bundle_id='*').
        """
        if not self._pg_pool:
            return None

        try:
            async with self._pg_pool.acquire() as conn:
                # Try bundle-specific first
                row = await conn.fetchrow(f"""
                    SELECT 
                        tenant, project, user_id, bundle_id,
                        additional_requests_per_day,
                        additional_requests_per_month,
                        additional_total_requests,
                        additional_tokens_per_hour,
                        additional_tokens_per_day,
                        additional_tokens_per_month,
                        additional_max_concurrent,
                        expires_at, purchase_id, active
                    FROM {self.CONTROL_PLANE_SCHEMA}.user_quota_replenishment
                    WHERE tenant = $1 
                        AND project = $2 
                        AND user_id = $3 
                        AND bundle_id = $4
                        AND active = TRUE
                    ORDER BY created_at DESC
                    LIMIT 1
                """, tenant, project, user_id, bundle_id)

                # If not found, try global replenishment (bundle_id='*')
                if not row:
                    row = await conn.fetchrow(f"""
                        SELECT 
                            tenant, project, user_id, bundle_id,
                            additional_requests_per_day,
                            additional_requests_per_month,
                            additional_total_requests,
                            additional_tokens_per_hour,
                            additional_tokens_per_day,
                            additional_tokens_per_month,
                            additional_max_concurrent,
                            expires_at, purchase_id, active
                        FROM {self.CONTROL_PLANE_SCHEMA}.user_quota_replenishment
                        WHERE tenant = $1 
                            AND project = $2 
                            AND user_id = $3 
                            AND bundle_id = '*'
                            AND active = TRUE
                        ORDER BY created_at DESC
                        LIMIT 1
                    """, tenant, project, user_id)

                if row:
                    logger.debug(f"Replenishment cache MISS (DB hit): {user_id}")
                    replenishment = QuotaReplenishment(**dict(row))

                    # Validate before returning
                    if replenishment.is_valid():
                        return replenishment

                return None
        except Exception as e:
            logger.error(f"Database query error for replenishment: {e}")
            return None

    async def get_replenishment(
            self,
            *,
            tenant: str,
            project: str,
            user_id: str,
            bundle_id: str,
    ) -> Optional[QuotaReplenishment]:
        """
        Get quota replenishment for a user.

        Lookup order:
        1. Memory cache (process-local, 5s TTL)
        2. Redis cache (cross-process, 10s TTL)
        3. PostgreSQL (source of truth)

        If bundle-specific replenishment not found, tries global (bundle_id='*').

        Args:
            tenant: Tenant ID
            project: Project ID
            user_id: User ID
            bundle_id: Bundle ID

        Returns:
            QuotaReplenishment if found and valid, None otherwise
        """
        # Step 1: Check memory cache
        memory_result = self._check_memory_cache(tenant, project, user_id, bundle_id)
        if memory_result is not None:
            return memory_result

        # Step 2: Check Redis cache
        redis_result = await self._check_redis_cache(tenant, project, user_id, bundle_id)
        if redis_result is not None:
            self._store_memory_cache(tenant, project, user_id, bundle_id, redis_result)
            return redis_result

        # Step 3: Query PostgreSQL
        replenishment = await self._query_database(tenant, project, user_id, bundle_id)

        # Cache the result (even if None)
        await self._store_redis_cache(tenant, project, user_id, bundle_id, replenishment)
        self._store_memory_cache(tenant, project, user_id, bundle_id, replenishment)

        return replenishment

    async def create_replenishment(
            self,
            *,
            tenant: str,
            project: str,
            user_id: str,
            bundle_id: str,
            additional_requests_per_day: Optional[int] = None,
            additional_requests_per_month: Optional[int] = None,
            additional_total_requests: Optional[int] = None,
            additional_tokens_per_hour: Optional[int] = None,
            additional_tokens_per_day: Optional[int] = None,
            additional_tokens_per_month: Optional[int] = None,
            additional_max_concurrent: Optional[int] = None,
            expires_at: Optional[datetime] = None,
            purchase_id: Optional[str] = None,
            purchase_amount_usd: Optional[float] = None,
            purchase_notes: Optional[str] = None,
    ) -> QuotaReplenishment:
        """
        Create or update quota replenishment for a user.

        Automatically invalidates Redis + memory cache after update.

        Args:
            tenant: Tenant ID
            project: Project ID
            user_id: User ID
            bundle_id: Bundle ID (use '*' for all bundles)
            additional_*: Additional quotas to add to base policy
            expires_at: When these quotas expire (None = never)
            purchase_id: Reference to payment/transaction
            purchase_amount_usd: Purchase amount in USD
            purchase_notes: Notes about the purchase

        Returns:
            Created QuotaReplenishment
        """
        if not self._pg_pool:
            raise RuntimeError("PostgreSQL pool not initialized")

        async with self._pg_pool.acquire() as conn:
            row = await conn.fetchrow(f"""
                INSERT INTO {self.CONTROL_PLANE_SCHEMA}.user_quota_replenishment (
                    tenant, project, user_id, bundle_id,
                    additional_requests_per_day,
                    additional_requests_per_month,
                    additional_total_requests,
                    additional_tokens_per_hour,
                    additional_tokens_per_day,
                    additional_tokens_per_month,
                    additional_max_concurrent,
                    expires_at,
                    purchase_id,
                    purchase_amount_usd,
                    purchase_notes
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                ON CONFLICT (tenant, project, user_id, bundle_id)
                DO UPDATE SET
                    additional_requests_per_day = COALESCE(EXCLUDED.additional_requests_per_day, {self.CONTROL_PLANE_SCHEMA}.user_quota_replenishment.additional_requests_per_day),
                    additional_requests_per_month = COALESCE(EXCLUDED.additional_requests_per_month, {self.CONTROL_PLANE_SCHEMA}.user_quota_replenishment.additional_requests_per_month),
                    additional_total_requests = COALESCE(EXCLUDED.additional_total_requests, {self.CONTROL_PLANE_SCHEMA}.user_quota_replenishment.additional_total_requests),
                    additional_tokens_per_hour = COALESCE(EXCLUDED.additional_tokens_per_hour, {self.CONTROL_PLANE_SCHEMA}.user_quota_replenishment.additional_tokens_per_hour),
                    additional_tokens_per_day = COALESCE(EXCLUDED.additional_tokens_per_day, {self.CONTROL_PLANE_SCHEMA}.user_quota_replenishment.additional_tokens_per_day),
                    additional_tokens_per_month = COALESCE(EXCLUDED.additional_tokens_per_month, {self.CONTROL_PLANE_SCHEMA}.user_quota_replenishment.additional_tokens_per_month),
                    additional_max_concurrent = COALESCE(EXCLUDED.additional_max_concurrent, {self.CONTROL_PLANE_SCHEMA}.user_quota_replenishment.additional_max_concurrent),
                    expires_at = COALESCE(EXCLUDED.expires_at, {self.CONTROL_PLANE_SCHEMA}.user_quota_replenishment.expires_at),
                    purchase_id = EXCLUDED.purchase_id,
                    purchase_amount_usd = EXCLUDED.purchase_amount_usd,
                    purchase_notes = EXCLUDED.purchase_notes,
                    updated_at = NOW()
                RETURNING *
            """,
                                      tenant, project, user_id, bundle_id,
                                      additional_requests_per_day,
                                      additional_requests_per_month,
                                      additional_total_requests,
                                      additional_tokens_per_hour,
                                      additional_tokens_per_day,
                                      additional_tokens_per_month,
                                      additional_max_concurrent,
                                      expires_at,
                                      purchase_id,
                                      purchase_amount_usd,
                                      purchase_notes
                                      )

        # Invalidate cache (both Redis and memory)
        await self._invalidate_cache(tenant, project, user_id, bundle_id)

        return QuotaReplenishment(**dict(row))

    async def _invalidate_cache(
            self, tenant: str, project: str, user_id: str, bundle_id: str
    ):
        """Invalidate cache after update (both Redis and memory)."""
        # Redis cache
        if self._redis:
            cache_key = self._cache_key(tenant, project, user_id, bundle_id)
            try:
                await self._redis.delete(cache_key)
            except Exception:
                pass

        # Memory cache
        cache_key = self._cache_key(tenant, project, user_id, bundle_id)
        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]

    async def deactivate_replenishment(
            self, *, tenant: str, project: str, user_id: str, bundle_id: str
    ):
        """
        Deactivate a replenishment (soft delete).

        Automatically invalidates Redis + memory cache.
        """
        if not self._pg_pool:
            raise RuntimeError("PostgreSQL pool not initialized")

        async with self._pg_pool.acquire() as conn:
            await conn.execute(f"""
                UPDATE {self.CONTROL_PLANE_SCHEMA}.user_quota_replenishment
                SET active = FALSE, updated_at = NOW()
                WHERE tenant = $1 AND project = $2 AND user_id = $3 AND bundle_id = $4
            """, tenant, project, user_id, bundle_id)

        await self._invalidate_cache(tenant, project, user_id, bundle_id)

    async def list_user_replenishments(
            self, *, tenant: str, project: str, user_id: str, include_expired: bool = False
    ) -> List[QuotaReplenishment]:
        """
        List all replenishments for a user (across all bundles).

        Args:
            tenant: Tenant ID
            project: Project ID
            user_id: User ID
            include_expired: Include expired replenishments

        Returns:
            List of QuotaReplenishment objects
        """
        if not self._pg_pool:
            return []

        expired_filter = "" if include_expired else "AND (expires_at IS NULL OR expires_at > NOW())"

        async with self._pg_pool.acquire() as conn:
            rows = await conn.fetch(f"""
                SELECT *
                FROM {self.CONTROL_PLANE_SCHEMA}.user_quota_replenishment
                WHERE tenant = $1 AND project = $2 AND user_id = $3 AND active = TRUE
                {expired_filter}
                ORDER BY created_at DESC
            """, tenant, project, user_id)

        return [QuotaReplenishment(**dict(row)) for row in rows]

    async def cleanup_expired(self) -> int:
        """
        Cleanup expired replenishments (soft delete by setting active=FALSE).

        Returns number of replenishments deactivated.
        """
        if not self._pg_pool:
            return 0

        async with self._pg_pool.acquire() as conn:
            result = await conn.execute(f"""
                UPDATE {self.CONTROL_PLANE_SCHEMA}.user_quota_replenishment
                SET active = FALSE, updated_at = NOW()
                WHERE active = TRUE
                  AND expires_at IS NOT NULL
                  AND expires_at <= NOW()
            """)

        # Clear all memory cache (could have affected many users)
        self._memory_cache.clear()

        # Note: Redis cache will expire naturally (10s TTL)

        try:
            return int(result.split()[-1])
        except Exception:
            return 0