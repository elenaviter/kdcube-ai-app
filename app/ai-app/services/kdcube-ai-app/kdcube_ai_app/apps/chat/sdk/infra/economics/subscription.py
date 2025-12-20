# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# apps/chat/sdk/infra/economics/subscription.py
from __future__ import annotations
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import asyncpg
from calendar import monthrange

from kdcube_ai_app.apps.chat.sdk.infra.economics.project_budget import ProjectBudgetLimiter

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _tier_from_user_type(user_type: str) -> str:
    return {
        "registered": "free",
        "paid": "paid",
        "privileged": "premium",
        "admin": "admin",
    }.get(user_type, "unknown")


def _add_one_month(dt: datetime) -> datetime:
    # Preserve time + tz, clamp day (e.g. Jan 31 -> Feb 28/29)
    dt = dt.astimezone(timezone.utc)
    y = dt.year + (dt.month // 12)
    m = (dt.month % 12) + 1
    last_day = monthrange(y, m)[1]
    d = min(dt.day, last_day)
    return dt.replace(year=y, month=m, day=d)

@dataclass(frozen=True)
class Subscription:
    tenant: str
    project: str
    user_id: str
    tier: str
    status: str
    monthly_price_cents: int
    started_at: datetime
    next_charge_at: Optional[datetime]
    last_charged_at: Optional[datetime]
    provider: str
    stripe_customer_id: Optional[str]
    stripe_subscription_id: Optional[str]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class InternalRenewOnceResult:
    status: str            # "ok" | "error"
    action: str            # "applied" | "duplicate"
    message: str
    external_id: str
    user_id: str
    usd_amount: float
    charged_at: datetime


class SubscriptionManager:
    CP = "kdcube_control_plane"
    TABLE = "user_subscriptions"
    EXT_EVENTS_TABLE = "external_economics_events"

    def __init__(self, pg_pool: asyncpg.Pool):
        self.pg_pool = pg_pool

    def _from_row(self, row: asyncpg.Record) -> Subscription:
        return Subscription(**dict(row))

    async def get_subscription(
        self,
        *,
        tenant: str,
        project: str,
        user_id: str,
        conn: Optional[asyncpg.Connection] = None,
    ) -> Optional[Subscription]:
        sql = f"SELECT * FROM {self.CP}.{self.TABLE} WHERE tenant=$1 AND project=$2 AND user_id=$3"
        if conn:
            row = await conn.fetchrow(sql, tenant, project, user_id)
        else:
            async with self.pg_pool.acquire() as c:
                row = await c.fetchrow(sql, tenant, project, user_id)
        return self._from_row(row) if row else None

    async def ensure_subscription_for_user(
            self,
            *,
            tenant: str,
            project: str,
            user_id: str,
            user_type: str,
            monthly_price_cents_free: int = 0,
            monthly_price_cents_paid: int = 2000,
            monthly_price_cents_admin: int = 0,
            now: Optional[datetime] = None,
            conn: Optional[asyncpg.Connection] = None,
    ) -> Subscription:
        """
        Ensure a subscription row exists for a user (internal bootstrap).

        Rules:
          - If an existing row is provider='stripe', DO NOT MODIFY it at all.
          - Otherwise, ensure/refresh an internal row.
          - Internal paid/premium ARE scheduled via next_charge_at.
        """
        now = now or _now()
        tier = _tier_from_user_type(user_type)

        if tier == "unknown" or user_type == "anonymous":
            raise ValueError(f"unsupported user_type={user_type}")

        if tier == "free":
            price = int(monthly_price_cents_free)
        elif tier == "admin":
            price = int(monthly_price_cents_admin)
        else:
            # paid OR premium default to paid price
            price = int(monthly_price_cents_paid)

        tbl = f"{self.CP}.{self.TABLE}"

        sql = f"""
        INSERT INTO {tbl} (
          tenant, project, user_id,
          tier, status, monthly_price_cents,
          started_at, next_charge_at, last_charged_at,
          provider, stripe_customer_id, stripe_subscription_id
        ) VALUES (
          $1,$2,$3,
          $4,'active',$5,
          $6,
          CASE
            WHEN $4 IN ('paid','premium') AND $5 > 0 THEN $6
            ELSE NULL
          END,
          NULL,
          'internal', NULL, NULL
        )
        ON CONFLICT (tenant, project, user_id)
        DO UPDATE SET
          tier = EXCLUDED.tier,
          status = 'active',
          monthly_price_cents = EXCLUDED.monthly_price_cents,

          -- keep started_at stable
          started_at = {tbl}.started_at,

          -- schedule only if chargeable
          next_charge_at = CASE
            WHEN EXCLUDED.tier IN ('paid','premium') AND EXCLUDED.monthly_price_cents > 0 THEN
              COALESCE(
                {tbl}.next_charge_at,
                ({tbl}.last_charged_at + interval '1 month'),
                $6
              )
            ELSE NULL
          END,

          provider = 'internal',

          stripe_customer_id = NULL,
          stripe_subscription_id = NULL,

          updated_at = NOW()
        WHERE {tbl}.provider IS DISTINCT FROM 'stripe'
        RETURNING *
        """

        async def _run(c: asyncpg.Connection) -> asyncpg.Record:
            row = await c.fetchrow(sql, tenant, project, user_id, tier, price, now)
            if row:
                return row
            # If conflict row exists and is stripe => UPDATE skipped => RETURNING empty => fetch existing row
            existing = await c.fetchrow(
                f"SELECT * FROM {tbl} WHERE tenant=$1 AND project=$2 AND user_id=$3",
                tenant, project, user_id
            )
            if not existing:
                raise RuntimeError(f"failed to ensure subscription row for {tenant}/{project}/{user_id}")
            return existing

        if conn:
            row = await _run(conn)
        else:
            async with self.pg_pool.acquire() as c:
                row = await _run(c)

        return self._from_row(row)


    async def upsert_from_stripe_invoice_paid(
            self,
            *,
            tenant: str,
            project: str,
            user_id: str,
            tier: str,
            monthly_price_cents: int,
            stripe_customer_id: Optional[str],
            stripe_subscription_id: Optional[str],
            next_charge_at: Optional[datetime],
            charged_at: Optional[datetime] = None,
            conn: Optional[asyncpg.Connection] = None,
    ) -> Subscription:
        charged_at = charged_at or _now()

        sql = f"""
        INSERT INTO {self.CP}.{self.TABLE} (
          tenant, project, user_id,
          tier, status, monthly_price_cents,
          started_at, next_charge_at, last_charged_at,
          provider, stripe_customer_id, stripe_subscription_id
        ) VALUES ($1,$2,$3, $4,'active',$5, NOW(), $6,$7, 'stripe', $8,$9)
        ON CONFLICT (tenant, project, user_id)
        DO UPDATE SET
          tier=EXCLUDED.tier,
          status='active',
          monthly_price_cents=EXCLUDED.monthly_price_cents,
          -- keep started_at stable (don’t reset on renewals)
          started_at={self.CP}.{self.TABLE}.started_at,
          last_charged_at=EXCLUDED.last_charged_at,
          next_charge_at=COALESCE(EXCLUDED.next_charge_at, {self.CP}.{self.TABLE}.next_charge_at),
          provider='stripe',
          stripe_customer_id=COALESCE(EXCLUDED.stripe_customer_id, {self.CP}.{self.TABLE}.stripe_customer_id),
          stripe_subscription_id=COALESCE(EXCLUDED.stripe_subscription_id, {self.CP}.{self.TABLE}.stripe_subscription_id),
          updated_at=NOW()
        RETURNING *
        """

        if conn:
            row = await conn.fetchrow(
                sql, tenant, project, user_id, tier, int(monthly_price_cents),
                next_charge_at, charged_at, stripe_customer_id, stripe_subscription_id
            )
        else:
            async with self.pg_pool.acquire() as c:
                row = await c.fetchrow(
                    sql, tenant, project, user_id, tier, int(monthly_price_cents),
                    next_charge_at, charged_at, stripe_customer_id, stripe_subscription_id
                )
        return self._from_row(row)

    async def list_due_internal_paid_subscriptions(
            self,
            *,
            tenant: str,
            project: str,
            now: Optional[datetime] = None,
            limit: int = 500,
            conn: Optional[asyncpg.Connection] = None,
    ) -> List[Subscription]:
        now = now or _now()
        sql = f"""
        SELECT *
        FROM {self.CP}.{self.TABLE}
        WHERE tenant=$1 AND project=$2
          AND provider='internal'
          AND tier IN ('paid','premium')
          AND status='active'
          AND next_charge_at IS NOT NULL
          AND next_charge_at <= $3
        ORDER BY next_charge_at ASC
        LIMIT {int(limit)}
        """
        if conn:
            rows = await conn.fetch(sql, tenant, project, now)
        else:
            async with self.pg_pool.acquire() as c:
                rows = await c.fetch(sql, tenant, project, now)
        return [self._from_row(r) for r in rows]

    async def list_subscriptions(
            self,
            *,
            tenant: str,
            project: str,
            provider: Optional[str] = None,
            user_id: Optional[str] = None,
            limit: int = 100,
            offset: int = 0,
            conn: Optional[asyncpg.Connection] = None,
    ) -> List[Subscription]:
        tbl = f"{self.CP}.{self.TABLE}"
        where = ["tenant=$1", "project=$2"]
        args = [tenant, project]
        i = 3

        if provider:
            where.append(f"provider=${i}")
            args.append(provider)
            i += 1
        if user_id:
            where.append(f"user_id=${i}")
            args.append(user_id)
            i += 1

        sql = f"""
        SELECT *
        FROM {tbl}
        WHERE {" AND ".join(where)}
        ORDER BY updated_at DESC
        LIMIT {int(limit)} OFFSET {int(offset)}
        """

        if conn:
            rows = await conn.fetch(sql, *args)
        else:
            async with self.pg_pool.acquire() as c:
                rows = await c.fetch(sql, *args)

        return [self._from_row(r) for r in rows]

    async def mark_internal_charge(
            self,
            *,
            tenant: str,
            project: str,
            user_id: str,
            charged_at: Optional[datetime],
            next_charge_at: Optional[datetime],
            conn: Optional[asyncpg.Connection] = None,
    ) -> Subscription:
        charged_at = charged_at or _now()
        sql = f"""
        UPDATE {self.CP}.{self.TABLE}
        SET last_charged_at=$4, next_charge_at=$5, updated_at=NOW()
        WHERE tenant=$1 AND project=$2 AND user_id=$3
        RETURNING *
        """
        if conn:
            row = await conn.fetchrow(sql, tenant, project, user_id, charged_at, next_charge_at)
        else:
            async with self.pg_pool.acquire() as c:
                row = await c.fetchrow(sql, tenant, project, user_id, charged_at, next_charge_at)
        if not row:
            raise ValueError(f"subscription not found: {tenant}/{project}/{user_id}")
        return self._from_row(row)


    # ---------------- internal idempotency (no SQL in routes) ----------------

    async def _lock_or_create_internal_event(
            self,
            *,
            conn: asyncpg.Connection,
            external_id: str,
            tenant: str,
            project: str,
            user_id: str,
            amount_cents: int,
            metadata: Dict[str, Any],
    ) -> str:
        tbl = f"{self.CP}.{self.EXT_EVENTS_TABLE}"
        await conn.execute(f"""
            INSERT INTO {tbl} (
              source, kind, external_id,
              tenant, project, user_id,
              amount_cents, tokens, currency,
              status, stripe_event_id, metadata
            ) VALUES (
              'internal', 'subscription_topup', $1,
              $2, $3, $4,
              $5, NULL, 'usd',
              'pending', NULL, $6::jsonb
            )
            ON CONFLICT (source, kind, external_id) DO NOTHING
        """, external_id, tenant, project, user_id, int(amount_cents), json.dumps(metadata))

        row = await conn.fetchrow(f"""
            SELECT status
            FROM {tbl}
            WHERE source='internal' AND kind='subscription_topup' AND external_id=$1
            FOR UPDATE
        """, external_id)

        if not row:
            raise RuntimeError("Failed to lock/create external_economics_events row (internal)")

        return str(row["status"])

    async def _mark_internal_event_applied(self, conn: asyncpg.Connection, *, external_id: str) -> None:
        tbl = f"{self.CP}.{self.EXT_EVENTS_TABLE}"
        await conn.execute(f"""
            UPDATE {tbl}
            SET status='applied', applied_at=NOW(), error=NULL, updated_at=NOW()
            WHERE source='internal' AND kind='subscription_topup' AND external_id=$1
        """, external_id)

    async def _mark_internal_event_failed(self, conn: asyncpg.Connection, *, external_id: str, error: str) -> None:
        tbl = f"{self.CP}.{self.EXT_EVENTS_TABLE}"
        await conn.execute(f"""
            UPDATE {tbl}
            SET status='failed', error=$2, updated_at=NOW()
            WHERE source='internal' AND kind='subscription_topup' AND external_id=$1
        """, external_id, (error or "")[:2000])

    async def renew_internal_subscription_once(
            self,
            *,
            tenant: str,
            project: str,
            user_id: str,
            budget: ProjectBudgetLimiter,
            charged_at: Optional[datetime] = None,
            idempotency_key: Optional[str] = None,
            actor: Optional[str] = None,
            conn: Optional[asyncpg.Connection] = None,
    ) -> InternalRenewOnceResult:
        """
        One-shot manual renewal for INTERNAL subscriptions:
          - idempotent via external_economics_events (source='internal', kind='subscription_topup')
          - tops up project budget
          - marks last_charged_at (and keeps next_charge_at NULL; no scheduling)
        """
        charged_at = charged_at or _now()
        next_due = _add_one_month(charged_at)
        ym = charged_at.strftime("%Y-%m")
        external_id = idempotency_key or f"internal:renew:{tenant}:{project}:{user_id}:{ym}"

        async def _run(c: asyncpg.Connection) -> InternalRenewOnceResult:
            sub = await self.get_subscription(tenant=tenant, project=project, user_id=user_id, conn=c)
            if not sub:
                raise ValueError("subscription not found")
            if sub.provider != "internal":
                raise ValueError("not an internal subscription")
            if sub.status != "active":
                raise ValueError("subscription not active")
            if sub.tier not in ("paid", "premium"):
                raise ValueError(f"tier={sub.tier} is not chargeable")
            if int(sub.monthly_price_cents or 0) <= 0:
                raise ValueError("monthly_price_cents is 0; nothing to charge")

            status = await self._lock_or_create_internal_event(
                conn=c,
                external_id=external_id,
                tenant=tenant,
                project=project,
                user_id=user_id,
                amount_cents=int(sub.monthly_price_cents),
                metadata={"tier": sub.tier, "by": actor or "unknown"},
            )
            if status == "applied":
                return InternalRenewOnceResult(
                    status="ok",
                    action="duplicate",
                    message="Already applied (idempotent)",
                    external_id=external_id,
                    user_id=user_id,
                    usd_amount=float(int(sub.monthly_price_cents) / 100.0),
                    charged_at=charged_at,
                )

            usd_amount = float(int(sub.monthly_price_cents) / 100.0)

            try:
                await budget.topup_app_budget(
                    usd_amount=usd_amount,
                    notes=f"internal subscription renewal user_id={user_id} tier={sub.tier}",
                    request_id=f"internal:renew:{external_id}",
                    conn=c,
                    user_id=user_id,
                )

                # no scheduling => next_charge_at stays NULL
                await self.mark_internal_charge(
                    tenant=tenant,
                    project=project,
                    user_id=user_id,
                    charged_at=charged_at,
                    next_charge_at=next_due,
                    conn=c,
                )

                await self._mark_internal_event_applied(c, external_id=external_id)
            except Exception as e:
                # NOTE: if transaction rolls back, this "failed" mark also rolls back (same pattern as Stripe handler).
                # That’s acceptable for now: next attempt will re-run since status remains pending/not-applied.
                await self._mark_internal_event_failed(c, external_id=external_id, error=str(e))
                raise

            return InternalRenewOnceResult(
                status="ok",
                action="applied",
                message=f"Renewed internal subscription and topped up budget: +${usd_amount:.2f}",
                external_id=external_id,
                user_id=user_id,
                usd_amount=usd_amount,
                charged_at=charged_at,
            )

        if conn:
            return await _run(conn)

        async with self.pg_pool.acquire() as c:
            async with c.transaction():
                return await _run(c)
