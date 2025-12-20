# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# apps/chat/sdk/infra/economics/stripe.py
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple, Callable

import asyncpg

from kdcube_ai_app.apps.chat.sdk.infra.economics.subscription import SubscriptionManager
from kdcube_ai_app.apps.chat.sdk.infra.economics.user_budget import UserCreditsManager
from kdcube_ai_app.infra.accounting.usage import quote_tokens_for_usd
from kdcube_ai_app.apps.chat.sdk.infra.economics.project_budget import ProjectBudgetLimiter

logger = logging.getLogger(__name__)

BudgetLimiterFactory = Callable[[str, str], ProjectBudgetLimiter]


@dataclass(frozen=True)
class StripeHandleResult:
    status: str
    action: str
    message: str
    kind: Optional[str] = None
    external_id: Optional[str] = None
    tenant: Optional[str] = None
    project: Optional[str] = None
    user_id: Optional[str] = None

@dataclass(frozen=True)
class StripeCreateSubscriptionResult:
    status: str
    action: str
    message: str
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    stripe_latest_invoice_id: Optional[str] = None
    stripe_payment_intent_id: Optional[str] = None
    tenant: Optional[str] = None
    project: Optional[str] = None
    user_id: Optional[str] = None

class StripeSubscriptionService:
    """
    Stripe subscription creation (admin-driven for now).
    Keeps Stripe-related logic out of routes.
    """

    def __init__(
            self,
            *,
            pg_pool: asyncpg.Pool,
            subscription_mgr: SubscriptionManager,
            default_tenant: str,
            default_project: str,
            stripe_api_key: Optional[str] = None,
    ):
        self.pg_pool = pg_pool
        self.subscription_mgr = subscription_mgr
        self.default_tenant = default_tenant
        self.default_project = default_project
        self.stripe_api_key = stripe_api_key or os.getenv("STRIPE_SECRET_KEY") or os.getenv("STRIPE_API_KEY")

    def _stripe(self):
        import stripe
        if not self.stripe_api_key:
            raise RuntimeError("Stripe API key not configured (STRIPE_SECRET_KEY)")
        stripe.api_key = self.stripe_api_key
        return stripe

    @staticmethod
    def _map_stripe_subscription_status_to_cp(status: str) -> str:
        # CP constraint: active|canceled|suspended
        s = (status or "").lower()
        if s in ("active", "trialing"):
            return "active"
        if s in ("canceled",):
            return "canceled"
        return "suspended"

    async def create_subscription(
            self,
            *,
            tenant: str,
            project: str,
            user_id: str,
            tier: str,
            stripe_price_id: str,
            stripe_customer_id: Optional[str] = None,
            monthly_price_cents_hint: Optional[int] = None,
            metadata: Optional[Dict[str, str]] = None,
    ) -> StripeCreateSubscriptionResult:
        """
        Creates a Stripe subscription and stores a best-effort snapshot in user_subscriptions.
        NOTE: If the customer has no default payment method, the subscription may start as 'incomplete'
        until payment is completed.
        """
        tier = (tier or "paid").lower()
        if tier not in ("free", "paid", "premium", "admin"):
            tier = "paid"

        stripe = self._stripe()

        md = dict(metadata or {})
        md.setdefault("tenant", tenant or self.default_tenant)
        md.setdefault("project", project or self.default_project)
        md.setdefault("user_id", user_id)
        md.setdefault("tier", tier)
        md.setdefault("kdcube_invoice_kind", "subscription")

        # Ensure customer
        customer_id = stripe_customer_id
        if not customer_id:
            customer = stripe.Customer.create(metadata=md)
            customer_id = str(customer["id"])

        # If you want invoice.paid to be resolvable without relying on invoice.metadata,
        # it’s enough that subscription/customer have metadata; webhook will fetch them.
        sub = stripe.Subscription.create(
            customer=customer_id,
            items=[{"price": stripe_price_id}],
            metadata=md,
            payment_behavior="default_incomplete",
            expand=["latest_invoice.payment_intent"],
        )

        sub_id = str(sub.get("id"))
        sub_status = self._map_stripe_subscription_status_to_cp(str(sub.get("status") or ""))

        # next charge info (Stripe gives current_period_end)
        next_charge_at = None
        cpe = sub.get("current_period_end")
        if cpe:
            next_charge_at = datetime.fromtimestamp(int(cpe), tz=timezone.utc)

        # price cents
        price_cents = int(monthly_price_cents_hint or 0)
        if price_cents <= 0:
            try:
                price_obj = stripe.Price.retrieve(stripe_price_id)
                unit_amount = price_obj.get("unit_amount")
                if unit_amount:
                    price_cents = int(unit_amount)
            except Exception:
                price_cents = 0

        latest_invoice = sub.get("latest_invoice") or {}
        inv_id = str(latest_invoice.get("id") or "") or None
        pi = (latest_invoice.get("payment_intent") or {})
        pi_id = str(pi.get("id") or "") or None

        # Store snapshot (best effort). Webhook invoice.paid will do the authoritative “charged” update.
        async with self.pg_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(f"""
                    INSERT INTO {SubscriptionManager.CP}.{SubscriptionManager.TABLE} (
                      tenant, project, user_id,
                      tier, status, monthly_price_cents,
                      started_at, next_charge_at, last_charged_at,
                      provider, stripe_customer_id, stripe_subscription_id
                    ) VALUES (
                      $1,$2,$3,
                      $4,$5,$6,
                      NOW(), $7, NULL,
                      'stripe', $8, $9
                    )
                    ON CONFLICT (tenant, project, user_id)
                    DO UPDATE SET
                      provider='stripe',
                      tier=EXCLUDED.tier,
                      status=EXCLUDED.status,
                      monthly_price_cents=CASE
                        WHEN EXCLUDED.monthly_price_cents > 0 THEN EXCLUDED.monthly_price_cents
                        ELSE {SubscriptionManager.CP}.{SubscriptionManager.TABLE}.monthly_price_cents
                      END,
                      next_charge_at=COALESCE(EXCLUDED.next_charge_at, {SubscriptionManager.CP}.{SubscriptionManager.TABLE}.next_charge_at),
                      stripe_customer_id=COALESCE(EXCLUDED.stripe_customer_id, {SubscriptionManager.CP}.{SubscriptionManager.TABLE}.stripe_customer_id),
                      stripe_subscription_id=COALESCE(EXCLUDED.stripe_subscription_id, {SubscriptionManager.CP}.{SubscriptionManager.TABLE}.stripe_subscription_id),
                      updated_at=NOW()
                """, tenant, project, user_id, tier, sub_status, price_cents, next_charge_at, customer_id, sub_id)

        return StripeCreateSubscriptionResult(
            status="ok",
            action="created",
            message=f"Stripe subscription created: {sub_id} (status={sub_status})",
            stripe_customer_id=customer_id,
            stripe_subscription_id=sub_id,
            stripe_latest_invoice_id=inv_id,
            stripe_payment_intent_id=pi_id,
            tenant=tenant,
            project=project,
            user_id=user_id,
        )

class StripeEconomicsWebhookHandler:
    """
    Stripe webhook handler that:
      - verifies Stripe signatures
      - provides idempotency via kdcube_control_plane.external_economics_events
      - applies effects via managers:
          * TierBalanceManager (user credits)
          * ProjectBudgetLimiter (project budget topups + ledger)
      - maintains subscription state in user_subscriptions (DB upsert)

    Supported:
      - payment_intent.succeeded -> add lifetime user tokens
      - invoice.paid -> if subscription invoice -> top up project budget + update user_subscriptions
    """

    CP = "kdcube_control_plane"

    def __init__(
        self,
        *,
        pg_pool: asyncpg.Pool,
        user_credits_mgr: UserCreditsManager,
        budget_limiter_factory: BudgetLimiterFactory,
        subscription_mgr: SubscriptionManager,
        default_tenant: str,
        default_project: str,
        stripe_webhook_secret: Optional[str] = None,
        ref_provider: str = "anthropic",
        ref_model: str = "claude-sonnet-4-5-20250929",
    ):
        self.pg_pool = pg_pool
        self.user_credits_mgr = user_credits_mgr
        self.budget_limiter_factory = budget_limiter_factory
        self.subscription_mgr = subscription_mgr

        self.default_tenant = default_tenant
        self.default_project = default_project

        self.webhook_secret = stripe_webhook_secret or os.getenv("STRIPE_WEBHOOK_SECRET")
        self.stripe_api_key = os.getenv("STRIPE_SECRET_KEY") or os.getenv("STRIPE_API_KEY")
        self.ref_provider = ref_provider
        self.ref_model = ref_model

    def _stripe(self):
        import stripe
        if not self.stripe_api_key:
            return None
        stripe.api_key = self.stripe_api_key
        return stripe

    def _merge_meta(self, *metas: Dict[str, str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for m in metas:
            if not m:
                continue
            for k, v in m.items():
                if v is None:
                    continue
                # keep first non-empty (invoice metadata wins, then lines, then subscription/customer)
                if k not in out or out[k] == "":
                    out[k] = str(v)
        return out

    def _meta_from_invoice_lines(self, invoice: Dict[str, Any]) -> Dict[str, str]:
        try:
            lines = (invoice.get("lines") or {}).get("data") or []
            if not lines:
                return {}
            m = lines[0].get("metadata") or {}
            if not isinstance(m, dict):
                return {}
            return {str(k): str(v) for k, v in m.items() if v is not None}
        except Exception:
            return {}

    def _fetch_subscription_meta(self, subscription_id: Optional[str]) -> Dict[str, str]:
        if not subscription_id:
            return {}
        stripe = self._stripe()
        if not stripe:
            return {}
        try:
            sub = stripe.Subscription.retrieve(subscription_id)
            return self._meta(sub)
        except Exception:
            logger.exception("stripe fetch subscription metadata failed: %s", subscription_id)
            return {}

    def _fetch_customer_meta(self, customer_id: Optional[str]) -> Dict[str, str]:
        if not customer_id:
            return {}
        stripe = self._stripe()
        if not stripe:
            return {}
        try:
            cust = stripe.Customer.retrieve(customer_id)
            return self._meta(cust)
        except Exception:
            logger.exception("stripe fetch customer metadata failed: %s", customer_id)
            return {}

    def _resolve_invoice_context(self, invoice: Dict[str, Any]) -> Tuple[Dict[str, str], str, str, Optional[str]]:
        inv_meta = self._meta(invoice)
        line_meta = self._meta_from_invoice_lines(invoice)

        # Fetch from API if needed (only if API key exists)
        sub_meta = {}
        cust_meta = {}

        # only fetch if user_id missing after invoice+lines
        tmp = self._merge_meta(inv_meta, line_meta)
        if (not tmp.get("user_id")) or (not tmp.get("kdcube_invoice_kind")):
            sub_meta = self._fetch_subscription_meta(invoice.get("subscription"))
            tmp = self._merge_meta(inv_meta, line_meta, sub_meta)

        if (not tmp.get("user_id")) or (not tmp.get("kdcube_invoice_kind")):
            cust_meta = self._fetch_customer_meta(invoice.get("customer"))
            tmp = self._merge_meta(inv_meta, line_meta, sub_meta, cust_meta)

        tenant, project, user_id = self._resolve_tenant_project_user(tmp)
        return tmp, tenant, project, user_id

    async def handle_webhook(self, *, body: bytes, stripe_signature: Optional[str]) -> Dict[str, Any]:
        event = self._verify_and_parse(body=body, stripe_signature=stripe_signature)
        etype = event.get("type")
        obj = (event.get("data") or {}).get("object") or {}

        try:
            if etype == "payment_intent.succeeded":
                res = await self._handle_payment_intent_succeeded(event, obj)
            elif etype == "invoice.paid":
                res = await self._handle_invoice_paid(event, obj)
            else:
                res = StripeHandleResult(status="ok", action="unsupported", message=f"Event {etype} not processed")
        except Exception as e:
            logger.exception("Stripe webhook failed: type=%s", etype)
            return {"status": "error", "action": "failed", "message": str(e), "event_type": etype}

        return {
            "status": res.status,
            "action": res.action,
            "message": res.message,
            "kind": res.kind,
            "external_id": res.external_id,
            "tenant": res.tenant,
            "project": res.project,
            "user_id": res.user_id,
        }

    # ---------------- verification ----------------

    def _verify_and_parse(self, *, body: bytes, stripe_signature: Optional[str]) -> Dict[str, Any]:
        if not self.webhook_secret:
            logger.warning("STRIPE_WEBHOOK_SECRET not set: parsing Stripe event WITHOUT verification")
            return json.loads(body.decode("utf-8"))

        if not stripe_signature:
            raise ValueError("Missing Stripe-Signature header")

        import stripe
        return stripe.Webhook.construct_event(payload=body, sig_header=stripe_signature, secret=self.webhook_secret)

    # ---------------- helpers ----------------

    def _meta(self, obj: Dict[str, Any]) -> Dict[str, str]:
        m = obj.get("metadata") or {}
        if not isinstance(m, dict):
            return {}
        return {str(k): str(v) for k, v in m.items() if v is not None}

    def _resolve_tenant_project_user(self, meta: Dict[str, str]) -> Tuple[str, str, Optional[str]]:
        tenant = meta.get("tenant") or self.default_tenant
        project = meta.get("project") or self.default_project
        user_id = meta.get("user_id")
        return tenant, project, user_id

    def _is_subscription_invoice(self, invoice: Dict[str, Any], meta: Dict[str, str]) -> bool:
        # Your "special invoice type"
        if (meta.get("kdcube_invoice_kind") or "").lower() == "subscription":
            return True

        # Optional fallback: Stripe billing reason
        if (invoice.get("billing_reason") or "") in ("subscription_cycle", "subscription_update"):
            return True

        return False

    # ---------------- idempotency internals ----------------

    async def _lock_or_create_ext_event(
        self,
        *,
        conn: asyncpg.Connection,
        kind: str,
        external_id: str,
        tenant: str,
        project: str,
        user_id: Optional[str],
        amount_cents: Optional[int],
        tokens: Optional[int],
        currency: Optional[str],
        stripe_event_id: Optional[str],
        metadata: Dict[str, str],
    ) -> str:
        await conn.execute(f"""
            INSERT INTO {self.CP}.external_economics_events (
              source, kind, external_id,
              tenant, project, user_id,
              amount_cents, tokens, currency,
              status, stripe_event_id, metadata
            ) VALUES (
              'stripe', $1, $2,
              $3, $4, $5,
              $6, $7, $8,
              'pending', $9, $10::jsonb
            )
            ON CONFLICT (source, kind, external_id) DO NOTHING
        """, kind, external_id, tenant, project, user_id, amount_cents, tokens, currency, stripe_event_id, json.dumps(metadata))

        row = await conn.fetchrow(f"""
            SELECT status
            FROM {self.CP}.external_economics_events
            WHERE source='stripe' AND kind=$1 AND external_id=$2
            FOR UPDATE
        """, kind, external_id)

        if not row:
            raise RuntimeError("Failed to lock/create external_economics_events row")

        return str(row["status"])

    async def _mark_ext_event_applied(self, conn: asyncpg.Connection, *, kind: str, external_id: str) -> None:
        await conn.execute(f"""
            UPDATE {self.CP}.external_economics_events
            SET status='applied', applied_at=NOW(), error=NULL, updated_at=NOW()
            WHERE source='stripe' AND kind=$1 AND external_id=$2
        """, kind, external_id)

    async def _mark_ext_event_failed(self, conn: asyncpg.Connection, *, kind: str, external_id: str, error: str) -> None:
        await conn.execute(f"""
            UPDATE {self.CP}.external_economics_events
            SET status='failed', error=$3, updated_at=NOW()
            WHERE source='stripe' AND kind=$1 AND external_id=$2
        """, kind, external_id, error[:2000])

    # ---------------- event handlers ----------------

    async def _handle_payment_intent_succeeded(self, event: Dict[str, Any], pi: Dict[str, Any]) -> StripeHandleResult:
        meta = self._meta(pi)
        tenant, project, user_id = self._resolve_tenant_project_user(meta)
        if not user_id:
            return StripeHandleResult(status="ok", action="ignored", message="Missing user_id in payment_intent metadata")

        currency = (pi.get("currency") or "usd").lower()
        amount_cents = int(pi.get("amount_received") or pi.get("amount") or 0)
        if currency != "usd" or amount_cents <= 0:
            return StripeHandleResult(status="ok", action="ignored", message=f"Unsupported currency/amount ({currency},{amount_cents})")

        payment_intent_id = str(pi.get("id") or "")
        stripe_event_id = str(event.get("id") or "")
        if not payment_intent_id:
            return StripeHandleResult(status="ok", action="ignored", message="Missing payment_intent id")

        usd_amount = amount_cents / 100.0
        tokens_added, _ = quote_tokens_for_usd(
            usd_amount=usd_amount,
            ref_provider=self.ref_provider,
            ref_model=self.ref_model,
        )
        notes = meta.get("notes") or meta.get("package") or "Stripe payment_intent.succeeded"

        async with self.pg_pool.acquire() as conn:
            async with conn.transaction():
                status = await self._lock_or_create_ext_event(
                    conn=conn,
                    kind="user_credits",
                    external_id=payment_intent_id,
                    tenant=tenant,
                    project=project,
                    user_id=user_id,
                    amount_cents=amount_cents,
                    tokens=int(tokens_added),
                    currency=currency,
                    stripe_event_id=stripe_event_id,
                    metadata=meta,
                )
                if status == "applied":
                    return StripeHandleResult(
                        status="ok", action="duplicate", message="Already applied (idempotent)",
                        kind="user_credits", external_id=payment_intent_id, tenant=tenant, project=project, user_id=user_id,
                    )

                try:
                    # APPLY via TierBalanceManager inside SAME transaction
                    await self.user_credits_mgr.add_lifetime_tokens(
                        tenant=tenant,
                        project=project,
                        user_id=user_id,
                        tokens=int(tokens_added),
                        usd_amount=float(usd_amount),
                        purchase_id=payment_intent_id,
                        notes=notes,
                        conn=conn,
                    )
                    await self._mark_ext_event_applied(conn, kind="user_credits", external_id=payment_intent_id)
                except Exception as e:
                    await self._mark_ext_event_failed(conn, kind="user_credits", external_id=payment_intent_id, error=str(e))
                    raise

        return StripeHandleResult(
            status="ok",
            action="applied",
            message=f"Added user credits: +{int(tokens_added)} tokens (~${usd_amount:.2f})",
            kind="user_credits",
            external_id=payment_intent_id,
            tenant=tenant,
            project=project,
            user_id=user_id,
        )

    async def _handle_invoice_paid(self, event: Dict[str, Any], invoice: Dict[str, Any]) -> StripeHandleResult:
        meta, tenant, project, user_id = self._resolve_invoice_context(invoice)
        if not user_id:
            return StripeHandleResult(status="ok", action="ignored", message="Missing user_id (invoice/lines/subscription/customer metadata)")

        if not self._is_subscription_invoice(invoice, meta):
            return StripeHandleResult(status="ok", action="ignored", message="invoice.paid not marked as subscription invoice")

        currency = (invoice.get("currency") or "usd").lower()
        amount_cents = int(invoice.get("amount_paid") or invoice.get("amount_due") or invoice.get("total") or 0)
        if currency != "usd" or amount_cents <= 0:
            return StripeHandleResult(status="ok", action="ignored", message=f"Unsupported currency/amount ({currency},{amount_cents})")

        invoice_id = str(invoice.get("id") or "")
        stripe_event_id = str(event.get("id") or "")
        if not invoice_id:
            return StripeHandleResult(status="ok", action="ignored", message="Missing invoice id")

        usd_amount = amount_cents / 100.0
        tier = (meta.get("tier") or meta.get("plan") or "paid").lower()
        if tier not in ("free", "paid", "premium", "admin"):
            tier = "paid"

        stripe_customer_id = invoice.get("customer")
        stripe_subscription_id = invoice.get("subscription")

        next_charge_at = None
        # Prefer subscription.current_period_end if we can (more reliable), else fallback to invoice lines period end
        stripe = self._stripe()
        if stripe and stripe_subscription_id:
            try:
                sub = stripe.Subscription.retrieve(stripe_subscription_id)
                cpe = sub.get("current_period_end")
                if cpe:
                    next_charge_at = datetime.fromtimestamp(int(cpe), tz=timezone.utc)
            except Exception:
                logger.exception("Failed to fetch subscription current_period_end: %s", stripe_subscription_id)

        if not next_charge_at:
            try:
                lines = (invoice.get("lines") or {}).get("data") or []
                if lines:
                    period_end = ((lines[0].get("period") or {}).get("end"))
                    if period_end:
                        next_charge_at = datetime.fromtimestamp(int(period_end), tz=timezone.utc)
            except Exception:
                next_charge_at = None

        notes = meta.get("notes") or f"Stripe subscription invoice.paid {invoice_id}"
        request_id = f"stripe:invoice:{invoice_id}"

        budget = self.budget_limiter_factory(tenant, project)

        async with self.pg_pool.acquire() as conn:
            async with conn.transaction():
                status = await self._lock_or_create_ext_event(
                    conn=conn,
                    kind="subscription_topup",
                    external_id=invoice_id,
                    tenant=tenant,
                    project=project,
                    user_id=user_id,
                    amount_cents=amount_cents,
                    tokens=None,
                    currency=currency,
                    stripe_event_id=stripe_event_id,
                    metadata=meta,
                )
                if status == "applied":
                    return StripeHandleResult(
                        status="ok", action="duplicate", message="Already applied (idempotent)",
                        kind="subscription_topup", external_id=invoice_id, tenant=tenant, project=project, user_id=user_id,
                    )

                try:
                    # APPLY via ProjectBudgetLimiter inside SAME transaction (+ledger)
                    await budget.topup_app_budget(
                        usd_amount=float(usd_amount),
                        notes=notes,
                        request_id=request_id,
                        conn=conn,
                        user_id=user_id
                    )

                    # Update subscription snapshot
                    now = datetime.now(timezone.utc)
                    await self.subscription_mgr.upsert_from_stripe_invoice_paid(
                        tenant=tenant,
                        project=project,
                        user_id=user_id,
                        tier=tier,
                        monthly_price_cents=int(amount_cents),
                        stripe_customer_id=stripe_customer_id,
                        stripe_subscription_id=stripe_subscription_id,
                        next_charge_at=next_charge_at,
                        charged_at=now,
                        conn=conn,
                    )

                    await self._mark_ext_event_applied(conn, kind="subscription_topup", external_id=invoice_id)
                except Exception as e:
                    await self._mark_ext_event_failed(conn, kind="subscription_topup", external_id=invoice_id, error=str(e))
                    raise

        return StripeHandleResult(
            status="ok",
            action="applied",
            message=f"Subscription invoice credited project budget: +${usd_amount:.2f}",
            kind="subscription_topup",
            external_id=invoice_id,
            tenant=tenant,
            project=project,
            user_id=user_id,
        )
