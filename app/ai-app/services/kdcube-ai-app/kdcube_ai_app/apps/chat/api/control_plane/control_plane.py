# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

# apps/chat/api/control_plane/control_plane.py

"""
Control Plane API

Provides REST endpoints for managing:
1. User tier balance (tier overrides + lifetime budget)
2. User quota policies (base tier limits by user type)
3. Application budget policies (spending limits per provider - NO bundle_id!)

Includes Stripe webhook integration for automated credit purchases.

Admin-only access with similar patterns to OPEX API.
"""

from typing import Optional
import logging
import hmac
import hashlib
import os
import json

from pydantic import BaseModel, Field
from fastapi import Depends, HTTPException, Request, APIRouter, Query, Header
from datetime import datetime, timedelta, timezone

from kdcube_ai_app.apps.chat.api.resolvers import auth_without_pressure
from kdcube_ai_app.apps.chat.sdk.config import get_settings
from kdcube_ai_app.apps.chat.sdk.infra.rate_limit.limiter import RateLimiter
from kdcube_ai_app.apps.chat.sdk.infra.rate_limit.project_budget_limiter import ProjectBudgetLimiter
from kdcube_ai_app.auth.sessions import UserSession

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# ============================================================================
# Request/Response Models
# ============================================================================

class GrantTrialRequest(BaseModel):
    """
    Grant temporary tier OVERRIDE (7-day trial).

    **IMPORTANT:** This OVERRIDES the user's base tier, does NOT add to it.

    Example:
    - Free user normally has: 10 req/day
    - You grant trial: 100 req/day for 7 days
    - During trial: User gets exactly 100 req/day (NOT 110)
    - After trial expires: User reverts to 10 req/day

    These quotas RESET daily/monthly like tier limits.
    """
    user_id: str = Field(..., description="User ID")
    days: int = Field(7, description="Trial duration in days")
    requests_per_day: int = Field(100, description="Requests/day during trial (OVERRIDES base)")
    requests_per_month: Optional[int] = Field(None, description="Requests/month during trial")
    tokens_per_day: Optional[int] = Field(None, description="Tokens/day during trial (OVERRIDES base)")
    tokens_per_month: Optional[int] = Field(None, description="Tokens/month during trial")
    max_concurrent: Optional[int] = Field(None, description="Max concurrent during trial")
    notes: Optional[str] = Field(None, description="Notes")


class UpdateTierBudgetRequest(BaseModel):
    """
    Update user's tier budget (tier override).

    Like trial but more flexible - can set exact limits and expiry.
    Supports PARTIAL UPDATES - only updates fields you provide!

    **IMPORTANT:** This OVERRIDES the user's base tier, does NOT add to it.
    """
    user_id: str = Field(..., description="User ID")
    requests_per_day: Optional[int] = Field(None, description="Requests/day (OVERRIDES base)")
    requests_per_month: Optional[int] = Field(None, description="Requests/month (OVERRIDES base)")
    tokens_per_day: Optional[int] = Field(None, description="Tokens/day (OVERRIDES base)")
    tokens_per_month: Optional[int] = Field(None, description="Tokens/month (OVERRIDES base)")
    max_concurrent: Optional[int] = Field(None, description="Max concurrent (OVERRIDES base)")
    expires_in_days: Optional[int] = Field(30, description="Days until expiration (None = never)")
    notes: Optional[str] = Field(None, description="Notes")


class AddLifetimeCreditsRequest(BaseModel):
    """
    Add purchased credits in USD (converted to lifetime tokens).

    Balance depletes on use, does NOT reset.
    Completely separate from tier quotas.
    """
    user_id: str = Field(..., description="User ID")
    usd_amount: float = Field(..., gt=0, description="Amount in USD")
    ref_provider: str = Field(default="anthropic", description="Reference model provider")
    ref_model: str = Field(default="claude-sonnet-4-5-20250929", description="Reference model")
    purchase_id: Optional[str] = Field(None, description="Payment/transaction ID")
    notes: Optional[str] = Field(None, description="Purchase notes")


class SetQuotaPolicyRequest(BaseModel):
    """Set quota policy for a user type (tier limits - NO bundle_id!)."""
    user_type: str = Field(..., description="User type (free, paid, premium, etc.)")
    max_concurrent: Optional[int] = Field(None, description="Max concurrent requests")
    requests_per_day: Optional[int] = Field(None, description="Requests per day")
    requests_per_month: Optional[int] = Field(None, description="Requests per month")
    total_requests: Optional[int] = Field(None, description="Total requests")
    tokens_per_hour: Optional[int] = Field(None, description="Tokens per hour")
    tokens_per_day: Optional[int] = Field(None, description="Tokens per day")
    tokens_per_month: Optional[int] = Field(None, description="Tokens per month")
    notes: Optional[str] = Field(None, description="Notes")


class SetBudgetPolicyRequest(BaseModel):
    """Set budget policy for a provider (app spending limits )."""
    provider: str = Field(..., description="Provider (anthropic, openai, etc.)")
    usd_per_hour: Optional[float] = Field(None, description="USD per hour")
    usd_per_day: Optional[float] = Field(None, description="USD per day")
    usd_per_month: Optional[float] = Field(None, description="USD per month")
    notes: Optional[str] = Field(None, description="Notes")


class TopUpAppBudgetRequest(BaseModel):
    """Top up application budget (company money)."""
    usd_amount: float = Field(..., gt=0)
    notes: Optional[str] = Field(None)


# ============================================================================
# Helper Functions
# ============================================================================

def _get_control_plane_manager(ctx):
    """
    Get or create ControlPlaneManager instance.
    Matches pattern from opex.py for consistency.
    """
    from kdcube_ai_app.apps.chat.sdk.infra.control_plane.manager import ControlPlaneManager

    # Try to reuse cached instance
    mgr = getattr(ctx.state, "control_plane_manager", None)
    if mgr:
        return mgr

    # Get PostgreSQL pool and Redis from middleware
    pg_pool = getattr(ctx.state, "pg_pool", None)
    redis = getattr(ctx.state.middleware, "redis", None)

    if not pg_pool or not redis:
        raise HTTPException(
            status_code=503,
            detail="Control plane dependencies not initialized"
        )

    mgr = ControlPlaneManager(pg_pool=pg_pool, redis=redis)

    # Cache on router state
    ctx.state.control_plane_manager = mgr
    return mgr


# ============================================================================
# TIER BALANCE (Tier Overrides + Lifetime Budget)
# ============================================================================

@router.post("/tier-balance/grant-trial", status_code=201)
async def grant_trial_bonus(
        payload: GrantTrialRequest,
        session: UserSession = Depends(auth_without_pressure())
):
    """
    Grant 7-day trial with temporary tier OVERRIDE.

    **How it works:**
    1. User's base tier: Free (10 req/day)
    2. You grant trial: 100 req/day for 7 days
    3. During days 1-7: User gets exactly 100 req/day (tier is OVERRIDDEN)
    4. Day 8+: User reverts to base tier (10 req/day)

    **Use Cases:**
    - New user registration bonus
    - Marketing campaigns
    - User retention
    """
    try:
        mgr = _get_control_plane_manager(router)
        settings = get_settings()

        expires_at = datetime.now(timezone.utc) + timedelta(days=payload.days)

        tier_balance = await mgr.update_user_tier_budget(
            tenant=settings.TENANT,
            project=settings.PROJECT,
            user_id=payload.user_id,
            requests_per_day=payload.requests_per_day,
            requests_per_month=payload.requests_per_month,
            tokens_per_day=payload.tokens_per_day,
            tokens_per_month=payload.tokens_per_month,
            max_concurrent=payload.max_concurrent,
            expires_at=expires_at,
            purchase_notes=payload.notes or f"{payload.days}-day trial",
        )

        logger.info(
            f"[grant_trial] {payload.user_id}: {payload.days} days, "
            f"{payload.requests_per_day} req/day (OVERRIDE) by {session.username}"
        )

        return {
            "status": "ok",
            "message": f"Trial granted to {payload.user_id}",
            "tier_balance": {
                "user_id": tier_balance.user_id,
                "requests_per_day": tier_balance.requests_per_day,
                "tokens_per_day": tier_balance.tokens_per_day,
                "expires_at": tier_balance.expires_at.isoformat() if tier_balance.expires_at else None,
            }
        }
    except Exception as e:
        logger.exception(f"[grant_trial] Failed for {payload.user_id}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tier-balance/update", status_code=201)
async def update_tier_budget(
        payload: UpdateTierBudgetRequest,
        session: UserSession = Depends(auth_without_pressure())
):
    """
    Update user's tier budget (supports PARTIAL updates).

    **How it works:**
    - Sets specific limits for a period
    - OVERRIDES base tier (does not add)
    - Resets daily/monthly like tier quotas
    - Expires after X days

    **Partial Updates:**
    - Only updates fields you provide
    - Other fields stay unchanged (COALESCE in SQL)

    **Use Cases:**
    - Promotional campaigns
    - Beta tester credits
    - Apology/compensation
    - Extending existing trials
    """
    try:
        mgr = _get_control_plane_manager(router)
        settings = get_settings()

        expires_at = None
        if payload.expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=payload.expires_in_days)

        tier_balance = await mgr.update_user_tier_budget(
            tenant=settings.TENANT,
            project=settings.PROJECT,
            user_id=payload.user_id,
            requests_per_day=payload.requests_per_day,
            requests_per_month=payload.requests_per_month,
            tokens_per_day=payload.tokens_per_day,
            tokens_per_month=payload.tokens_per_month,
            max_concurrent=payload.max_concurrent,
            expires_at=expires_at,
            purchase_notes=payload.notes or "Admin tier budget update",
        )

        logger.info(f"[update_tier_budget] {payload.user_id} by {session.username}")

        return {
            "status": "ok",
            "message": f"Tier budget updated for {payload.user_id}",
            "tier_balance": {
                "user_id": tier_balance.user_id,
                "requests_per_day": tier_balance.requests_per_day,
                "tokens_per_day": tier_balance.tokens_per_day,
                "expires_at": tier_balance.expires_at.isoformat() if tier_balance.expires_at else None,
            }
        }
    except Exception as e:
        logger.exception(f"[update_tier_budget] Failed for {payload.user_id}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tier-balance/user/{user_id}")
async def get_user_tier_balance(
        user_id: str,
        include_expired: bool = Query(False),
        session: UserSession = Depends(auth_without_pressure())
):
    """
    Get user's tier balance (tier override + lifetime budget).

    Shows currently active tier override with expiration date.
    Also shows lifetime budget if user has purchased credits.
    """
    try:
        mgr = _get_control_plane_manager(router)
        settings = get_settings()

        tier_balance = await mgr.get_user_tier_balance(
            tenant=settings.TENANT,
            project=settings.PROJECT,
            user_id=user_id,
        )

        if not tier_balance:
            return {
                "status": "ok",
                "user_id": user_id,
                "has_tier_override": False,
                "has_lifetime_budget": False,
                "message": "User has no tier balance"
            }

        # Check if expired
        if not include_expired and tier_balance.is_expired():
            return {
                "status": "ok",
                "user_id": user_id,
                "has_tier_override": False,
                "has_lifetime_budget": tier_balance.has_lifetime_budget(),
                "message": "Tier override expired"
            }

        return {
            "status": "ok",
            "user_id": user_id,
            "has_tier_override": tier_balance.has_tier_override(),
            "has_lifetime_budget": tier_balance.has_lifetime_budget(),
            "tier_override": {
                "requests_per_day": tier_balance.requests_per_day,
                "requests_per_month": tier_balance.requests_per_month,
                "tokens_per_day": tier_balance.tokens_per_day,
                "tokens_per_month": tier_balance.tokens_per_month,
                "max_concurrent": tier_balance.max_concurrent,
                "expires_at": tier_balance.expires_at.isoformat() if tier_balance.expires_at else None,
                "notes": tier_balance.purchase_notes,
                "is_expired": tier_balance.is_expired(),
            } if tier_balance.has_tier_override() else None,
            "lifetime_budget": {
                "tokens_purchased": tier_balance.lifetime_tokens_purchased,
                "tokens_consumed": tier_balance.lifetime_tokens_consumed,
                "tokens_remaining": (tier_balance.lifetime_tokens_purchased or 0) - (tier_balance.lifetime_tokens_consumed or 0),
                "purchase_amount_usd": float(tier_balance.purchase_amount_usd) if tier_balance.purchase_amount_usd else None,
            } if tier_balance.has_lifetime_budget() else None,
        }
    except Exception as e:
        logger.exception(f"[get_user_tier_balance] Failed for {user_id}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tier-balance/user/{user_id}")
async def deactivate_tier_balance(
        user_id: str,
        session: UserSession = Depends(auth_without_pressure())
):
    """
    Deactivate (soft delete) user's tier balance.

    **WARNING:** This clears BOTH tier override AND lifetime budget!

    **Use Cases:**
    1. **Refund Processing**: Remove all credits after issuing refund
    2. **Abuse Prevention**: Revoke all credits from fraudulent accounts
    3. **Policy Violation**: Remove all credits for TOS violations
    """
    try:
        mgr = _get_control_plane_manager(router)
        settings = get_settings()

        await mgr.deactivate_tier_balance(
            tenant=settings.TENANT,
            project=settings.PROJECT,
            user_id=user_id,
        )

        logger.info(
            f"[deactivate_tier_balance] {settings.TENANT}/{settings.PROJECT}/{user_id}: "
            f"deactivated by {session.username or session.user_id}"
        )

        return {
            "status": "ok",
            "message": f"Tier balance deactivated for user {user_id}",
        }
    except Exception as e:
        logger.exception(f"[deactivate_tier_balance] Failed for {user_id}")
        raise HTTPException(status_code=500, detail=f"Failed to deactivate: {str(e)}")


@router.post("/tier-balance/add-lifetime-credits", status_code=201)
async def add_lifetime_credits(
        payload: AddLifetimeCreditsRequest,
        session: UserSession = Depends(auth_without_pressure())
):
    """
    Add purchased credits in USD (converted to lifetime tokens).

    User's purchased credits - separate from tier budget.
    Balance depletes on use, does NOT reset.
    """
    try:
        mgr = _get_control_plane_manager(router)
        settings = get_settings()

        # Use ControlPlaneManager method (which delegates to TierBalanceManager)
        tier_balance = await mgr.add_user_credits_usd(
            tenant=settings.TENANT,
            project=settings.PROJECT,
            user_id=payload.user_id,
            usd_amount=payload.usd_amount,
            ref_provider=payload.ref_provider,
            ref_model=payload.ref_model,
            purchase_id=payload.purchase_id,
            notes=payload.notes,
        )

        # Get current balance
        balance = await mgr.tier_balance_mgr.get_lifetime_balance(
            tenant=settings.TENANT,
            project=settings.PROJECT,
            user_id=payload.user_id,
        )

        # Convert to USD for display
        from kdcube_ai_app.infra.accounting.usage import _find_llm_price
        pr = _find_llm_price(payload.ref_provider, payload.ref_model)
        p_ref_out = float(pr["output_tokens_1M"]) / 1_000_000
        balance_usd = balance * p_ref_out if balance else 0

        logger.info(f"[add_lifetime_credits] {payload.user_id}: +${payload.usd_amount} by {session.username}")

        return {
            "success": True,
            "user_id": payload.user_id,
            "usd_amount": payload.usd_amount,
            "tokens_added": tier_balance.lifetime_tokens_purchased,
            "new_balance_tokens": balance,
            "new_balance_usd": round(balance_usd, 2),
            "reference_model": f"{payload.ref_provider}/{payload.ref_model}",
        }

    except Exception as e:
        logger.error(f"Failed to add lifetime credits for {payload.user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tier-balance/lifetime-balance/{user_id}")
async def get_lifetime_balance(
        user_id: str,
        session: UserSession = Depends(auth_without_pressure())
):
    """
    Get user's lifetime purchased balance (tokens + USD equivalent).
    """
    try:
        mgr = _get_control_plane_manager(router)
        settings = get_settings()

        # Get lifetime balance from TierBalanceManager
        balance_tokens = await mgr.tier_balance_mgr.get_lifetime_balance(
            tenant=settings.TENANT,
            project=settings.PROJECT,
            user_id=user_id,
        )

        if balance_tokens is None:
            return {
                "user_id": user_id,
                "has_purchased_credits": False,
                "balance_tokens": 0,
                "balance_usd": 0,
                "message": "User has no purchased credits"
            }

        # Convert to USD
        from kdcube_ai_app.infra.accounting.usage import _find_llm_price
        pr = _find_llm_price("anthropic", "claude-sonnet-4-5-20250929")
        p_ref_out = float(pr["output_tokens_1M"]) / 1_000_000
        balance_usd = balance_tokens * p_ref_out

        return {
            "user_id": user_id,
            "has_purchased_credits": True,
            "balance_tokens": balance_tokens,
            "balance_usd": round(balance_usd, 2),
            "minimum_required_tokens": 50_000,
            "can_use_budget": balance_tokens >= 50_000,
        }

    except Exception as e:
        logger.error(f"Failed to get balance for {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Stripe Webhook - Public Endpoint (No Auth)
# ============================================================================

@router.post("/webhooks/stripe")
async def stripe_webhook(
        request: Request,
        stripe_signature: Optional[str] = Header(None, alias="Stripe-Signature")
):
    """
    Stripe webhook endpoint for automated credit purchases.

    **Webhook Events Handled:**
    - `payment_intent.succeeded` - Grant credits after successful payment

    **Security:** Verifies Stripe signature using HMAC SHA256.
    **Public endpoint** - No authentication required (Stripe signature verification instead).
    """
    try:
        # Get webhook secret from environment
        webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET")
        if not webhook_secret:
            logger.warning("[stripe_webhook] STRIPE_WEBHOOK_SECRET not set, skipping verification")

        # Read raw body
        body = await request.body()

        # Verify signature if secret is set
        if webhook_secret and stripe_signature:
            try:
                # Simple HMAC verification (Stripe format: t=timestamp,v1=signature)
                expected_sig = hmac.new(
                    webhook_secret.encode(),
                    body,
                    hashlib.sha256
                ).hexdigest()

                # Parse Stripe signature
                sig_parts = dict(part.split("=") for part in stripe_signature.split(","))
                provided_sig = sig_parts.get("v1", "")

                if not hmac.compare_digest(expected_sig, provided_sig):
                    logger.error("[stripe_webhook] Signature verification failed")
                    raise HTTPException(status_code=401, detail="Invalid signature")
            except Exception as e:
                logger.error(f"[stripe_webhook] Signature verification error: {e}")
                raise HTTPException(status_code=401, detail="Invalid signature")

        # Parse event
        event = json.loads(body.decode())

        # Handle payment success
        if event["type"] == "payment_intent.succeeded":
            payment = event["data"]["object"]
            metadata = payment.get("metadata", {})

            # Extract metadata
            user_id = metadata.get("user_id")
            tenant = metadata.get("tenant")
            project = metadata.get("project")
            package = metadata.get("package")

            if not user_id:
                logger.warning("[stripe_webhook] Missing user_id in payment metadata")
                return {"status": "ok", "message": "Missing user_id"}

            # Get control plane manager
            mgr = _get_control_plane_manager(router)
            settings = get_settings()

            # Use metadata or fallback to settings
            tenant = tenant or settings.TENANT
            project = project or settings.PROJECT

            usd = payment["amount"] / 100  # Cents to dollars

            # Add lifetime credits
            await mgr.add_user_credits_usd(
                tenant=tenant,
                project=project,
                user_id=user_id,
                usd_amount=usd,
                purchase_id=payment["id"],
                notes=f"Stripe payment - {package or 'custom'} package",
            )

            logger.info(
                f"[stripe_webhook] Credits granted to {user_id}: "
                f"${usd}, payment_id={payment['id']}"
            )

            return {
                "status": "ok",
                "message": f"Credits granted to user {user_id}",
            }

        # Other event types
        return {
            "status": "ok",
            "message": f"Event {event['type']} received but not processed",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[stripe_webhook] Failed to process webhook")
        raise HTTPException(status_code=500, detail=f"Webhook processing failed: {str(e)}")


# ============================================================================
# Policy Management Endpoints - Admin Only
# ============================================================================

@router.get("/policies/quota")
async def list_quota_policies(
        tenant: Optional[str] = Query(None),
        project: Optional[str] = Query(None),
        limit: int = Query(100),
        session: UserSession = Depends(auth_without_pressure())
):
    """
    List user quota policies (base policies by user type).

    Shows configured policies for different user tiers (free, paid, premium, etc.).
    """
    try:
        mgr = _get_control_plane_manager(router)
        settings = get_settings()

        policies = await mgr.list_quota_policies(
            tenant=tenant or settings.TENANT,
            project=project or settings.PROJECT,
            limit=limit,
        )

        result = []
        for p in policies:
            result.append({
                "tenant": p.tenant,
                "project": p.project,
                "user_type": p.user_type,
                "max_concurrent": p.max_concurrent,
                "requests_per_day": p.requests_per_day,
                "requests_per_month": p.requests_per_month,
                "tokens_per_day": p.tokens_per_day,
                "tokens_per_month": p.tokens_per_month,
                "notes": p.notes,
            })

        return {
            "status": "ok",
            "count": len(result),
            "policies": result,
        }
    except Exception as e:
        logger.exception("[list_quota_policies] Failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/policies/quota", status_code=201)
async def set_quota_policy(
        payload: SetQuotaPolicyRequest,
        session: UserSession = Depends(auth_without_pressure())
):
    """
    Set the quota policy for a user type.

    Defines base rate limits for different user tiers.
    Supports partial updates via COALESCE.
    """
    try:
        mgr = _get_control_plane_manager(router)
        settings = get_settings()

        policy = await mgr.set_tenant_project_user_quota_policy(
            tenant=settings.TENANT,
            project=settings.PROJECT,
            user_type=payload.user_type,
            max_concurrent=payload.max_concurrent,
            requests_per_day=payload.requests_per_day,
            requests_per_month=payload.requests_per_month,
            total_requests=payload.total_requests,
            tokens_per_hour=payload.tokens_per_hour,
            tokens_per_day=payload.tokens_per_day,
            tokens_per_month=payload.tokens_per_month,
            created_by=session.username or session.user_id,
            notes=payload.notes,
        )

        logger.info(
            f"[set_quota_policy] {settings.TENANT}/{settings.PROJECT}/{payload.user_type}: "
            f"policy updated by {session.username or session.user_id}"
        )

        return {
            "status": "ok",
            "message": f"Quota policy set for {payload.user_type}",
            "policy": {
                "user_type": policy.user_type,
                "requests_per_day": policy.requests_per_day,
                "tokens_per_day": policy.tokens_per_day,
            }
        }
    except Exception as e:
        logger.exception(f"[set_quota_policy] Failed for {payload.user_type}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/policies/budget")
async def list_budget_policies(
        tenant: Optional[str] = Query(None),
        project: Optional[str] = Query(None),
        limit: int = Query(100),
        session: UserSession = Depends(auth_without_pressure())
):
    """
    List application budget policies (spending limits per provider).

    Shows configured spending limits for AI providers (Anthropic, OpenAI, etc.).

    **Admin only.** Requires authentication.
    """
    try:
        mgr = _get_control_plane_manager(router)
        settings = get_settings()

        policies = await mgr.list_tenant_project_budget_policies(
            tenant=tenant or settings.TENANT,
            project=project or settings.PROJECT,
            limit=limit,
        )

        result = []
        for p in policies:
            result.append({
                "tenant": p.tenant,
                "project": p.project,
                "provider": p.provider,
                "usd_per_hour": float(p.usd_per_hour) if p.usd_per_hour else None,
                "usd_per_day": float(p.usd_per_day) if p.usd_per_day else None,
                "usd_per_month": float(p.usd_per_month) if p.usd_per_month else None,
                "notes": p.notes,
            })

        return {
            "status": "ok",
            "count": len(result),
            "policies": result,
        }
    except Exception as e:
        logger.exception("[list_budget_policies] Failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/policies/budget", status_code=201)
async def set_budget_policy(
        payload: SetBudgetPolicyRequest,
        session: UserSession = Depends(auth_without_pressure())
):
    """
    Set budget policy for a provider.

    Configures spending limits to prevent runaway costs.
    Supports partial updates via COALESCE.
    """
    try:
        mgr = _get_control_plane_manager(router)
        settings = get_settings()

        policy = await mgr.set_tenant_project_budget_policy(
            tenant=settings.TENANT,
            project=settings.PROJECT,
            provider=payload.provider,
            usd_per_hour=payload.usd_per_hour,
            usd_per_day=payload.usd_per_day,
            usd_per_month=payload.usd_per_month,
            created_by=session.username or session.user_id,
            notes=payload.notes,
        )

        logger.info(
            f"[set_budget_policy] {settings.TENANT}/{settings.PROJECT}/{payload.provider}: "
            f"policy updated by {session.username or session.user_id}"
        )

        return {
            "status": "ok",
            "message": f"Budget policy set for {payload.provider}",
            "policy": {
                "provider": policy.provider,
                "usd_per_day": float(policy.usd_per_day) if policy.usd_per_day else None,
            }
        }
    except Exception as e:
        logger.exception(f"[set_budget_policy] Failed for {payload.provider}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# User Quota Breakdown & Utilities
# ============================================================================

@router.get("/users/{user_id}/quota-breakdown")
async def get_user_quota_breakdown(
        user_id: str,
        user_type: str = Query(..., description="User type (free, paid, premium, etc.)"),
        session: UserSession = Depends(auth_without_pressure())
):
    """
    Get the detailed quota breakdown for a user.

    Shows base tier, tier override, effective policy, and current usage.
    """
    try:
        mgr = _get_control_plane_manager(router)
        settings = get_settings()

        tenant = settings.TENANT
        project = settings.PROJECT

        # Get base policy
        base_policy = await mgr.get_user_quota_policy(
            tenant=tenant,
            project=project,
            user_type=user_type,
        )

        if not base_policy:
            raise HTTPException(
                status_code=404,
                detail=f"No policy found for user_type={user_type}"
            )

        # Get tier balance
        tier_balance = await mgr.get_user_tier_balance(
            tenant=tenant,
            project=project,
            user_id=user_id,
        )

        # Build rate limiter to read counters
        redis = getattr(router.state.middleware, "redis", None)
        if not redis:
            raise HTTPException(status_code=503, detail="Redis not available")

        rl = RateLimiter(redis)

        # Get usage breakdown (handles "*" for all bundles)
        usage_breakdown = await rl.breakdown(
            tenant=tenant,
            project=project,
            user_id=user_id,
            bundle_ids=["*"],
        )

        # Use totals for display
        totals = usage_breakdown["totals"]
        req_day = totals["requests_today"]
        req_month = totals["requests_this_month"]
        req_total = totals["requests_total"]
        tok_day = totals["tokens_today"]
        tok_month = totals["tokens_this_month"]

        # Merge policies (OVERRIDE semantics)
        from kdcube_ai_app.apps.chat.sdk.infra.rate_limit.limiter import _merge_policy_with_tier_balance

        # Convert tier_balance to QuotaReplenishment-like object for compatibility
        if tier_balance:
            effective_policy = _merge_policy_with_tier_balance(base_policy, tier_balance)
        else:
            effective_policy = base_policy

        # Calculate remaining
        def calc_remaining(limit, used):
            if limit is None:
                return None
            return max(limit - used, 0)

        remaining_req_day = calc_remaining(effective_policy.requests_per_day, req_day)
        remaining_req_month = calc_remaining(effective_policy.requests_per_month, req_month)
        remaining_tok_day = calc_remaining(effective_policy.tokens_per_day, tok_day)
        remaining_tok_month = calc_remaining(effective_policy.tokens_per_month, tok_month)

        # Calculate percentage used
        percentage_used = None
        if effective_policy.requests_per_day and effective_policy.requests_per_day > 0:
            percentage_used = round((req_day / effective_policy.requests_per_day) * 100, 1)

        return {
            "status": "ok",
            "user_id": user_id,
            "user_type": user_type,
            "bundle_breakdown": usage_breakdown.get("bundles"),
            "base_policy": {
                "max_concurrent": base_policy.max_concurrent,
                "requests_per_day": base_policy.requests_per_day,
                "requests_per_month": base_policy.requests_per_month,
                "tokens_per_day": base_policy.tokens_per_day,
                "tokens_per_month": base_policy.tokens_per_month,
            },
            "tier_override": {
                "has_override": tier_balance is not None and tier_balance.has_tier_override(),
                "max_concurrent": tier_balance.max_concurrent if tier_balance else None,
                "requests_per_day": tier_balance.requests_per_day if tier_balance else None,
                "requests_per_month": tier_balance.requests_per_month if tier_balance else None,
                "tokens_per_day": tier_balance.tokens_per_day if tier_balance else None,
                "tokens_per_month": tier_balance.tokens_per_month if tier_balance else None,
                "expires_at": tier_balance.expires_at.isoformat() if tier_balance and tier_balance.expires_at else None,
                "purchase_notes": tier_balance.purchase_notes if tier_balance else None,
            } if tier_balance else None,
            "effective_policy": {
                "max_concurrent": effective_policy.max_concurrent,
                "requests_per_day": effective_policy.requests_per_day,
                "requests_per_month": effective_policy.requests_per_month,
                "tokens_per_day": effective_policy.tokens_per_day,
                "tokens_per_month": effective_policy.tokens_per_month,
            },
            "current_usage": {
                "requests_today": req_day,
                "requests_this_month": req_month,
                "requests_total": req_total,
                "tokens_today": tok_day,
                "tokens_this_month": tok_month,
                "concurrent": 0,  # Can't aggregate concurrent across bundles
            },
            "remaining": {
                "requests_today": remaining_req_day,
                "requests_this_month": remaining_req_month,
                "tokens_today": remaining_tok_day,
                "tokens_this_month": remaining_tok_month,
                "percentage_used": percentage_used,
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[get_user_quota_breakdown] Failed for user {user_id}")
        raise HTTPException(status_code=500, detail=f"Failed to get quota breakdown: {str(e)}")


# ============================================================================
# App Budget Operations
# ============================================================================

@router.post("/app-budget/topup")
async def topup_app_budget(
        payload: TopUpAppBudgetRequest,
        session: UserSession = Depends(auth_without_pressure())
):
    """
    Top up application budget (company money).

    This is the TENANT/PROJECT budget used to pay for tier-funded requests.
    """
    try:
        settings = get_settings()

        pg_pool = getattr(router.state, "pg_pool", None)
        redis = getattr(router.state.middleware, "redis", None)

        limiter = ProjectBudgetLimiter(redis, pg_pool, tenant=settings.TENANT, project=settings.PROJECT)
        result = await limiter.topup_app_budget(
            usd_amount=payload.usd_amount,
            notes=payload.notes or f"Top-up by {session.username}",
        )

        logger.info(f"[topup_app_budget] {settings.TENANT}/{settings.PROJECT}: +${payload.usd_amount} by {session.username}")

        return {
            "status": "ok",
            "new_balance_usd": result["new_balance_usd"],
            "lifetime_added_usd": result["lifetime_added_usd"],
        }
    except Exception as e:
        logger.exception("[topup_app_budget] Failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/app-budget/balance")
async def get_app_budget_balance(
        session: UserSession = Depends(auth_without_pressure())
):
    """Get current app budget balance."""
    try:
        pg_pool = getattr(router.state, "pg_pool", None)
        redis = getattr(router.state.middleware, "redis", None)

        settings = get_settings()
        limiter = ProjectBudgetLimiter(redis, pg_pool, tenant=settings.TENANT, project=settings.PROJECT)

        balance = await limiter.get_app_budget_balance()
        spending = await limiter.get_spending_by_bundle()

        return {
            "status": "ok",
            "balance": balance,
            "current_month_spending": spending["totals"],
            "by_bundle": spending["bundles"],
        }
    except Exception as e:
        logger.exception("[get_app_budget_balance] Failed")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Health & Admin Utilities
# ============================================================================

@router.get("/health")
async def health_check(
        session: UserSession = Depends(auth_without_pressure())
):
    """Health check for control plane."""
    try:
        mgr = _get_control_plane_manager(router)
        return {
            "status": "ok",
            "service": "control_plane",
            "postgresql": "ready" if mgr._pg_pool else "not initialized",
            "redis": "ready" if mgr._redis else "not initialized",
        }
    except Exception as e:
        logger.exception("[health_check] Failed")
        return {
            "status": "error",
            "service": "control_plane",
            "error": str(e)
        }


@router.post("/clear-cache")
async def admin_clear_cache(
        request: Request,
        session: UserSession = Depends(auth_without_pressure())
):
    """
    Clear cached control plane manager (forces recreation with fresh connections).

    Useful after configuration changes or to troubleshoot caching issues.
    """
    try:
        if hasattr(request.app.state, "control_plane_manager"):
            delattr(request.app.state, "control_plane_manager")

        return {
            "status": "ok",
            "message": "Control plane cache cleared"
        }
    except Exception as e:
        logger.exception("[admin_clear_cache] Failed")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")