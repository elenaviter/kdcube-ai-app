-- =========================================
-- deploy-kdcube-control-plane.sql
-- Control Plane Schema
-- =========================================

CREATE SCHEMA IF NOT EXISTS kdcube_control_plane;

-- =========================================
-- User Tier Balance Table
-- Stores user's current effective tier + lifetime budget
--
-- Two types of data:
-- 1. Tier override (admin grants, no $) - temporary tier upgrade
-- 2. User budget ($ purchase) - lifetime token balance
-- =========================================

CREATE TABLE IF NOT EXISTS kdcube_control_plane.user_tier_balance (
    -- Primary identification
    tenant VARCHAR(255) NOT NULL,
    project VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,

    -- Tier Override Limits (temporary tier upgrade)
    -- If ANY of these is set, this becomes the user's tier (overrides base tier)
    max_concurrent INTEGER DEFAULT NULL,
    requests_per_day INTEGER DEFAULT NULL,
    requests_per_month INTEGER DEFAULT NULL,
    total_requests INTEGER DEFAULT NULL,
    tokens_per_hour BIGINT DEFAULT NULL,
    tokens_per_day BIGINT DEFAULT NULL,
    tokens_per_month BIGINT DEFAULT NULL,

    -- Expiry and tracking
    expires_at TIMESTAMPTZ DEFAULT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Purchase tracking (for tier overrides granted via payment)
    purchase_id VARCHAR(255) DEFAULT NULL,
    purchase_amount_usd NUMERIC(10, 2) DEFAULT NULL,
    purchase_notes TEXT DEFAULT NULL,

    -- Lifetime token budget (SEPARATE from tier - for pay-as-you-go)
    lifetime_tokens_purchased BIGINT DEFAULT NULL,
    lifetime_tokens_consumed BIGINT DEFAULT NULL,

    -- Status
    active BOOLEAN NOT NULL DEFAULT TRUE,

    -- Primary key (one balance per user, GLOBAL)
    PRIMARY KEY (tenant, project, user_id)
);

COMMENT ON TABLE kdcube_control_plane.user_tier_balance IS
    'User tier balance - stores both tier overrides and lifetime token budget.

     Tier Override: Admin grants temporary tier upgrade (expires after N days)
       Example: Grant 100 req/day for 7 days (trial)

     Lifetime Budget: User purchases tokens that deplete on use
       Example: Buy $10 → 666,667 tokens (never expires, just depletes)';

COMMENT ON COLUMN kdcube_control_plane.user_tier_balance.requests_per_day IS
    'Tier override - if set, replaces base tier limit (NULL = use base tier)';

COMMENT ON COLUMN kdcube_control_plane.user_tier_balance.lifetime_tokens_purchased IS
    'Total tokens purchased (separate from tier). Decrements lifetime_tokens_consumed on use.';

-- Index for lifetime balance lookups
CREATE INDEX IF NOT EXISTS idx_cp_utb_lifetime_balance
    ON kdcube_control_plane.user_tier_balance(tenant, project, user_id)
    WHERE purchase_amount_usd IS NOT NULL AND active = TRUE;

CREATE INDEX IF NOT EXISTS idx_cp_utb_lookup
    ON kdcube_control_plane.user_tier_balance(tenant, project, user_id)
    WHERE active = TRUE;

CREATE INDEX IF NOT EXISTS idx_cp_utb_expires
    ON kdcube_control_plane.user_tier_balance(expires_at)
    WHERE active = TRUE AND expires_at IS NOT NULL;

-- =========================================
-- User Quota Policies Table
-- Defines base tier limits by user_type (free, paid, premium)
-- GLOBAL per tenant/project
-- =========================================

CREATE TABLE IF NOT EXISTS kdcube_control_plane.user_quota_policies (
    -- Primary identification
    tenant VARCHAR(255) NOT NULL,
    project VARCHAR(255) NOT NULL,
    user_type VARCHAR(255) NOT NULL,  -- 'anonymous', 'registered' ('free'), 'payed' ('paid'), 'privileged' ('premium' or 'admin')

    -- Quota limits (NULL = unlimited)
    max_concurrent INTEGER DEFAULT NULL,
    requests_per_day INTEGER DEFAULT NULL,
    requests_per_month INTEGER DEFAULT NULL,
    total_requests INTEGER DEFAULT NULL,
    tokens_per_hour BIGINT DEFAULT NULL,
    tokens_per_day BIGINT DEFAULT NULL,
    tokens_per_month BIGINT DEFAULT NULL,

    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(255) DEFAULT NULL,
    notes TEXT DEFAULT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,

    PRIMARY KEY (tenant, project, user_type)
);

COMMENT ON TABLE kdcube_control_plane.user_quota_policies IS
    'Base tier policies - defines limits for user types (free, paid, premium).
     These are the DEFAULT limits before any tier overrides are applied.';

CREATE INDEX IF NOT EXISTS idx_cp_uqp_lookup
    ON kdcube_control_plane.user_quota_policies(tenant, project, user_type)
    WHERE active = TRUE;

-- =========================================
-- Application Budget Policies Table
-- Per bundle+provider for breakdown (aggregated when checking)
-- =========================================

CREATE TABLE IF NOT EXISTS kdcube_control_plane.application_budget_policies (
    tenant VARCHAR(255) NOT NULL,
    project VARCHAR(255) NOT NULL,
    provider VARCHAR(255) NOT NULL,

    -- Budget limits in USD
    usd_per_hour NUMERIC(10, 2) DEFAULT NULL,
    usd_per_day NUMERIC(10, 2) DEFAULT NULL,
    usd_per_month NUMERIC(10, 2) DEFAULT NULL,

    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(255) DEFAULT NULL,
    notes TEXT DEFAULT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,

    PRIMARY KEY (tenant, project, provider)
);

COMMENT ON TABLE kdcube_control_plane.application_budget_policies IS
    'App budget limits per bundle+provider. Aggregated across bundles when checking total spend.';

CREATE INDEX IF NOT EXISTS idx_cp_abp_lookup
    ON kdcube_control_plane.application_budget_policies(tenant, project, provider)
    WHERE active = TRUE;

-- =========================================
-- Update Triggers
-- =========================================

CREATE OR REPLACE FUNCTION kdcube_control_plane.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
    DROP TRIGGER IF EXISTS trg_cp_utb_updated_at ON kdcube_control_plane.user_tier_balance;
    CREATE TRIGGER trg_cp_utb_updated_at
        BEFORE UPDATE ON kdcube_control_plane.user_tier_balance
        FOR EACH ROW EXECUTE FUNCTION kdcube_control_plane.update_updated_at();

    DROP TRIGGER IF EXISTS trg_cp_uqp_updated_at ON kdcube_control_plane.user_quota_policies;
    CREATE TRIGGER trg_cp_uqp_updated_at
        BEFORE UPDATE ON kdcube_control_plane.user_quota_policies
        FOR EACH ROW EXECUTE FUNCTION kdcube_control_plane.update_updated_at();

    DROP TRIGGER IF EXISTS trg_cp_abp_updated_at ON kdcube_control_plane.application_budget_policies;
    CREATE TRIGGER trg_cp_abp_updated_at
        BEFORE UPDATE ON kdcube_control_plane.application_budget_policies
        FOR EACH ROW EXECUTE FUNCTION kdcube_control_plane.update_updated_at();
END $$;


-- =========================================
-- Tenant/Project Budget Balance
-- =========================================

CREATE TABLE IF NOT EXISTS kdcube_control_plane.tenant_project_budget (
    tenant VARCHAR(255) NOT NULL,
    project VARCHAR(255) NOT NULL,

    -- Budget balance in USD cents (to avoid float precision issues)
    balance_cents BIGINT NOT NULL DEFAULT 0,

    -- Lifetime statistics
    lifetime_added_cents BIGINT NOT NULL DEFAULT 0,
    lifetime_spent_cents BIGINT NOT NULL DEFAULT 0,

    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    notes TEXT DEFAULT NULL,

    PRIMARY KEY (tenant, project),

    CONSTRAINT balance_non_negative CHECK (balance_cents >= 0)
);

CREATE INDEX IF NOT EXISTS idx_cp_budget_balance
    ON kdcube_control_plane.tenant_project_budget(tenant, project, balance_cents);

-- Trigger for updated_at
DROP TRIGGER IF EXISTS trg_cp_budget_updated_at ON kdcube_control_plane.tenant_project_budget;
CREATE TRIGGER trg_cp_budget_updated_at
    BEFORE UPDATE ON kdcube_control_plane.tenant_project_budget
    FOR EACH ROW EXECUTE FUNCTION kdcube_control_plane.update_updated_at();