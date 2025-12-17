-- =========================================
-- deploy-kdcube-control-plane.sql
-- Control Plane Schema - Global configuration, user quotas, and policies
-- NOT tenant/project-specific
-- =========================================

CREATE SCHEMA IF NOT EXISTS kdcube_control_plane;

-- =========================================
-- User Quota Replenishment Table
-- Stores additional quotas purchased or granted to users ABOVE their base policy
-- =========================================

CREATE TABLE IF NOT EXISTS kdcube_control_plane.user_quota_replenishment (
    -- Primary identification
    tenant VARCHAR(255) NOT NULL,
    project VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    bundle_id VARCHAR(255) NOT NULL,  -- '*' for all bundles

    -- Additional quotas (add to base policy)
    -- NULL = no additional quota for that dimension
    additional_requests_per_day INTEGER DEFAULT NULL,
    additional_requests_per_month INTEGER DEFAULT NULL,
    additional_total_requests INTEGER DEFAULT NULL,
    additional_tokens_per_hour BIGINT DEFAULT NULL,
    additional_tokens_per_day BIGINT DEFAULT NULL,
    additional_tokens_per_month BIGINT DEFAULT NULL,
    additional_max_concurrent INTEGER DEFAULT NULL,

    -- Expiry and tracking
    expires_at TIMESTAMPTZ DEFAULT NULL,  -- NULL = never expires
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Purchase tracking (for billing integration)
    purchase_id VARCHAR(255) DEFAULT NULL,  -- Reference to purchase/transaction
    purchase_amount_usd NUMERIC(10, 2) DEFAULT NULL,
    purchase_notes TEXT DEFAULT NULL,

    -- Status
    active BOOLEAN NOT NULL DEFAULT TRUE,

    -- Composite primary key (one replenishment per user per bundle)
    PRIMARY KEY (tenant, project, user_id, bundle_id)
);

-- =========================================
-- User Quota Policies Table
-- Defines base quota policies by user type/role
-- =========================================

CREATE TABLE IF NOT EXISTS kdcube_control_plane.user_quota_policies (
    -- Primary identification
    tenant VARCHAR(255) NOT NULL,
    project VARCHAR(255) NOT NULL,
    user_type VARCHAR(255) NOT NULL,  -- 'free', 'paid', 'premium', etc.
    bundle_id VARCHAR(255) NOT NULL,  -- '*' for all bundles

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

    -- Status
    active BOOLEAN NOT NULL DEFAULT TRUE,

    PRIMARY KEY (tenant, project, user_type, bundle_id)
);

-- =========================================
-- Application Budget Policies Table
-- Defines spending limits per provider
-- =========================================

CREATE TABLE IF NOT EXISTS kdcube_control_plane.application_budget_policies (
    -- Primary identification
    tenant VARCHAR(255) NOT NULL,
    project VARCHAR(255) NOT NULL,
    bundle_id VARCHAR(255) NOT NULL,  -- Bundle identifier
    provider VARCHAR(255) NOT NULL,    -- 'anthropic', 'openai', 'brave', etc.

    -- Budget limits in USD (NULL = unlimited)
    usd_per_hour NUMERIC(10, 2) DEFAULT NULL,
    usd_per_day NUMERIC(10, 2) DEFAULT NULL,
    usd_per_month NUMERIC(10, 2) DEFAULT NULL,

    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(255) DEFAULT NULL,
    notes TEXT DEFAULT NULL,

    -- Status
    active BOOLEAN NOT NULL DEFAULT TRUE,

    PRIMARY KEY (tenant, project, bundle_id, provider)
);

-- =========================================
-- Indexes
-- =========================================

-- Replenishment indexes
CREATE INDEX IF NOT EXISTS idx_cp_uqr_lookup
    ON kdcube_control_plane.user_quota_replenishment(tenant, project, user_id, bundle_id)
    WHERE active = TRUE;

CREATE INDEX IF NOT EXISTS idx_cp_uqr_expires
    ON kdcube_control_plane.user_quota_replenishment(expires_at)
    WHERE active = TRUE AND expires_at IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_cp_uqr_user
    ON kdcube_control_plane.user_quota_replenishment(tenant, project, user_id)
    WHERE active = TRUE;

CREATE INDEX IF NOT EXISTS idx_cp_uqr_purchase
    ON kdcube_control_plane.user_quota_replenishment(purchase_id)
    WHERE purchase_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_cp_uqr_tenant
    ON kdcube_control_plane.user_quota_replenishment(tenant, project)
    WHERE active = TRUE;

-- Quota policy indexes
CREATE INDEX IF NOT EXISTS idx_cp_uqp_lookup
    ON kdcube_control_plane.user_quota_policies(tenant, project, user_type, bundle_id)
    WHERE active = TRUE;

CREATE INDEX IF NOT EXISTS idx_cp_uqp_tenant
    ON kdcube_control_plane.user_quota_policies(tenant, project)
    WHERE active = TRUE;

-- Budget policy indexes
CREATE INDEX IF NOT EXISTS idx_cp_abp_lookup
    ON kdcube_control_plane.application_budget_policies(tenant, project, bundle_id, provider)
    WHERE active = TRUE;

CREATE INDEX IF NOT EXISTS idx_cp_abp_tenant
    ON kdcube_control_plane.application_budget_policies(tenant, project)
    WHERE active = TRUE;

-- =========================================
-- Triggers
-- =========================================

-- Update trigger for replenishment updated_at
CREATE OR REPLACE FUNCTION kdcube_control_plane.update_uqr_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'trg_cp_uqr_updated_at'
          AND tgrelid = 'kdcube_control_plane.user_quota_replenishment'::regclass
    ) THEN
        CREATE TRIGGER trg_cp_uqr_updated_at
            BEFORE UPDATE ON kdcube_control_plane.user_quota_replenishment
            FOR EACH ROW
            EXECUTE FUNCTION kdcube_control_plane.update_uqr_updated_at();
    END IF;
END $$;

-- Update trigger for quota policies updated_at
CREATE OR REPLACE FUNCTION kdcube_control_plane.update_uqp_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'trg_cp_uqp_updated_at'
          AND tgrelid = 'kdcube_control_plane.user_quota_policies'::regclass
    ) THEN
        CREATE TRIGGER trg_cp_uqp_updated_at
            BEFORE UPDATE ON kdcube_control_plane.user_quota_policies
            FOR EACH ROW
            EXECUTE FUNCTION kdcube_control_plane.update_uqp_updated_at();
    END IF;
END $$;

-- Update trigger for budget policies updated_at
CREATE OR REPLACE FUNCTION kdcube_control_plane.update_abp_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'trg_cp_abp_updated_at'
          AND tgrelid = 'kdcube_control_plane.application_budget_policies'::regclass
    ) THEN
        CREATE TRIGGER trg_cp_abp_updated_at
            BEFORE UPDATE ON kdcube_control_plane.application_budget_policies
            FOR EACH ROW
            EXECUTE FUNCTION kdcube_control_plane.update_abp_updated_at();
    END IF;
END $$;

-- =========================================
-- Utility Functions
-- =========================================

-- Check if a replenishment is expired
CREATE OR REPLACE FUNCTION kdcube_control_plane.is_replenishment_expired(expiration_ts TIMESTAMPTZ)
    RETURNS BOOLEAN
AS $$
BEGIN
    RETURN expiration_ts IS NOT NULL AND expiration_ts <= NOW();
END;
$$ LANGUAGE plpgsql STABLE STRICT PARALLEL SAFE;

-- Clean up expired replenishments (soft delete by setting active=false)
CREATE OR REPLACE FUNCTION kdcube_control_plane.cleanup_expired_replenishments()
RETURNS INT
AS $$
DECLARE
    count INT := 0;
BEGIN
    UPDATE kdcube_control_plane.user_quota_replenishment
    SET active = FALSE, updated_at = NOW()
    WHERE active = TRUE
      AND kdcube_control_plane.is_replenishment_expired(expires_at);

    GET DIAGNOSTICS count = ROW_COUNT;
    RETURN count;
END;
$$ LANGUAGE plpgsql;

-- =========================================
-- Views
-- =========================================

-- Active (non-expired) replenishments
CREATE OR REPLACE VIEW kdcube_control_plane.active_replenishments AS
SELECT *
FROM kdcube_control_plane.user_quota_replenishment
WHERE active = TRUE
  AND (expires_at IS NULL OR expires_at > NOW());

-- Expired replenishments
CREATE OR REPLACE VIEW kdcube_control_plane.expired_replenishments AS
SELECT *
FROM kdcube_control_plane.user_quota_replenishment
WHERE active = TRUE
  AND expires_at IS NOT NULL
  AND expires_at <= NOW();

-- Active quota policies
CREATE OR REPLACE VIEW kdcube_control_plane.active_quota_policies AS
SELECT *
FROM kdcube_control_plane.user_quota_policies
WHERE active = TRUE;

-- Active budget policies
CREATE OR REPLACE VIEW kdcube_control_plane.active_budget_policies AS
SELECT *
FROM kdcube_control_plane.application_budget_policies
WHERE active = TRUE;

-- =========================================
-- Comments for documentation
-- =========================================

COMMENT ON SCHEMA kdcube_control_plane IS
    'Global control plane schema - not tenant/project-specific. Contains system-wide configuration, user quotas, and policies.';

COMMENT ON TABLE kdcube_control_plane.user_quota_replenishment IS
    'Stores additional quotas purchased or granted to users above their base policy';

COMMENT ON TABLE kdcube_control_plane.user_quota_policies IS
    'Defines base quota policies by user type/role (free, paid, premium, etc.)';

COMMENT ON TABLE kdcube_control_plane.application_budget_policies IS
    'Defines application spending limits per provider (anthropic, openai, etc.)';

COMMENT ON COLUMN kdcube_control_plane.user_quota_policies.bundle_id IS
    'Bundle ID this policy applies to. Use "*" for all bundles.';

COMMENT ON COLUMN kdcube_control_plane.application_budget_policies.provider IS
    'Provider identifier (anthropic, openai, brave, etc.)';