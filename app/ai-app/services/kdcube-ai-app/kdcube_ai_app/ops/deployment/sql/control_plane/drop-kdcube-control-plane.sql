-- =========================================
-- drop-kdcube-control-plane.sql
-- Removes redesigned control plane components
-- =========================================

-- =========================================
-- Drop triggers (guarded by table existence)
-- =========================================

DO $$
BEGIN
    IF to_regclass('kdcube_control_plane.user_tier_balance') IS NOT NULL THEN
        DROP TRIGGER IF EXISTS trg_cp_utb_updated_at
            ON kdcube_control_plane.user_tier_balance;
    END IF;
END $$;

DO $$
BEGIN
    IF to_regclass('kdcube_control_plane.user_quota_policies') IS NOT NULL THEN
        DROP TRIGGER IF EXISTS trg_cp_uqp_updated_at
            ON kdcube_control_plane.user_quota_policies;
    END IF;
END $$;

DO $$
BEGIN
    IF to_regclass('kdcube_control_plane.application_budget_policies') IS NOT NULL THEN
        DROP TRIGGER IF EXISTS trg_cp_abp_updated_at
            ON kdcube_control_plane.application_budget_policies;
    END IF;
END $$;

DO $$
BEGIN
    IF to_regclass('kdcube_control_plane.tenant_project_budget') IS NOT NULL THEN
        DROP TRIGGER IF EXISTS trg_cp_budget_updated_at
            ON kdcube_control_plane.tenant_project_budget;
    END IF;
END $$;

-- =========================================
-- Drop function(s)
-- =========================================

DROP FUNCTION IF EXISTS kdcube_control_plane.update_updated_at() CASCADE;

-- =========================================
-- Drop indexes
-- (These are dropped automatically with tables,
-- but listed explicitly for clarity/documentation.)
-- =========================================

-- User tier balance indexes
DROP INDEX IF EXISTS kdcube_control_plane.idx_cp_utb_lifetime_balance;
DROP INDEX IF EXISTS kdcube_control_plane.idx_cp_utb_lookup;
DROP INDEX IF EXISTS kdcube_control_plane.idx_cp_utb_expires;

-- Quota policy indexes
DROP INDEX IF EXISTS kdcube_control_plane.idx_cp_uqp_lookup;

-- Application budget policy indexes
DROP INDEX IF EXISTS kdcube_control_plane.idx_cp_abp_lookup;

-- Tenant/project budget indexes
DROP INDEX IF EXISTS kdcube_control_plane.idx_cp_budget_balance;

-- =========================================
-- Drop tables
-- =========================================

DROP TABLE IF EXISTS kdcube_control_plane.tenant_project_budget CASCADE;
DROP TABLE IF EXISTS kdcube_control_plane.application_budget_policies CASCADE;
DROP TABLE IF EXISTS kdcube_control_plane.user_quota_policies CASCADE;
DROP TABLE IF EXISTS kdcube_control_plane.user_tier_balance CASCADE;

-- =========================================
-- Optional: Drop schema (if you want it gone)
-- =========================================
-- DROP SCHEMA IF EXISTS kdcube_control_plane CASCADE;
