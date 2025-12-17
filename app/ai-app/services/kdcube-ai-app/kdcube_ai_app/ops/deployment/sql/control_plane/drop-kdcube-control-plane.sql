-- =========================================
-- drop-kdcube-control-plane.sql
-- Removes control plane components
-- =========================================

-- Drop views first
DROP VIEW IF EXISTS kdcube_control_plane.active_budget_policies CASCADE;
DROP VIEW IF EXISTS kdcube_control_plane.active_quota_policies CASCADE;
DROP VIEW IF EXISTS kdcube_control_plane.expired_replenishments CASCADE;
DROP VIEW IF EXISTS kdcube_control_plane.active_replenishments CASCADE;

-- Drop functions
DROP FUNCTION IF EXISTS kdcube_control_plane.cleanup_expired_replenishments() CASCADE;
DROP FUNCTION IF EXISTS kdcube_control_plane.is_replenishment_expired(TIMESTAMPTZ) CASCADE;
DROP FUNCTION IF EXISTS kdcube_control_plane.update_abp_updated_at() CASCADE;
DROP FUNCTION IF EXISTS kdcube_control_plane.update_uqp_updated_at() CASCADE;
DROP FUNCTION IF EXISTS kdcube_control_plane.update_uqr_updated_at() CASCADE;

-- Drop triggers
DO $$
BEGIN
    IF to_regclass('kdcube_control_plane.application_budget_policies') IS NOT NULL THEN
        DROP TRIGGER IF EXISTS trg_cp_abp_updated_at ON kdcube_control_plane.application_budget_policies;
    END IF;
END $$;

DO $$
BEGIN
    IF to_regclass('kdcube_control_plane.user_quota_policies') IS NOT NULL THEN
        DROP TRIGGER IF EXISTS trg_cp_uqp_updated_at ON kdcube_control_plane.user_quota_policies;
    END IF;
END $$;

DO $$
BEGIN
    IF to_regclass('kdcube_control_plane.user_quota_replenishment') IS NOT NULL THEN
        DROP TRIGGER IF EXISTS trg_cp_uqr_updated_at ON kdcube_control_plane.user_quota_replenishment;
    END IF;
END $$;

-- Drop indexes (automatically dropped with tables, but listed for documentation)

-- Budget policy indexes
DROP INDEX IF EXISTS kdcube_control_plane.idx_cp_abp_tenant;
DROP INDEX IF EXISTS kdcube_control_plane.idx_cp_abp_lookup;

-- Quota policy indexes
DROP INDEX IF EXISTS kdcube_control_plane.idx_cp_uqp_tenant;
DROP INDEX IF EXISTS kdcube_control_plane.idx_cp_uqp_lookup;

-- Replenishment indexes
DROP INDEX IF EXISTS kdcube_control_plane.idx_cp_uqr_tenant;
DROP INDEX IF EXISTS kdcube_control_plane.idx_cp_uqr_purchase;
DROP INDEX IF EXISTS kdcube_control_plane.idx_cp_uqr_user;
DROP INDEX IF EXISTS kdcube_control_plane.idx_cp_uqr_expires;
DROP INDEX IF EXISTS kdcube_control_plane.idx_cp_uqr_lookup;

-- Drop tables
DROP TABLE IF EXISTS kdcube_control_plane.application_budget_policies CASCADE;
DROP TABLE IF EXISTS kdcube_control_plane.user_quota_policies CASCADE;
DROP TABLE IF EXISTS kdcube_control_plane.user_quota_replenishment CASCADE;

-- Optional: Drop schema (uncomment if you want to remove the entire schema)
-- DROP SCHEMA IF EXISTS kdcube_control_plane CASCADE;