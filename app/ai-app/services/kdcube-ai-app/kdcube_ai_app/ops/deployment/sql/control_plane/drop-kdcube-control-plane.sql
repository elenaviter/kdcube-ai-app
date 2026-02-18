-- =========================================
-- drop-kdcube-control-plane.sql
-- Clean reset of control plane schema
-- =========================================

-- Drop entire control plane schema and all objects within it.
-- Safe for a clean start before re-deploying the schema.

DROP SCHEMA IF EXISTS kdcube_control_plane CASCADE;
