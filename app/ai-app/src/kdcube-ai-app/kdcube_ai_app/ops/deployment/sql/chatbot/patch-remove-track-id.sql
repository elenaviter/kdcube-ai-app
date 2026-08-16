-- =========================================
-- patch-remove-track-id.sql
-- Remove track_id artifacts and conv_track_programs table.
--
-- Set schema name below and run the whole script.
-- =========================================

DO $$
DECLARE
  schema_name text := '<SCHEMA>'; -- set this to your schema, e.g. 'kdcube_demo_tenant_demo_project'
BEGIN
  -- ---------- conv_messages ----------
  EXECUTE format('DROP VIEW IF EXISTS %I.conv_messages_expired', schema_name);

  EXECUTE format('ALTER TABLE IF EXISTS %I.conv_messages DROP COLUMN IF EXISTS track_id', schema_name);

  EXECUTE format('DROP INDEX IF EXISTS %I.%I', schema_name, 'idx_' || schema_name || '_conv_user_conv_track_ts');
  EXECUTE format('DROP INDEX IF EXISTS %I.%I', schema_name, 'idx_' || schema_name || '_cm_scope_time');

  EXECUTE format(
    'CREATE INDEX IF NOT EXISTS %I ON %I.conv_messages (user_id, conversation_id, role, ts DESC)',
    'idx_' || schema_name || '_cm_scope_time',
    schema_name
  );

  EXECUTE format(
    'CREATE OR REPLACE VIEW %I.conv_messages_expired AS ' ||
    'SELECT * FROM %I.conv_messages ' ||
    'WHERE ts + (ttl_days || '' days'')::interval < now()',
    schema_name,
    schema_name
  );

  -- ---------- conv_track_tickets ----------
  EXECUTE format('ALTER TABLE IF EXISTS %I.conv_track_tickets DROP COLUMN IF EXISTS track_id', schema_name);

  EXECUTE format('DROP INDEX IF EXISTS %I.%I', schema_name, 'idx_' || schema_name || '_tickets_track');
  EXECUTE format('DROP INDEX IF EXISTS %I.%I', schema_name, 'conv_track_tickets_lookup_idx');

  EXECUTE format(
    'CREATE INDEX IF NOT EXISTS %I ON %I.conv_track_tickets (user_id, conversation_id, status, updated_at DESC)',
    'conv_track_tickets_lookup_idx',
    schema_name
  );

  -- ---------- conv_tracks / conv_track_programs ----------
  EXECUTE format('DROP TABLE IF EXISTS %I.conv_tracks CASCADE', schema_name);
  EXECUTE format('DROP TABLE IF EXISTS %I.conv_track_programs CASCADE', schema_name);
END $$;
