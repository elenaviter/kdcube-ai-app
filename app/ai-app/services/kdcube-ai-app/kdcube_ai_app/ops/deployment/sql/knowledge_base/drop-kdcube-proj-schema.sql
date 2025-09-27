-- =========================================
-- drop-knowledge-base.sql (robust)
-- =========================================

-- 0) Drop ALL views in schema (handles dependencies automatically)
DO $$
DECLARE v record;
BEGIN
  FOR v IN
    SELECT schemaname, viewname
    FROM pg_catalog.pg_views
    WHERE schemaname = '<SCHEMA>'
  LOOP
    EXECUTE format('DROP VIEW IF EXISTS %I.%I CASCADE;', v.schemaname, v.viewname);
  END LOOP;
END $$;

-- 1) Triggers (if table still exists)
DO $$
BEGIN
  IF to_regclass('<SCHEMA>.retrieval_segment') IS NOT NULL THEN
    EXECUTE 'DROP TRIGGER IF EXISTS trg_<SCHEMA>_update_search_vector ON <SCHEMA>.retrieval_segment;';
  END IF;
END $$;

-- 2) (Optional) Explicit index drops — not required if you DROP TABLE CASCADE later
--    Safe to keep or remove. Leaving them for completeness.
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_rs_ext_exp;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_rs_ext_mod;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_rs_ext_pub;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_rs_ext_provider;

DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_rs_provider_created;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_rs_provider_resource;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_rs_provider;

DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_rs_resource_created;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_rs_created_at;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_rs_tags;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_rs_resource;

DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_rs_entity_values;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_rs_entities_gin;

DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_rs_embedding;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_rs_search_vector;

DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_ds_mod_text;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_ds_pub_text;

DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_ds_cache_lookup;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_ds_expired;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_ds_expiration;

DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_ds_provider_type;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_ds_provider;

DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_ds_metadata;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_ds_created_at;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_ds_status;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_ds_id_version;

DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_ch_name;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_ch_value;

DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_events_timestamp;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_events_service_metadata;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_events_entity;

-- 3) Functions
DROP FUNCTION IF EXISTS <SCHEMA>.cleanup_expired_data_<SCHEMA>();
DROP FUNCTION IF EXISTS <SCHEMA>.is_datasource_expired_<SCHEMA>(TIMESTAMPTZ);
DROP FUNCTION IF EXISTS <SCHEMA>.extract_entity_values_<SCHEMA>(JSONB);
DROP FUNCTION IF EXISTS <SCHEMA>.update_search_vector_<SCHEMA>();

-- legacy cleanups
DROP FUNCTION IF EXISTS <SCHEMA>.generate_retrieval_segment_rn();
DROP FUNCTION IF EXISTS <SCHEMA>.generate_datasource_rn();

-- 4) Tables (child → parent)
DROP TABLE IF EXISTS <SCHEMA>.retrieval_segment CASCADE;
DROP TABLE IF EXISTS <SCHEMA>.content_hash CASCADE;
DROP TABLE IF EXISTS <SCHEMA>.datasource CASCADE;
DROP TABLE IF EXISTS <SCHEMA>.events CASCADE;

-- 5) Schema
-- DROP SCHEMA IF EXISTS <SCHEMA>;
