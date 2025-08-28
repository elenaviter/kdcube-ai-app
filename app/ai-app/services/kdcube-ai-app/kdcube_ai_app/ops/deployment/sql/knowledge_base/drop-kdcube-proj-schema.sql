-- =========================================
-- drop-knowledge-base.sql (FIXED for schema-specific names)
-- =========================================

-- Drop all indexes first (optional, CASCADE will handle this)
-- Retrieval Segment indexes (schema-specific names)
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_rs_entity_values;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_rs_resource_created;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_rs_created_at;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_rs_tags;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_rs_resource;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_rs_entities_gin;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_rs_embedding;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_rs_search_vector;

-- Datasource indexes (schema-specific names)
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_ds_metadata;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_ds_created_at;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_ds_status;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_ds_id_version;

-- Events indexes (schema-specific names)
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_events_timestamp;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_events_service_metadata;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_events_entity;

-- Drop triggers (schema-specific names)
DROP TRIGGER IF EXISTS trg_<SCHEMA>_update_search_vector ON <SCHEMA>.retrieval_segment;

-- Drop functions (schema-specific names)
DROP FUNCTION IF EXISTS <SCHEMA>.extract_entity_values_<SCHEMA>(JSONB);
DROP FUNCTION IF EXISTS <SCHEMA>.update_search_vector_<SCHEMA>();

-- Drop any other functions that might exist (legacy names for safety)
DROP FUNCTION IF EXISTS <SCHEMA>.generate_retrieval_segment_rn();
DROP FUNCTION IF EXISTS <SCHEMA>.generate_datasource_rn();

-- Drop tables (order matters due to foreign key constraints)
DROP TABLE IF EXISTS <SCHEMA>.retrieval_segment CASCADE;
DROP TABLE IF EXISTS <SCHEMA>.datasource CASCADE;
DROP TABLE IF EXISTS <SCHEMA>.events CASCADE;

-- Drop schema (only if empty)
DROP SCHEMA IF EXISTS <SCHEMA>;
