DROP VIEW IF EXISTS <SCHEMA>.conv_messages_expired;

DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_conv_embedding;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_conv_tags;
DROP INDEX IF EXISTS <SCHEMA>.idx_<SCHEMA>_conv_user_conversation_ts;

DROP TABLE IF EXISTS <SCHEMA>.conv_messages CASCADE;

-- Note: Extensions are shared across database, so we don't drop them
-- DROP EXTENSION IF EXISTS btree_gin;
-- DROP EXTENSION IF EXISTS pg_trgm;
-- DROP EXTENSION IF EXISTS vector;

-- Drop schema (only if empty)
DROP SCHEMA IF EXISTS <SCHEMA>;