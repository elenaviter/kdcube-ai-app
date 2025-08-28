CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS <SCHEMA>;

-- =========================================
-- 9) deploy-conversation-history.sql
-- =========================================

CREATE TABLE IF NOT EXISTS <SCHEMA>.conv_messages (
                                                      id           BIGSERIAL PRIMARY KEY,
    -- note: tenant & project are NOT stored here; you deploy per-project schema
                                                      user_id      TEXT NOT NULL,
  conversation_id  TEXT NOT NULL,  -- thread id
  message_id       TEXT,           -- source message id from object store

                                                      role         TEXT NOT NULL, -- 'user' | 'assistant' | 'artifact'
                                                      text         TEXT NOT NULL,
                                                      s3_uri       TEXT NOT NULL, -- generic URI (file:// or s3://), kept for lineage
                                                      ts           TIMESTAMPTZ NOT NULL DEFAULT now(),
    ttl_days     INT NOT NULL DEFAULT 365,
    tags         TEXT[] NOT NULL DEFAULT '{}',
    embedding    VECTOR(1536)   -- adjust to your embedding size
    );

CREATE INDEX IF NOT EXISTS idx_<SCHEMA>_conv_user_conversation_ts
  ON <SCHEMA>.conv_messages (user_id, conversation_id, ts DESC);

CREATE INDEX IF NOT EXISTS idx_<SCHEMA>_conv_tags
    ON <SCHEMA>.conv_messages USING GIN (tags);

-- Build IVFFLAT index (OK to create early; for best recall, create after initial inserts)
CREATE INDEX IF NOT EXISTS idx_<SCHEMA>_conv_embedding
    ON <SCHEMA>.conv_messages USING ivfflat (embedding vector_cosine_ops)
WITH (lists=100);

-- Optional: view for expired rows
CREATE OR REPLACE VIEW <SCHEMA>.conv_messages_expired AS
SELECT *
FROM <SCHEMA>.conv_messages
WHERE ts + (ttl_days || ' days')::interval < now();