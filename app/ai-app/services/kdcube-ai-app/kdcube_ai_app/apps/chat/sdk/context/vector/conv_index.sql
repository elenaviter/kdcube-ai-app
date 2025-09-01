-- =========================================
-- deploy-conversation-history.sql
-- =========================================

CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS <SCHEMA>;

CREATE TABLE IF NOT EXISTS <SCHEMA>.conv_messages (
  id               BIGSERIAL PRIMARY KEY,
  user_id          TEXT NOT NULL,
  conversation_id  TEXT NOT NULL,
  message_id       TEXT,
  role             TEXT NOT NULL,                -- 'user' | 'assistant' | 'artifact'
  text             TEXT NOT NULL,
  s3_uri           TEXT NOT NULL,
  ts               TIMESTAMPTZ NOT NULL DEFAULT now(),
  ttl_days         INT NOT NULL DEFAULT 365,     -- set by app per user_type
  user_type        TEXT NOT NULL DEFAULT 'anonymous',  -- anonymous | registered | privileged | paid
  tags             TEXT[] NOT NULL DEFAULT '{}',
  embedding        VECTOR(1536)                  -- adjust to your embedding size
);

CREATE INDEX IF NOT EXISTS idx_<SCHEMA>_conv_user_conversation_ts
  ON <SCHEMA>.conv_messages (user_id, conversation_id, ts DESC);

CREATE INDEX IF NOT EXISTS idx_<SCHEMA>_conv_user_type_ts
  ON <SCHEMA>.conv_messages (user_type, ts DESC);

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

-- NOTE:
-- If you're on pgvector >= 0.5 and want HNSW instead of IVFFLAT, you can use:
-- CREATE INDEX IF NOT EXISTS idx_<SCHEMA>_conv_embedding
--   ON <SCHEMA>.conv_messages USING hnsw (embedding vector_cosine_ops)
-- WITH (m=16, ef_construction=200);
-- and at query time: SET hnsw.ef_search = 100;  (tune as desired)
