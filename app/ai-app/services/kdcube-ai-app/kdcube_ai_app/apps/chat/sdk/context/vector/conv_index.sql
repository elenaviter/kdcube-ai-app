-- =========================================
-- deploy-conversation-history.sql
-- =========================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE SCHEMA IF NOT EXISTS <SCHEMA>;

CREATE TABLE IF NOT EXISTS <SCHEMA>.conv_messages (
                                                      id               BIGSERIAL PRIMARY KEY,
                                                      user_id          TEXT NOT NULL,
                                                      conversation_id  TEXT NOT NULL,
                                                      message_id       TEXT,                           -- ConversationStore id; present for artifacts
                                                      role             TEXT NOT NULL,                  -- 'user' | 'assistant' | 'artifact'
                                                      text             TEXT NOT NULL,
                                                      s3_uri           TEXT NOT NULL,
                                                      ts               TIMESTAMPTZ NOT NULL DEFAULT now(),
    ttl_days         INT NOT NULL DEFAULT 365,
    user_type        TEXT NOT NULL DEFAULT 'anonymous',
    tags             TEXT[] NOT NULL DEFAULT '{}',
    embedding        VECTOR(1536),
    track_id         TEXT
    );

CREATE INDEX IF NOT EXISTS idx_<SCHEMA>_conv_user_conversation_ts
  ON <SCHEMA>.conv_messages (user_id, conversation_id, ts DESC);

CREATE INDEX IF NOT EXISTS idx_<SCHEMA>_conv_user_conv_track_ts
  ON <SCHEMA>.conv_messages (user_id, conversation_id, track_id, ts DESC);

CREATE INDEX IF NOT EXISTS idx_<SCHEMA>_conv_user_type_ts
  ON <SCHEMA>.conv_messages (user_type, ts DESC);

CREATE INDEX IF NOT EXISTS idx_<SCHEMA>_conv_tags
  ON <SCHEMA>.conv_messages USING GIN (tags);

-- speed up recency & scope
CREATE INDEX IF NOT EXISTS idx_<SCHEMA>_cm_scope_time ON <SCHEMA>.conv_messages
  (user_id, conversation_id, track_id, role, ts DESC);

CREATE INDEX IF NOT EXISTS idx_<SCHEMA>_cm_text_trgm
ON <SCHEMA>.conv_messages USING gin (text gin_trgm_ops);

-- ANN (embeddings)
CREATE INDEX IF NOT EXISTS idx_<SCHEMA>_conv_embedding
  ON <SCHEMA>.conv_messages USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);

CREATE OR REPLACE VIEW <SCHEMA>.conv_messages_expired AS
SELECT * FROM <SCHEMA>.conv_messages
WHERE ts + (ttl_days || ' days')::interval < now();


CREATE TABLE IF NOT EXISTS <SCHEMA>.conv_artifact_edges (
                                                            from_id    BIGINT NOT NULL REFERENCES <SCHEMA>.conv_messages(id) ON DELETE CASCADE,
    to_id      BIGINT NOT NULL REFERENCES <SCHEMA>.conv_messages(id) ON DELETE CASCADE,
    policy     TEXT   NOT NULL DEFAULT 'none',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (from_id, to_id)
    );

-- (edges)
CREATE INDEX IF NOT EXISTS idx_<SCHEMA>_edge_from_id ON <SCHEMA>.conv_artifact_edges (from_id);
CREATE INDEX IF NOT EXISTS idx_<SCHEMA>_edge_to_id   ON <SCHEMA>.conv_artifact_edges (to_id);

-- === tracks (first-class) ===
CREATE TABLE IF NOT EXISTS <SCHEMA>.conv_tracks (
                                                    track_id        TEXT PRIMARY KEY,
                                                    user_id         TEXT NOT NULL,
                                                    conversation_id TEXT NOT NULL,
                                                    title           TEXT NOT NULL,
                                                    summary         TEXT NOT NULL DEFAULT '',
                                                    topics          TEXT[] NOT NULL DEFAULT '{}',
                                                    prefs           JSONB NOT NULL DEFAULT '{}',
                                                    message_count   INT NOT NULL DEFAULT 0,
                                                    last_activity   TIMESTAMPTZ NOT NULL DEFAULT now(),
    centroid        VECTOR(1536),
    tags            TEXT[] NOT NULL DEFAULT '{}',
    status          TEXT NOT NULL DEFAULT 'open',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
    );
CREATE INDEX IF NOT EXISTS idx_<SCHEMA>_tracks_user_conv
  ON <SCHEMA>.conv_tracks (user_id, conversation_id);
CREATE INDEX IF NOT EXISTS idx_<SCHEMA>_tracks_updated
  ON <SCHEMA>.conv_tracks (updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_<SCHEMA>_tracks_centroid
  ON <SCHEMA>.conv_tracks USING ivfflat (centroid vector_cosine_ops) WITH (lists=50);

-- === per-track programs ('DAGs') ===
CREATE TABLE IF NOT EXISTS <SCHEMA>.conv_track_programs (
                                                            program_id      TEXT PRIMARY KEY,
                                                            track_id        TEXT NOT NULL,
                                                            user_id         TEXT NOT NULL,
                                                            conversation_id TEXT NOT NULL,
                                                            title           TEXT NOT NULL,
                                                            language        TEXT NOT NULL DEFAULT 'python',
                                                            code            TEXT NOT NULL,
                                                            params          JSONB NOT NULL DEFAULT '{}'::jsonb,
                                                            deliverables    JSONB NOT NULL DEFAULT '{}'::jsonb,
                                                            status          TEXT NOT NULL DEFAULT 'active',
                                                            revision        INT  NOT NULL DEFAULT 1,
                                                            last_run_at     TIMESTAMPTZ,
                                                            last_run_meta   JSONB NOT NULL DEFAULT '{}'::jsonb,
                                                            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
    );
CREATE INDEX IF NOT EXISTS idx_<SCHEMA>_programs_track_updated
  ON <SCHEMA>.conv_track_programs (track_id, updated_at DESC);

-- === per-track tickets ===
CREATE TABLE IF NOT EXISTS <SCHEMA>.conv_track_tickets (
                                                           ticket_id   TEXT PRIMARY KEY,
                                                           track_id    TEXT NOT NULL,
                                                           user_id     TEXT NOT NULL,
                                                           conversation_id TEXT NOT NULL,
                                                           title       TEXT NOT NULL,
                                                           description TEXT NOT NULL DEFAULT '',
                                                           status      TEXT NOT NULL DEFAULT 'open',
                                                           priority    SMALLINT NOT NULL DEFAULT 3,
                                                           assignee    TEXT,
                                                           tags        TEXT[] NOT NULL DEFAULT '{}',
                                                           created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding   VECTOR(1536)
    );
CREATE INDEX IF NOT EXISTS idx_<SCHEMA>_tickets_track
  ON <SCHEMA>.conv_track_tickets (track_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_<SCHEMA>_tickets_status
  ON <SCHEMA>.conv_track_tickets (status, priority DESC);
CREATE INDEX IF NOT EXISTS idx_<SCHEMA>_tickets_embedding
  ON <SCHEMA>.conv_track_tickets USING ivfflat (embedding vector_cosine_ops) WITH (lists=50);
