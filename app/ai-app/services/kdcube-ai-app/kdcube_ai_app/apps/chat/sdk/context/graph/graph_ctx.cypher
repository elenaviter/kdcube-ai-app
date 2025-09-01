// ============ Nodes & Uniqueness ============

CREATE CONSTRAINT user_key IF NOT EXISTS
FOR (u:User) REQUIRE u.key IS UNIQUE;

CREATE CONSTRAINT conversation_key IF NOT EXISTS
FOR (c:Conversation) REQUIRE c.key IS UNIQUE;

CREATE CONSTRAINT assertion_id IF NOT EXISTS
FOR (a:Assertion) REQUIRE a.id IS UNIQUE;

CREATE CONSTRAINT exception_id IF NOT EXISTS
FOR (e:Exception) REQUIRE e.id IS UNIQUE;

// De-dup assertions by semantic identity (project + user + key + scope + desired + normalized value)
CREATE CONSTRAINT assertion_identity IF NOT EXISTS
FOR (a:Assertion)
REQUIRE (a.tenant, a.project, a.user, a.key, a.scope, a.desired, a.value_hash) IS UNIQUE;

// De-dup exceptions similarly (optional but useful)
CREATE CONSTRAINT exception_identity IF NOT EXISTS
FOR (e:Exception)
REQUIRE (e.tenant, e.project, e.user, e.rule_key, e.scope, e.value_hash) IS UNIQUE;

// ============ Helpful Indexes ============

CREATE INDEX user_user_type IF NOT EXISTS
FOR (u:User) ON (u.user_type);

CREATE INDEX user_created_at IF NOT EXISTS
FOR (u:User) ON (u.created_at);

CREATE INDEX conversation_user_id IF NOT EXISTS
FOR (c:Conversation) ON (c.user_id);

CREATE INDEX conversation_last_seen IF NOT EXISTS
FOR (c:Conversation) ON (c.last_seen_at);

CREATE INDEX conversation_topic_latest IF NOT EXISTS
FOR (c:Conversation) ON (c.topic_latest);

CREATE INDEX conversation_meta_updated IF NOT EXISTS
FOR (c:Conversation) ON (c.meta_updated_at);

CREATE INDEX assertion_lookup IF NOT EXISTS
FOR (a:Assertion) ON (a.user, a.key, a.scope);

CREATE INDEX assertion_created_at IF NOT EXISTS
FOR (a:Assertion) ON (a.created_at);

CREATE INDEX exception_lookup IF NOT EXISTS
FOR (e:Exception) ON (e.user, e.rule_key, e.scope);

CREATE INDEX exception_created_at IF NOT EXISTS
FOR (e:Exception) ON (e.created_at);
