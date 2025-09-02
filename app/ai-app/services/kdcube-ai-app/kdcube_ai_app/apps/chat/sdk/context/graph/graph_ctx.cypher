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

// ===== Helpful indexes (new/extra) =====
CREATE INDEX assertion_last_seen IF NOT EXISTS
FOR (a:Assertion) ON (a.last_seen_at);

CREATE INDEX assertion_hits IF NOT EXISTS
FOR (a:Assertion) ON (a.hits);

// Relationship property indexes (Neo4j 5+)
CREATE INDEX includes_last_seen IF NOT EXISTS
FOR ()-[r:INCLUDES]-() ON (r.last_seen);

CREATE INDEX includes_hits IF NOT EXISTS
FOR ()-[r:INCLUDES]-() ON (r.hits);
