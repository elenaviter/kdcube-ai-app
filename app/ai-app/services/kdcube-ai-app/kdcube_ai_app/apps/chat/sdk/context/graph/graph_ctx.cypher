// Constraints (Neo4j 5+)
CREATE CONSTRAINT user_id IF NOT EXISTS
FOR (u:User) REQUIRE (u.key) IS UNIQUE;

CREATE CONSTRAINT conversation_key IF NOT EXISTS
FOR (c:Conversation) REQUIRE (c.key) IS UNIQUE;

CREATE CONSTRAINT assertion_id IF NOT EXISTS
FOR (a:Assertion) REQUIRE (a.id) IS UNIQUE;

CREATE CONSTRAINT exception_id IF NOT EXISTS
FOR (e:Exception) REQUIRE (e.id) IS UNIQUE;

// Strong upsert/dedupe for assertions by semantic identity
CREATE CONSTRAINT assertion_identity IF NOT EXISTS
FOR (a:Assertion)
REQUIRE (a.tenant, a.project, a.user, a.key, a.scope, a.desired, a.value_hash) IS UNIQUE;

// Helpful indexes
CREATE INDEX assertion_lookup IF NOT EXISTS
FOR (a:Assertion) ON (a.user, a.key, a.scope);

CREATE INDEX assertion_created_at IF NOT EXISTS
FOR (a:Assertion) ON (a.created_at);

CREATE INDEX exception_lookup IF NOT EXISTS
FOR (e:Exception) ON (e.user, e.rule_key, e.scope);
