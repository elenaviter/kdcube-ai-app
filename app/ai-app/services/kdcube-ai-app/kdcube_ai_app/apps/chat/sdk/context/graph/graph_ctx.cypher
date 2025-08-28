// Constraints (Neo4j 5+)
CREATE CONSTRAINT user_id IF NOT EXISTS
FOR (u:User) REQUIRE (u.key) IS UNIQUE;

CREATE CONSTRAINT conversation_key IF NOT EXISTS
FOR (c:Conversation) REQUIRE (c.key) IS UNIQUE;

CREATE CONSTRAINT assertion_id IF NOT EXISTS
FOR (a:Assertion) REQUIRE (a.id) IS UNIQUE;

CREATE CONSTRAINT exception_id IF NOT EXISTS
FOR (e:Exception) REQUIRE (e.id) IS UNIQUE;
