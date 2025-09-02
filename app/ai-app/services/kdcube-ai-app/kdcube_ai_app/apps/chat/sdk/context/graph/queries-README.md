# Useful Neo4j queries

You store assertions/exceptions with `{tenant, project, user, session, scope, desired, key, value, ...}`. Here are practical queries you can paste into Neo4j Browser or your code.

### (A) **Session-scoped** positives/negatives + exceptions

*Give me what we collected for this conversation only.*

```cypher
// Params:
// :param tenant => "TENANT_ID";
// :param project => "PROJECT_ID";
// :param user => "USER_ID";
// :param session => "SESSION_ID";

MATCH (s:Session {key: $tenant + ":" + $project + ":" + $session})-[:INCLUDES]->(a:Assertion)
WHERE a.tenant = $tenant AND a.project = $project
WITH s, a
ORDER BY a.created_at DESC
WITH s,
     collect(CASE WHEN a.desired = true  THEN a END) AS positives,
     collect(CASE WHEN a.desired = false THEN a END) AS negatives
OPTIONAL MATCH (s)-[:INCLUDES]->(e:Exception)
WHERE e.tenant = $tenant AND e.project = $project
RETURN
  [x IN positives | {key:x.key, value:x.value, scope:x.scope, confidence:x.confidence, created_at:x.created_at, reason:x.reason}] AS positive_assertions,
  [x IN negatives | {key:x.key, value:x.value, scope:x.scope, confidence:x.confidence, created_at:x.created_at, reason:x.reason}] AS negative_assertions,
  [x IN collect(e) | {rule_key:x.rule_key, value:x.value, scope:x.scope, created_at:x.created_at, reason:x.reason}] AS exceptions;
```

### (B) **User-wide** view (all scopes) with quick precedence for assertions

*Summarize what we’ve collected for the user (project-scoped), separating positives/negatives and including exceptions.*

```cypher
// Params:
// :param tenant => "TENANT_ID";
// :param project => "PROJECT_ID";
// :param user => "USER_ID";

MATCH (u:User {key: $tenant + ":" + $project + ":" + $user})-[:HAS_ASSERTION]->(a:Assertion)
WHERE a.tenant = $tenant AND a.project = $project
WITH a
// crude precedence: session > user > global; neg beats pos within same scope
ORDER BY
  CASE a.scope WHEN "session" THEN 0 WHEN "user" THEN 1 ELSE 2 END,
  CASE WHEN a.desired = false THEN 0 ELSE 1 END,
  a.created_at DESC
WITH a.key AS key, collect(a) AS aa
WITH key, aa[0] AS head, aa[1..] AS tail
WITH key, head,
     CASE WHEN head.desired THEN "positive" ELSE "negative" END AS bucket
WITH bucket, collect({key:key, value:head.value, scope:head.scope, confidence:head.confidence, created_at:head.created_at, reason:head.reason}) AS items
RETURN
  [x IN (CASE WHEN bucket = "positive" THEN items ELSE [] END) | x] AS effective_positive_assertions,
  [x IN (CASE WHEN bucket = "negative" THEN items ELSE [] END) | x] AS effective_negative_assertions
;
```

### (C) **User-wide exceptions** (pair with B)

```cypher
MATCH (u:User {key: $tenant + ":" + $project + ":" + $user})-[:HAS_EXCEPTION]->(e:Exception)
WHERE e.tenant = $tenant AND e.project = $project
RETURN [x IN collect(e) | {rule_key:x.rule_key, value:x.value, scope:x.scope, created_at:x.created_at, reason:x.reason}] AS exceptions;
```


### (D) Wipe DB
```cypher
// 1) delete all nodes/relationships
MATCH (n) DETACH DELETE n;

// 2) drop your constraints (exact names from your file)
DROP CONSTRAINT user_key IF EXISTS;
DROP CONSTRAINT conversation_key IF EXISTS;
DROP CONSTRAINT assertion_id IF EXISTS;
DROP CONSTRAINT exception_id IF EXISTS;
DROP CONSTRAINT assertion_identity IF EXISTS;
DROP CONSTRAINT exception_identity IF EXISTS;

// 3) drop your indexes (tables + rel-prop indexes)
DROP INDEX user_user_type IF EXISTS;
DROP INDEX user_created_at IF EXISTS;

DROP INDEX conversation_user_id IF EXISTS;
DROP INDEX conversation_last_seen IF EXISTS;
DROP INDEX conversation_topic_latest IF EXISTS;
DROP INDEX conversation_meta_updated IF EXISTS;

DROP INDEX assertion_lookup IF EXISTS;
DROP INDEX assertion_created_at IF EXISTS;
DROP INDEX assertion_last_seen IF EXISTS;
DROP INDEX assertion_hits IF EXISTS;

DROP INDEX exception_lookup IF EXISTS;
DROP INDEX exception_created_at IF EXISTS;

DROP INDEX includes_last_seen IF EXISTS;
DROP INDEX includes_hits IF EXISTS;

// 4) verify clean
SHOW CONSTRAINTS;
SHOW INDEXES;
```

### cypher-shell Inside the docker
```shell
cypher-shell -a bolt://neo4j:7687 -u neo4j -p '<pass>'
```