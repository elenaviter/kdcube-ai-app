**Connection Pooling (Chat Service)**

This doc describes how **Redis** and **Postgres** pools are created, shared, and closed **per worker (process)** in the chat service.

---

**Where Pools Are Created (Chat)**

All pools are created once per process in `apps/chat/api/resolvers.py` and stored in `app.state` during FastAPI lifespan in `apps/chat/api/web_app.py`.

- Postgres:
  - `get_pg_pool()` → `app.state.pg_pool`
  - Closed on shutdown via `pg_pool.close()`
- Redis:
  - `get_redis_clients()` →
    - `app.state.redis_async`
    - `app.state.redis_async_decode`
    - `app.state.redis_sync`
  - Closed on shutdown via `close_redis_clients()`

---

**Redis Pools (Per Process)**

Each worker (process) creates **three shared Redis pools**:

1. `redis_async`
   - async client
   - `decode_responses=False`
2. `redis_async_decode`
   - async client
   - `decode_responses=True`
3. `redis_sync`
   - sync client
   - `decode_responses=False`

These are shared across all chat components (processor, monitoring, gateway, bundles, etc.).
No additional Redis pools should be created outside these.

**Size control**
- `REDIS_MAX_CONNECTIONS` (env) caps **each pool**.
- Approx max Redis connections per process:

```
max_redis_conns_per_process ≈ 3 * REDIS_MAX_CONNECTIONS
```

If `REDIS_MAX_CONNECTIONS` is **unset**, the pool size is unbounded and will grow with load.

**Client names (for `CLIENT LIST`)**

All pools set a Redis client name automatically so you can group them by service:

```
<REDIS_CLIENT_NAME or SERVICE_NAME or APP_NAME>:<INSTANCE_ID or HOSTNAME>:<PID>:<pool_kind>
```

Pool kinds:
- `async`
- `async_decode`
- `sync`

To override the prefix, set `REDIS_CLIENT_NAME`.

**Where implemented**
- `kdcube_ai_app/infra/redis/client.py`
- `apps/chat/api/resolvers.py`
- `apps/chat/api/web_app.py`

---

**Postgres Pool (Per Process)**

Each worker creates **one asyncpg pool**:

```
app.state.pg_pool
```

**Size control**
- `PGPOOL_MAX_SIZE` (env) → hard cap for pool size
- `PGPOOL_MIN_SIZE` (env) → minimum connections
- If `PGPOOL_MAX_SIZE` is **not set**, it defaults to:

```
gateway_config.service_capacity.concurrent_requests_per_process
```

**Approx Postgres connections per instance**

```
pg_conns_per_instance ≈ workers * pg_pool_max_size
```

Where:
- `workers` = `service_capacity.processes_per_instance`
- `pg_pool_max_size` = `PGPOOL_MAX_SIZE` or fallback

**Where implemented**
- `apps/chat/api/resolvers.py` → `get_pg_pool()`
- `apps/chat/api/web_app.py` → lifespan startup/shutdown

---

**Operational Notes**

- Redis and Postgres pools are **per process**. Total connections scale with worker count.
- When you raise `service_capacity.processes_per_instance`, you also raise total Postgres and Redis connections.
- For Redis max clients, ensure:

```
total_redis_connections ≈ processes * 3 * REDIS_MAX_CONNECTIONS
```

fits within Elasticache `maxclients`.

- For Postgres, ensure:

```
total_pg_connections ≈ processes * PGPOOL_MAX_SIZE
```

fits within `max_connections` on the DB.

---

**Redis Connection Monitor (Centralized)**

Chat starts a lightweight Redis health monitor in-process:

- Tracks `PING` health on the shared async client.
- Emits **up/down** events to registered listeners.
- Useful for components that need to **resubscribe** (e.g., pubsub listeners).

Env:
- `REDIS_HEALTHCHECK_INTERVAL_SEC` (default: `5`)
- `REDIS_HEALTHCHECK_TIMEOUT_SEC` (default: `2`)

Access:
- `app.state.redis_monitor` (created during FastAPI lifespan)

You can register a listener:

```python
monitor = request.app.state.redis_monitor
monitor.add_listener(lambda state, err: print("redis:", state, err))
```

---

**Code Executor Context (Docker/Fargate)**

The code executor runs in a **separate process/container** (Docker or Fargate).  
If a tool inside executed code emits chat events, it will create its **own Redis connection** in that isolated runtime.

Key points:
- Executor Redis is **not shared** with the main chat service pools.
- It is **lazy**: the Redis connection is created only when the tool emits via the communicator.
- This uses the standard communicator path (`ChatRelayCommunicator` → `ServiceCommunicator._ensure_async()`).

Relevant code:
- `apps/chat/sdk/runtime/bootstrap.py` → `make_chat_comm(...)` builds the communicator.
- `apps/chat/sdk/runtime/comm_ctx.py` → `get_comm()` returns the communicator lazily.
- `infra/orchestration/app/communicator.py` → `_ensure_async()` creates the Redis client on first publish.

Operational impact:
- Expect **+1 Redis connection per executor container** that actually emits events.
- If emitters are rare, this is negligible; if you scale executors, budget Redis `maxclients` accordingly.

---

**How To Check Limits (CLI)**

**Postgres**

Check max connections from SQL:

```bash
psql "host=<host> port=5432 dbname=<db> user=<user> sslmode=require" -c "SHOW max_connections;"
```

If you need a password prompt:

```bash
PGPASSWORD="<password>" psql "host=<host> port=5432 dbname=<db> user=<user>" -c "SHOW max_connections;"
```

Other useful counters:

```sql
SHOW max_connections;
SELECT count(*) FROM pg_stat_activity;
SELECT name, setting FROM pg_settings WHERE name IN ('max_connections','shared_buffers','work_mem');
```

**Redis**

Using redis-cli (works for ElastiCache if you can connect to the endpoint):

```bash
redis-cli -h <host> -p 6379 -a "<password>" INFO clients
```

Look for:
- `maxclients`
- `connected_clients`

ElastiCache usually disables `CONFIG` commands, so this may fail:

```bash
redis-cli -h <host> -p 6379 -a "<password>" CONFIG GET maxclients
```

If `CONFIG` is disabled, rely on `INFO clients`.

Additional useful counters:

```bash
redis-cli -h <host> -p 6379 -a "<password>" INFO stats
```

Check:
- `rejected_connections` (non‑zero means you *did* hit maxclients)
- `total_connections_received` (how many connections were opened since start)

**Who is holding connections (snapshot)**

```bash
redis-cli -h <host> -p 6379 -a "<password>" CLIENT LIST
```

To aggregate by client address:

```bash
redis-cli -h <host> -p 6379 -a "<password>" CLIENT LIST | \
  awk -F' ' '{addr=""; name=""; for(i=1;i<=NF;i++){if($i ~ /^addr=/) addr=substr($i,6); if($i ~ /^name=/) name=substr($i,6)}; print addr, name}' | \
  sort | uniq -c | sort -nr | head
```

---

**How To Change Limits**

**RDS Postgres**

`max_connections` is controlled by **DB parameter group** and also bounded by instance memory.

High-level steps:
1. Identify the DB parameter group:

```bash
aws rds describe-db-instances --db-instance-identifier <id> \
  --query 'DBInstances[0].DBParameterGroups[0].DBParameterGroupName' --output text
```

2. Update the parameter group (example):

```bash
aws rds modify-db-parameter-group \
  --db-parameter-group-name <param-group> \
  --parameters "ParameterName=max_connections,ParameterValue=200,ApplyMethod=pending-reboot"
```

3. Reboot the instance for it to take effect:

```bash
aws rds reboot-db-instance --db-instance-identifier <id>
```

Note: If you hit memory limits, you may need a larger instance class.

**ElastiCache Redis**

`maxclients` is **not** typically editable in ElastiCache; it is determined by the node type.

To increase it, **scale up the node type** (via console or AWS CLI):

```bash
aws elasticache modify-replication-group \
  --replication-group-id <id> \
  --cache-node-type <new-node-type> \
  --apply-immediately
```

If you use a standalone cluster:

```bash
aws elasticache modify-cache-cluster \
  --cache-cluster-id <id> \
  --cache-node-type <new-node-type> \
  --apply-immediately
```
