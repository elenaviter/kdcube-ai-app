- SSE connections: 50 users - 50 streams (if one per user)
  → does NOT multiply Redis connections, but adds memory/FD/queue load.
- Redis commands spike from:
    - rate limiter
    - throttling monitor
    - circuit breaker
    - backpressure checks
    - gateway config listener
    - session manager
      These all use the shared Redis pools, not per-user.
- Postgres spikes from:
    - conversations fetch
    - conversation browser/index
    - resources / attachments
    - control plane endpoints

### 1) Processing concurrency (proc)

This is the real bottleneck for chat turns.

proc_capacity_per_instance = processes_per_instance * concurrent_requests_per_process
total_proc_capacity = proc_instances * proc_capacity_per_instance

That’s the cap for simultaneous workflow runs.

### 2) SSE connections (ingress)

SSE does not consume Redis per user, but does consume:

- file descriptors
- memory queues
- event loop time

sse_capacity_per_instance = processes_per_instance * limits.max_sse_connections_per_instance
total_sse_capacity = ingress_instances * sse_capacity_per_instance

### 3) Redis connections

Per process you have:

- 1 pubsub connection
- 3 pools (async, async decode, sync), each up to redis_max_connections

Approx per process:

redis_conns_per_process ≈ 1 + 3 * redis_max_connections

Total Redis connections across services:

ingress_instances * ingress_workers * (1 + 3*redis_max_connections_ingress)
+ proc_instances * proc_workers * (1 + 3*redis_max_connections_proc)
+ exec_concurrency * 1   # exec processes set pools.redis_max_connections=1 in gateway config

### 4) Postgres connections

pg_conns_total = ingress_instances * ingress_workers * pg_pool_max_ingress
+ proc_instances * proc_workers * pg_pool_max_proc

Keep this below ~80% of max_connections.


## Answering your “100 users” question

You need two numbers:

1. SSE streams (users × tabs)
2. concurrent turns (users × “active” fraction)

Example:

- 100 users
- Avg tabs per user = 3 → 300 SSE streams
- Avg active fraction = 0.1 → 10 concurrent turns

So:

- Ingress: 1 instance with 4 workers × 200 SSE cap per worker → capacity 800 (safe)
- Processor: 1 instance with 4 workers × 8 concurrent → 32 capacity (safe)

If you expect 100 users actively chatting at the same time, use 0.3–0.5 active fraction and scale processors accordingly.

———

## What about gateway overhead?

The gateway uses Redis on each request for:

- rate limiter
- circuit breaker
- backpressure checks

These go through the same Redis pool in that process. So there’s no extra “gateway pool”; it’s accounted in redis_max_connections. To be safe, set:

- ingress: redis_max_connections = 15–30
- proc: redis_max_connections = 2× concurrent_requests_per_process

———

## What about integrations on proc?

Integrations share the same pools. If you need to prevent them from stealing all concurrency, add an explicit integration concurrency cap (semaphore). I can add:

MAX_INTEGRATIONS_CONCURRENCY

so that queue workers and integrations don’t fight.

———

## Recap: how to expose caps cleanly

We now have gateway config for:

- service_capacity (proc concurrency)
- pools (PG + Redis)

We need explicit SSE cap in config
```json
{
  "limits": {
    "ingress": {
      "max_sse_connections_per_instance": 200
    }
  }
}
```
