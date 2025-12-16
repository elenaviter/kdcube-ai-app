# Hybrid Backend Integration Guide

## Motivation
- Graceful service degradation 
- Clean, simple, accurate per-backend accounting with transparent hybrid operation

## How It Works

### Architecture

```
web_search() - NO decorator, just orchestrator
    ↓
get_search_backend_or_hybrid()
    ↓
    If enable_hybrid=True and backend != DDG:
        HybridSearchBackend(primary=Brave, spare=DDG)
    Else:
        Single backend
    ↓
HybridSearchBackend.search_many() - NO decorator
    ↓
    Sequential mode:
        1. primary.search_many() - @track_web_search decorator ✓
           → Event 1: brave, queries_successful=3
        2. spare.search_many(failed_only) - @track_web_search decorator ✓
           → Event 2: duckduckgo, queries_successful=2
    ↓
    Result substitution: client sees all 5 results
```

### Key Insight

**Decorators are ONLY on actual backend.search_many() methods**:
- ✅ `DDGSearchBackend.search_many()` - decorated
- ✅ `BraveSearchBackend.search_many()` - decorated
- ❌ `HybridSearchBackend.search_many()` - NOT decorated (just orchestrator)
- ❌ `web_search()` - NOT decorated (just orchestrator)

**Why this works:**
- HybridSearchBackend calls the actual backends
- Those backends have decorators
- Each backend call emits its own event
- Events show only successful queries per backend

## Usage

### Default (Hybrid Enabled)

```python
# With Brave primary, DDG fallback
export WEB_SEARCH_BACKEND="brave"
export BRAVE_API_KEY="your-key"

results = await web_search(
    _SERVICE=service,
    queries='["query1", "query2", "query3"]',
    objective="find something",
    # enable_hybrid=True by default
    # hybrid_mode="sequential" by default
)
```

**What happens:**
1. Brave.search_many() runs for all 3 queries
    - Succeeds on 2, fails on 1 (429)
    - Event: provider="brave", search_queries=2
2. DDG.search_many() runs for 1 failed query
    - Succeeds on 1
    - Event: provider="duckduckgo", search_queries=1
3. Results merged transparently
4. Client receives 3 results

### Parallel Mode

```python
results = await web_search(
    queries='["q1", "q2"]',
    enable_hybrid=True,
    hybrid_mode="parallel",  # Both run for all queries
)
```

**What happens:**
1. Both Brave and DDG run for all queries simultaneously
2. Brave event: search_queries=N (successful count)
3. DDG event: search_queries=M (successful count)
4. Best results used (primary wins ties)

### Disable Hybrid

```python
results = await web_search(
    queries='["q1"]',
    enable_hybrid=False,  # No fallback
)
```

**What happens:**
1. Only primary backend runs
2. One event from primary
3. If it fails, no results (no fallback)

## Accounting Events

### Sequential Mode Example

5 queries, Brave succeeds on 3, fails on 2:

**Event 1 (Brave):**
```json
{
  "event_id": "abc123",
  "service_type": "web_search",
  "provider": "brave",
  "model_or_service": "brave",
  "usage": {
    "search_queries": 3,
    "search_results": 12
  },
  "metadata": {
    "queries_attempted": 5,
    "queries_successful": 3,
    "query_variants": ["q1", "q3", "q5"],
    "per_query_max": 8
  }
}
```

**Event 2 (DDG for failures):**
```json
{
  "event_id": "abc124",
  "service_type": "web_search",
  "provider": "duckduckgo",
  "model_or_service": "duckduckgo",
  "usage": {
    "search_queries": 2,
    "search_results": 8
  },
  "metadata": {
    "queries_attempted": 2,
    "queries_successful": 2,
    "query_variants": ["q2", "q4"],
    "per_query_max": 8
  }
}
```

**Result:**
- Brave billed for: 3 queries
- DDG billed for: 2 queries
- Client sees: 5 results (transparent)

## Configuration

### Environment Variables

```bash
# Primary backend
export WEB_SEARCH_BACKEND="brave"
export BRAVE_API_KEY="your-key"

# Hybrid mode (optional, defaults to "sequential")
export WEB_SEARCH_HYBRID_MODE="sequential"  # or "parallel"

# For explicit hybrid mode
export WEB_SEARCH_BACKEND="hybrid"
export WEB_SEARCH_PRIMARY_BACKEND="brave"
```

### Per-Call Override

```python
# Override hybrid mode for specific call
results = await web_search(
    queries='["q1"]',
    enable_hybrid=True,
    hybrid_mode="parallel",  # Override default
)
```

## Code Flow

### File Organization

```
apps/chat/sdk/tools/backends/web/
├── search_backends.py          # Core backends + factories
│   ├── DDGSearchBackend        # @track_web_search on search_many()
│   ├── BraveSearchBackend      # @track_web_search on search_many()
│   ├── get_search_backend()
│   ├── get_search_backend_or_hybrid()
│   └── web_search()            # NO decorator
│
└── hybrid_search_backends.py   # Orchestration only
    └── HybridSearchBackend     # NO decorator, just calls backends
```

### Execution Flow (Sequential Mode)

```python
# User calls
await web_search(queries='["q1", "q2", "q3"]')

# web_search() gets backend
backend = get_search_backend_or_hybrid(
    enable_hybrid=True,
    hybrid_mode="sequential"
)
# Returns: HybridSearchBackend(primary=Brave, spare=DDG)

# web_search() calls backend
results = await backend.search_many(queries)

# HybridSearchBackend.search_many() (sequential mode)
primary_results = await self.primary.search_many(queries)
# ↑ This is BraveSearchBackend.search_many()
# ↑ Decorated with @track_web_search
# ↑ Tracks: self._last_successful_queries = ["q1", "q3"]
# ↑ Decorator fires: Event with search_queries=2

# Identify failures
failed = ["q2"]  # q2 had empty results

# Run spare for failures
spare_results = await self.spare.search_many(failed)
# ↑ This is DDGSearchBackend.search_many()
# ↑ Decorated with @track_web_search
# ↑ Tracks: self._last_successful_queries = ["q2"]
# ↑ Decorator fires: Event with search_queries=1

# Merge results
merged = [
    primary_results[0],  # q1 from Brave
    spare_results[0],    # q2 from DDG
    primary_results[2],  # q3 from Brave
]

return merged  # Transparent to client
```

## Benefits

✅ **Accurate Billing**
- Only successful queries count per backend
- 429 rate limits excluded from usage
- Clear per-backend accounting

✅ **Transparent Operation**
- Client always gets results
- Doesn't know which backend served what
- Seamless failover

✅ **Two Modes**
- Sequential: Efficient (spare only when needed)
- Parallel: Redundant (both always run)

✅ **Clean Architecture**
- Decorators only on actual backends
- Orchestrators (hybrid, web_search) have no accounting logic
- Each layer does one thing

## Testing

### Test Sequential Mode

```python
results = await web_search(
    queries='["test1", "test2", "test3"]',
    enable_hybrid=True,
    hybrid_mode="sequential",
)

events = await get_turn_events()

# Verify events
brave_events = [e for e in events if e["provider"] == "brave"]
ddg_events = [e for e in events if e["provider"] == "duckduckgo"]

print(f"Brave queries: {brave_events[0]['usage']['search_queries']}")
print(f"DDG queries: {ddg_events[0]['usage']['search_queries']}")
```

### Test Parallel Mode

```python
results = await web_search(
    queries='["test1", "test2"]',
    enable_hybrid=True,
    hybrid_mode="parallel",
)

events = await get_turn_events()

# Both backends should have events
assert len([e for e in events if e["provider"] == "brave"]) == 1
assert len([e for e in events if e["provider"] == "duckduckgo"]) == 1
```

### Test No Hybrid

```python
results = await web_search(
    queries='["test"]',
    enable_hybrid=False,
)

events = await get_turn_events()

# Only one event
assert len(events) == 1
```

## Summary

**The key insight:** Decorators are on the backends that do the actual work (DDG, Brave), not on the orchestrators (HybridSearchBackend, web_search). This way:

1. Each backend tracks its own successes
2. Each backend emits its own event
3. Events show accurate per-backend usage
4. Orchestrators just coordinate - no accounting logic needed
