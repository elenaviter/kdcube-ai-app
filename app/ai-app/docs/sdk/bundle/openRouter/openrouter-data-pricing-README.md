---
title: "OpenRouter Data Bundle — Pricing & Cost Tracking"
summary: "How OpenRouter pricing works, default model costs, integration with the accounting system, and cost optimization tips."
tags: ["bundle", "openrouter", "pricing", "costs", "accounting", "billing"]
keywords: ["openrouter pricing", "token cost", "cost tracking", "accounting", "billing", "data-processor", "usage metrics"]
see_also:
  - ks:docs/sdk/bundle/openRouter/openrouter-data-README.md
  - ks:docs/sdk/bundle/openRouter/entrypoint.md
---

# OpenRouter Data Bundle — Pricing & Cost Tracking

This document covers how pricing works with OpenRouter, how the bundle tracks costs,
and strategies for cost optimization.

## OpenRouter pricing model

OpenRouter uses **pay-per-token pricing**. You pay only for the tokens you consume,
with per-token rates that vary by model.

### Finding model pricing

- Browse all models: https://openrouter.ai/models
- Each model page shows `$X per 1M input tokens` and `$Y per 1M output tokens`
- Pricing updates in real-time on the platform

### Default model cost

The bundle defaults to `google/gemini-2.5-flash-preview`, one of the cheapest options:

| Metric | Value |
|--------|-------|
| Input tokens | ~$0.075 per 1M |
| Output tokens | ~$0.30 per 1M |
| Typical 4096-token response | ~$0.001–$0.002 |

Prices are as of March 2026 and may change. Always check https://openrouter.ai/models
for current rates.

### Cost calculation

```
total_cost = (input_tokens / 1_000_000) × input_price_per_1m
           + (output_tokens / 1_000_000) × output_price_per_1m
```

Example: 100 input tokens + 500 output tokens with Gemini Flash:
```
cost = (100 / 1_000_000) × 0.075 + (500 / 1_000_000) × 0.30
     = 0.0000075 + 0.00015
     = $0.00022500
```

## Cost tracking via accounting system

### Accounting integration

Every OpenRouter call is automatically tracked by wrapping it in an `with_accounting()` context:

```python
with with_accounting("data-processor", metadata={"openrouter_model": model}):
    result = await openrouter_completion(...)
```

The `openrouter_completion()` function extracts usage from the API response, and the
accounting layer automatically captures:

| Field | Value |
|-------|-------|
| `role` | `"data-processor"` |
| `input_tokens` | From OpenRouter response |
| `output_tokens` | From OpenRouter response |
| `total_tokens` | Sum of input + output |
| `model` | Model slug (e.g., `google/gemini-2.5-flash-preview`) |
| `metadata.openrouter_model` | Model slug for attribution |

### Cost data in SSE events

When a request completes successfully, the workflow emits a "done" event with usage metrics:

```json
{
  "type": "chat.step",
  "step": "processing",
  "status": "done",
  "title": "Processing Complete",
  "data": {
    "model": "google/gemini-2.5-flash-preview",
    "usage": {
      "prompt_tokens": 42,
      "completion_tokens": 100,
      "total_tokens": 142
    }
  },
  "markdown": "Processed by **google/gemini-2.5-flash-preview** (142 tokens)"
}
```

The client can use `usage.total_tokens` to:
- Display real-time cost to users (if desired)
- Alert on unexpectedly high token counts
- Track token distribution across requests

### Querying costs in the database

After a request completes, the accounting system stores cost data in:

| System | Schema | Lookup keys |
|--------|--------|-------------|
| PostgreSQL | `accounting.*` | `request_id`, `tenant`, `project`, `user`, `role` |
| Redis cache | `accounting:*` | `request_id` |

**Example query** (pseudocode):
```sql
SELECT request_id, role, usage, cost_usd, model, created_at
FROM accounting.cost_events
WHERE role = 'data-processor'
  AND tenant = 'my-tenant'
  AND created_at > NOW() - INTERVAL '7 days'
ORDER BY created_at DESC;
```

For the full accounting API, see `kdcube_ai_app/infra/accounting.py`.

## Cost visibility example

**Complete flow from request to cost data:**

```
1. User sends message → ChatTaskPayload with request_id, tenant, user
                    ↓
2. Bundle calls openrouter_completion(model, messages, ...)
                    ↓
3. OpenRouter API returns {text, usage: {prompt_tokens, completion_tokens}}
                    ↓
4. with_accounting() context captures usage + calculates cost
                    ↓
5. Accounting system logs event to PostgreSQL + Redis
                    ↓
6. SSE "done" event emitted with usage.total_tokens to client
                    ↓
7. Operator/billing system queries accounting.cost_events by request_id, tenant, time range
                    ↓
8. Cost report generated for chargeback or internal analytics
```

## Cost optimization strategies

### 1. Choose the right model

| Model | Cost | Latency | Quality | Best for |
|-------|------|---------|---------|----------|
| Gemini 2.5 Flash | ★☆☆ | Fast | Good | Default; structured data |
| Llama 3.1 8B | ★☆☆ | Fast | Fair | Budget-conscious |
| Mistral 7B | ★★☆ | Medium | Good | Balanced |
| Claude 3.5 Haiku | ★★☆ | Medium | Very good | Complex tasks |
| Claude 3.5 Sonnet | ★★★ | Slow | Excellent | High accuracy needed |

Override the default model via bundle configuration:

```python
config = Config(role_models={
    "data-processor": {
        "provider": "openrouter",
        "model": "meta-llama/llama-3.1-8b-instruct"
    }
})
```

### 2. Reduce input tokens

- **Summarize long documents** before sending (pre-processing)
- **Use concise system prompts** (the bundle's is already minimal)
- **Limit `max_tokens`** to what's truly needed (default is 4096)

### 3. Monitor and alert on token usage

Check the "done" SSE event for `usage.total_tokens`:
- If `total_tokens > 3000` for a simple task, investigate
- Set alerts on unusual patterns (e.g., 10x spike)
- Use token count as a canary for malformed inputs

### 4. Batch similar requests

If you have many similar extraction/classification tasks:
- Batch them in a single message (with clear delimiters)
- Reduces per-request overhead
- Trades off isolation (errors in one item may affect others)

Example:
```
System: Extract person name from each JSON object. Return as JSON array.

User:
{"text": "John works in sales"}
{"text": "Mary is an engineer"}
```

### 5. Cache frequently-used contexts (future)

OpenRouter does not yet support prompt caching like Claude does. If you repeatedly
process the same instruction set (e.g., a fixed tagging schema), future versions
may support caching at the API level.

## Cost alerts and limits

### Set budget limits

If using OpenRouter via the main API account:
- Set a monthly spending limit in your OpenRouter dashboard
- Configure alerts at 50%, 75%, 90% of limit

### Per-request safeguards

The bundle already has safeguards:
- `max_tokens=4096` prevents runaway responses
- `temperature=0.3` keeps output deterministic (less variance)
- No retries on failure (failures are reported, not auto-retried)

### Monitor token bloat

If you notice token usage climbing:
1. Check if input text is growing (user messages getting longer)
2. Validate that `max_tokens=4096` is appropriate for your task
3. Consider if summarization/filtering could reduce input size

## Cost reporting for billing

The accounting system supports chargeback by tenant/project/user:

```python
# Example (pseudocode): costs per tenant last 7 days
SELECT tenant, SUM(cost_usd) as total_cost, COUNT(*) as request_count
FROM accounting.cost_events
WHERE role = 'data-processor'
  AND created_at > NOW() - INTERVAL '7 days'
GROUP BY tenant
ORDER BY total_cost DESC;
```

Typical report columns:
- **Tenant** — which customer/team
- **Total cost** — sum of all requests
- **Request count** — volume
- **Avg cost per request** — cost_total / request_count
- **Model breakdown** — costs by model (if multiple models used)

For integration with your billing system, see your operator's accounting dashboard
or contact the platform team.

## FAQs

**Q: How are token counts calculated?**
A: OpenRouter returns `usage.prompt_tokens` and `usage.completion_tokens` from the LLM.
These follow OpenAI's tokenization (roughly 4 chars ≈ 1 token). There's no additional
tokenization by the bundle.

**Q: Can I see costs before making a request?**
A: Not directly. You must estimate based on input token count and model pricing.
The SSE "done" event gives the actual count post-request. Consider doing a small
test request first to calibrate.

**Q: Does the bundle support streaming to see costs as tokens arrive?**
A: Not currently. The bundle uses non-streaming `openrouter_completion()`.
Streaming support could be added (see [entrypoint.md](entrypoint.md) § "Use streaming").

**Q: How do I debug unexpectedly high costs?**
A: Check the SSE "done" event's `usage.total_tokens`. If higher than expected:
1. Verify the system prompt isn't excessive (it's minimal by design)
2. Check if user input includes large attachments
3. Confirm `max_tokens=4096` is appropriate
4. Try a different model to compare

**Q: What happens if OpenRouter is out of quota?**
A: The `openrouter_completion()` call fails and returns `{"success": false, "error": "..."}`.
The workflow emits an "error" SSE event. No charge is incurred for failed requests
(OpenRouter does not bill for failed API calls).

## Related documentation

- Bundle overview: [openrouter-data-README.md](openrouter-data-README.md)
- Entrypoint details: [entrypoint.md](entrypoint.md)
- Accounting system: `kdcube_ai_app/infra/accounting.py`
- OpenRouter API docs: https://openrouter.ai/docs/api/introduction