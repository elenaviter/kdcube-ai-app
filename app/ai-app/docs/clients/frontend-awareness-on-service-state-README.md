# Frontend Awareness Guide (Ingress/Proc, Rate Limits, and Serverless Behavior)

This guide is for **frontend developers** integrating with the chat platform.
The backend is **multi‑replica** and **serverless‑like**: instances can scale up/down at any time.

## Quick Takeaways

- **Never assume sticky connections.** Any request can land on any replica.
- **Expect drains / restarts.** SSE streams can close with a `server_shutdown` event.
- **Honor rate‑limits and backpressure.** Use exponential backoff and jitter.
- **Reduce request bursts.** Coalesce calls on page load and across tabs.

---

## A. Connection & Lifecycle Rules

| Area | What You’ll See | What You Must Do |
| --- | --- | --- |
| **SSE stream closes** | `server_shutdown` event, or connection drop | Reconnect with **backoff + jitter** (start 1–2s, cap 30s). |
| **Draining instance** | HTTP `503` with `{status:"draining"}` | Retry after short delay (1–3s + jitter). Don’t hammer. |
| **Scaled down instance** | SSE closes, no more events | Reconnect; don’t treat as fatal. |
| **No stickiness** | Requests land on different replicas | Keep **session_id** and **auth tokens** consistent per user. |
| **Multiple tabs** | Higher burst risk | Use shared storage and **coordinate** so only one tab does polling. |

---

## B. Rate Limits & Backpressure (Concrete)

### Typical errors

| Status | Meaning | What to do |
| --- | --- | --- |
| `429` | Rate limit exceeded | Backoff (2–5s) + jitter. Reduce burst. |
| `503` | Backpressure or draining | Backoff (1–3s) + jitter. Don’t retry immediately. |
| `401/403` | Auth missing/invalid | Refresh auth tokens / redirect to login. |

### Burst guidance

- **Page load** should not trigger more than 10–15 requests in <10s.
- **Chat send** should be serialized; avoid concurrent send bursts.
- **Polling** (if used) should be 5–10s minimum; prefer SSE.

---

## C. SSE (Streaming) Best Practices

### When opening SSE

- **Use a single SSE stream per user session**, not per component.
- If multiple tabs are open, **elect a leader tab** to keep SSE alive.

### On `server_shutdown`

- Treat it as **expected**.
- Close stream immediately.
- Reconnect after a short delay (1–2s + jitter).

### Keepalive handling

- The server emits `: keepalive` when idle. Ignore it.
- Don’t assume keepalive means new data.

---

## D. API Endpoints & Base URLs

| API | Base URL | Notes |
| --- | --- | --- |
| **Ingress (SSE + REST)** | `.../sse/*`, `.../api/chat/*` | Session + gateway checks |
| **Proc (Integrations)** | `.../api/integrations/*` | Long‑running bundle ops |

Frontends should support **two base URLs** (ingress/proc) in dev.
In prod, an Nginx or API gateway typically hides this and exposes a unified base.

---

## E. Retry & Backoff Recipes

### SSE reconnect

```
delay = min(30s, 2^attempt + jitter(0..1s))
```

### HTTP 429/503 retry

```
delay = base(1–3s) * 2^attempt + jitter(0..1s)
max_attempts = 5
```

Stop retrying after a few attempts and surface a UI message.

---

## F. Multi‑Tab Coordination

Recommended pattern:

1. Use `localStorage` or `BroadcastChannel` to elect a leader tab.
2. Only the leader maintains SSE.
3. Followers read updates from shared storage (or request on demand).

This avoids duplicate SSE streams and rate‑limit bursts.

---

## G. Auth Notes

- Non‑simple auth requires **access token + ID token**.
- SSE accepts auth tokens via **query params** if headers aren’t available.
- If using a proxylogin service, the browser should not handle raw tokens.

---

## H. Do/Don’t Summary

| ✅ Do | ❌ Don’t |
| --- | --- |
| Backoff + jitter on 429/503 | Fire retries in tight loops |
| Handle `server_shutdown` event | Treat it as a fatal error |
| Share SSE across tabs | Open SSE per widget or per tab |
| Use two base URLs in dev | Hardcode a single base forever |

---

## I. Troubleshooting Checklist

- 429 spikes? → reduce burst, add backoff, coalesce requests.
- 503 spikes? → capacity pressure, wait and retry.
- SSE drops? → check for `server_shutdown` and reconnect.
- No SSE events? → verify stream is open and user/session is valid.

