# Economics Model (Control Plane)

This document is the authoritative description of the current economics model and how it is enforced at runtime.
It replaces the older usage notes and reflects the production bundle flow and control‑plane schema.

Runtime entrypoint:
- [entrypoint_with_economic.py](services/kdcube-ai-app/kdcube_ai_app/apps/chat/sdk/solutions/chatbot/entrypoint_with_economic.py)

## Scope

The economics subsystem covers:

- Rate limits and quotas (requests, tokens, concurrency)
- Funding lanes (subscription budget, project budget, wallet credits)
- Reservation semantics for correctness under concurrency
- Accounting and cost attribution
- Subscription period management and rollovers

## Core Concepts

### Role (economics role)

Role is the funding access decision, not the quota policy. The gateway can override the authenticated role based on economics state.

Role resolution is applied in the gateway and stored in the session:

- If `privileged` or `admin`, role stays privileged.
- Else if active subscription or wallet credits exist, role becomes `paid`.
- Else role remains `registered`.

This overrides the session `user_type` (the role used by the runtime entrypoint). The session key uses the resolved role:

- `...:paid:<user_id>`
- `...:registered:<user_id>`

### Plan (quota policy identity)

Plan (`plan_id`) is the quota policy identity used by the rate limiter. It is distinct from role.

Plan is resolved in the entrypoint at request time. The active plan determines base quotas via `plan_quota_policies`.

### Funding Sources

Funding sources are money or tokens used to pay for requests:

- Subscription period budget (per‑month balance, USD)
- Wallet or lifetime credits (token bucket, USD‑quoted)
- Project budget (tenant/project balance, USD)

Role determines which funding sources are allowed, while plan determines rate limits.

## Plan Resolution (Runtime)

Plan resolution is performed in the entrypoint using the following priority:

1. `privileged/admin` → `admin`
2. `anonymous` → `anonymous`
3. active subscription → `subscription.plan_id`
4. wallet only → `payasyougo`
5. paid role without subscription → `payasyougo`
6. default → `free`

Visual summary:

```mermaid
flowchart TD
  A[Request arrives] --> B{Role}
  B -- admin/privileged --> P[plan_id = admin]
  B -- anonymous --> N[plan_id = anonymous]
  B -- other --> C{Active subscription?}
  C -- yes --> S[plan_id = subscription.plan_id]
  C -- no --> D{Wallet credits?}
  D -- yes --> W[plan_id = payasyougo]
  D -- no --> E{Role is paid?}
  E -- yes --> W
  E -- no --> F[plan_id = free]
```

## Where Limits Come From (Plan Quotas)

Plan quotas are stored in the control plane table `plan_quota_policies`.

**Scope and window semantics (important):**
- Quotas are enforced **per tenant/project** (global across bundles).
- Hourly token limits use a **rolling 60‑minute** window (minute buckets).
- Monthly limits use a **rolling 30‑day** window anchored to the user’s first usage **per tenant/project**.
- Daily limits use **calendar day (UTC)**.
- Total requests do not reset.

Accounting and spend are still recorded **per bundle** for reporting, but quota enforcement is global per tenant/project.
Global quota counters use bundle id `__project__` in Redis keys (subject_id already encodes tenant/project).

Seeding flow:

- A master bundle calls `ensure_policies_initialized()`.
- The entrypoint seeds `plan_quota_policies` from `app_quota_policies` only if records are missing.
- After initial seeding, updates should be made in the admin UI or directly in the table.
- Runtime always prefers the DB policy, with a fallback to `app_quota_policies` if a plan is missing.

```mermaid
flowchart TD
  A[Master bundle starts] --> B[ensure_policies_initialized]
  B --> C{plan_quota_policies empty?}
  C -- yes --> D[Insert defaults from app_quota_policies]
  C -- no --> E[No changes]
  D --> F[Runtime uses DB policies]
  E --> F
  F --> G[Fallback to app_quota_policies only if DB missing]
```

## Funding Lanes and Reservation Semantics

### Tier lane

Tier lane is the normal path when rate‑limit admit succeeds.

Funding for tier lane:

- Active subscription → reserve from subscription period budget.
- Registered role (no subscription) → reserve from project budget.
- Privileged bypass → no pre‑check; project budget is charged after run.

### Paid lane

Paid lane is used when tier admit is denied or when tier funding cannot be reserved.

Funding for paid lane:

- Active subscription → reserve subscription budget for the full estimated cost.
- Otherwise → reserve wallet credits (lifetime tokens).

### Reservation types

- Rate limiter token reservation (Redis) for tier lane
- Subscription reservations in `user_subscription_period_reservations`
- Project budget reservations in `tenant_project_budget_reservations`
- Wallet reservations in `user_token_reservations`

Reservations are committed or released after execution and accounting. Expired reservations are reaped automatically.

## Decision Tree (Role → Plan → Funding)

```mermaid
flowchart TD
  A[Request] --> B[Resolve role]
  B --> C[Resolve plan_id]
  C --> D[Load plan quota policy]
  D --> E{Rate‑limit admit?}
  E -- No --> F{Personal funding available?}
  F -- No --> X[Deny: rate limit]
  F -- Yes --> P1[Paid lane]

  E -- Yes --> T1[Tier lane]

  T1 --> S{Active subscription?}
  S -- Yes --> SB[Reserve subscription budget]
  S -- No --> R{Role allows project budget?}
  R -- Yes --> PB[Reserve project budget]
  R -- No --> P1

  P1 --> PF{Active subscription?}
  PF -- Yes --> PS[Reserve subscription budget]
  PF -- No --> PW[Reserve wallet credits]

  SB --> RUN[Execute]
  PB --> RUN
  PS --> RUN
  PW --> RUN

  RUN --> ACC[Accounting]
  ACC --> COMMIT[Commit reservations and spend]
```

## Subscription Periods and Rollovers

Subscription budgets are per billing period.

- Each period is keyed by `(tenant, project, user_id, period_key)`.
- Top‑up is once per period by default (idempotent).
- Periods are closed at the end date.
- Unused balance is rolled into project budget.

Maintenance entry points:

- `SubscriptionManager.sweep_due_subscription_rollovers(...)`
- Control plane endpoint: `POST /subscriptions/rollover/sweep`
- Reservation reaper: `POST /subscriptions/reservations/reap-all`

## Data Model (Tables)

Authoritative schema:

- [deploy-kdcube-control-plane.sql](services/kdcube-ai-app/kdcube_ai_app/ops/deployment/sql/control_plane/deploy-kdcube-control-plane.sql)

Key tables:

- `plan_quota_policies` — base policy per plan_id
- `user_tier_overrides` — temporary overrides
- `user_lifetime_credits` — wallet credits
- `user_token_reservations` — wallet reservations
- `tenant_project_budget` — project money balance
- `tenant_project_budget_reservations` — project budget holds
- `tenant_project_budget_ledger` — project budget ledger
- `subscription_plans` — plan catalog and Stripe price mapping
- `user_subscriptions` — subscription metadata
- `user_subscription_budget_settings` — per user subscription config (overdraft)
- `user_subscription_period_budget` — per period subscription balance
- `user_subscription_period_reservations` — subscription holds
- `user_subscription_period_ledger` — subscription ledger
- `external_economics_events` — idempotency and audit for external/internal economic operations

## Accounting and Costing

Accounting events are emitted by service wrappers (LLM calls, web search, etc.).
Events are aggregated per turn and converted to USD using the reference model.

Reference model conversion is used for:

- Wallet credit conversion (USD → tokens)
- Token balance display (tokens → USD)
- Estimation of request cost for reservations
