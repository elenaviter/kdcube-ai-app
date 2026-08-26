---
id: repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/kdcube-services/named-services-from-isolated-runtime-README.md
title: "Named-Service Calls From Isolated Runtimes: The Data Bus Relay"
summary: "How an isolated runtime carries caller identity plus a typed admission selector to a target worker, resolves current authority there, and executes one named-service invocation."
status: current
tags: ["kdcube-services", "named-services", "data-bus", "isolated-runtime", "exec", "identity", "relay", "admission"]
updated_at: 2026-08-26
keywords: ["named-service relay", "NamedServiceAdmission", "delegated card", "access_id", "Data Bus", "isolated runtime"]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/cross-runtime-context-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/arch/security-and-trust-model-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/connections/delegated-cards/delegated-cards-README.md
---

# Named-Service Calls From Isolated Runtimes: The Data Bus Relay

This document explains, end to end, how a named-service call made from inside
an isolated execution runtime reaches its provider, runs under the original
user's identity, and returns its result. It defines every component it
mentions; nothing here assumes prior knowledge beyond "KDCube runs apps and
agents".

## The cast, defined

- **Proc** — the long-running KDCube server process. It loads app bundles,
  serves REST/socket traffic, and holds *live* objects: the bundle registry
  (loaded bundle instances), database pools, the session store.
- **Named service** — a namespace of objects and operations an app bundle
  serves to agents and external clients: `mail` (accounts, messages, send),
  `slack` (channels, messages, upload), `mem` (durable memories), and so on.
  A provider implements the namespace; a client calls it through a uniform
  grammar (`object.list`, `object.search`, `object.get`, `object.action`,
  `object.upsert`, ...). In a standard deployment the providers for mail,
  slack, and conversations are served by the always-running
  `kdcube-services@1-0` bundle.
- **Isolated execution runtime** — where agent-generated Python code runs. In
  the reference Docker `split` strategy it is two sibling containers:
  - the **executor**: runs the generated code itself. No network. It cannot
    call anything directly.
  - the **supervisor**: a trusted sibling container. It *has* network access
    (Redis, storage, the internet for
    integrations). Platform tools run here. The executor reaches it over a
    local Unix socket: when generated code calls
    `agent_io_tools.tool_call(...)`, the executor sends the request to the
    supervisor, and the supervisor executes the actual tool function.
- **The portable context room** — the JSON snapshot (`PORTABLE_SPEC_JSON`)
  that travels with every execution into the supervisor: the request identity
  (who is asking, with which roles and authority), the acting agent id, the
  named-service client policy, accounting context. The supervisor restores it
  at bootstrap, so SDK code inside the supervisor can ask "who is the current
  user?" and get the same answer the proc would give. Reference:
  `docs/runtime/cross-runtime-context-README.md`.
- **Data Bus** — KDCube's durable message lane over Redis Streams. A *stream*
  is an append-only log in Redis. Bundles declare handlers for message
  *subjects* (a subject is a routing string such as `canvas.patch`); the proc
  runs one worker per bundle that consumes that bundle's stream and invokes
  the matching handler. Delivery is tracked per consumer *group*: a message
  is handed to a consumer, and the consumer must *acknowledge* (ack) it after
  processing; unacknowledged messages are automatically re-delivered to
  another consumer after an idle threshold. The bus also keeps a **results
  stream** (each processed message's outcome) and a **dead-letter stream**
  (DLQ — messages that permanently failed, kept for inspection).

## The problem this solves

Inside the proc, a named-service call is an in-process function call: the
platform binds a *live caller* — an object that can invoke another loaded
bundle's registry directly — into the task context, and the named-service
tools use it. That object cannot be serialized: it only exists where the
bundles are loaded.

The supervisor is a different process (usually a different container). It has
the *policy* to make the call (the portable room carries it) and the
*discovery* to find the provider (reconstructed from Redis), but no live
caller — so, before the relay, a call like "send this email" from generated
code failed with `named_service_api_endpoint_unavailable`.

The relay closes that last gap using infrastructure that already exists: the
Data Bus (transport, retries, identity binding) and the provider bundle's own
worker (execution inside the proc, where a local peer-bundle caller is bound).
The message also carries a typed, non-secret admission selector. The target
validates that selector against the restored actor and resolves current
Connection Hub authority immediately before provider dispatch.

## Architecture

```text
ISOLATED RUNTIME (container)                 REDIS                        PROC (host server)
┌──────────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────────────┐
│ executor (no network)        │   │ Data Bus streams of the │   │ Data Bus worker for             │
│   generated code             │   │ provider bundle:        │   │ kdcube-services@1-0             │
│   agent_io_tools.tool_call   │   │                         │   │   claims message                │
│        │ unix socket         │   │  ...:messages  ◄────────┼───┼── (consumer group, ack, retry)  │
│        ▼                     │   │  ...:results   ─────────┼──►│   binds actor as request        │
│ supervisor (network)         │   │  ...:dlq                │   │   context                       │
│   named_services.object_     │   │                         │   │        │                        │
│   action(...)                │   │                         │   │        ▼                        │
│        │                     │   │                         │   │ @data_bus_handler               │
│        ▼                     │   │                         │   │ named_service_relay             │
│ endpoint bridge:             │   │                         │   │   idempotency replay?           │
│   live caller bound? ── no ─►│   │                         │   │   dispatch through the          │
│   RELAY request + typed      │   │                         │   │   admission → local Hub check   │
│   admission selector ────────┼──►│  XADD ...:messages      │   │   → registry (mail/slack/...)   │
│   wait for result ◄──────────┼───┤  poll ...:results       │◄──┼── write result + ack            │
└──────────────────────────────┘   └─────────────────────────┘   └─────────────────────────────────┘
```

The relay is two functions plus one handler:

| Piece | Where it runs | File |
| --- | --- | --- |
| `relay_named_service_call` | supervisor (client side) | `kdcube_ai_app/apps/chat/sdk/solutions/named_services_providers/relay.py` |
| endpoint-bridge fallback | supervisor | `.../named_services_providers/transports/api_client.py` (`_call_bundle_registry_endpoint`) |
| `handle_named_service_relay` + `@data_bus_handler` | proc, inside the provider bundle's worker | `relay.py` + `kdcube-services@1-0/entrypoint.py` (`named_service_relay`) |

## The round trip, step by step

Numbered from the moment generated code calls a named-service tool. Assume
the request is `object.action action=send` on namespace `mail`, and the
provider bundle is `kdcube-services@1-0` in tenant `demo-tenant`, project
`demo-project`.

```text
 1. executor       generated code:  await agent_io_tools.tool_call(fn=named_services.object_action, params={...})
 2. exec→sup       the executor serializes the call and sends it over the supervisor's unix socket
 3. supervisor     named_services.object_action runs: client policy check (allowed for this agent
                   on this namespace?) → construct delegated admission from the restored source
                   bundle, agent, client, and user identities → provider discovery (which bundle
                   serves `mail`?) → endpoint bridge. This is a new invocation even when another
                   named-service tool already ran in the same agent turn.
 4. supervisor     the bridge asks: is a live bundle-registry caller bound in this process?
                   In the proc: yes → direct in-process dispatch (the relay never engages).
                   In the supervisor: no → relay path.
 5. supervisor     relay_named_service_call builds a Data Bus message:
                     stream    kdcube:data-bus:demo-tenant:demo-project:kdcube-services@1-0:messages
                     subject   kdcube.named_service.relay.v1
                     payload   {"request": <provider request JSON>,
                                "admission": <typed non-secret selector>}
                     actor     the restored request identity: user_id, user_type, roles,
                               permissions, identity_authority, session_id, source bundle/agent
                     message_id / idempotency_key   one generated id (nsrelay_...)
                   and appends it (XADD). Then it polls the bundle's results stream for an
                   entry with its message_id, up to NAMED_SERVICE_RELAY_TIMEOUT_MS (default 90s).
 6. proc           the Data Bus worker for kdcube-services@1-0 (one per proc instance, already
                   running because the bundle declares handlers) claims the message from the
                   consumer group.
 7. proc           the worker performs its standard pre-dispatch steps — these are the same for
                   every Data Bus message, relay or otherwise:
                     - visibility: the bundle's allowed_roles and the handler's declared roles
                       are checked against the actor's roles
                     - idempotency: the handler declares idempotency=required, so a message
                       without an idempotency key is rejected outright
                     - context: it builds the request context (the same structure a live chat
                       request binds) FROM THE MESSAGE ACTOR, and binds it around the handler
                       call together with an auth context
                     - peer calls: it binds a local bundle named-service caller built from the
                       worker's Redis, database pool, and reconstructed request context
 8. proc           the handler (named_service_relay) runs inside the kdcube-services bundle
                   instance:
                     a. replay check — was this message_id already processed? If Redis holds a
                        recorded result under kdcube:data-bus:...:relay-done:<message_id>,
                        return it verbatim. Nothing executes twice.
                     b. rebuild the named-service request from the payload
                     c. validate the admission selector against the actor. Application admission
                        requires a trusted source bundle. Delegated bearer admission must match
                        the authenticated session's exact `access_id` and delegated client;
                        delegated agent admission must match its source bundle and agent.
                     d. resolve delegated admission through the local Connection Hub operation.
                        This reads the exact current card and active catalog once for this
                        invocation and derives the account scope. Application admission proceeds
                        under the validated trusted source-bundle policy.
                     e. bind that scope and dispatch through the bundle's OWN registry: the same
                        registry object the bundle serves to every other surface (REST, MCP). The
                        provider runs with an auth context read from the bound request context —
                        i.e., the relayed user.
                     f. reset the invocation scope and record the response in Redis under the replay key
                        (NAMED_SERVICE_RELAY_RESULT_TTL_SECONDS, default 15 min)
 9. proc           the worker writes the handler's return value to the results stream and
                   acknowledges the message.
10. supervisor     the poll in step 5 finds the result, unwraps the named-service response, and
                   returns it to the tool call — which returns it over the unix socket to the
                   executor, into the generated code's `resp` variable.
```

From the generated code's point of view nothing changed: the same tool, the
same request shape, the same response envelope. The transport switched
underneath.

## Session, identity, and admission preservation

The provider authorizes the relayed call exactly as if the user's agent had
called it directly: same user, same consent state, same connected-account
claims, and the exact delegated card authenticated at the entrance. Identity
crosses four boundaries:

```text
proc (original request)          supervisor                    bus message              proc (handler)
──────────────────────           ─────────────────             ─────────────            ─────────────────
REQUEST_CONTEXT bound   ──ship──►restored from the   ──copy──► actor field:   ──bind──► request context
for the chat turn:               portable room at              user_id, type,           rebuilt from the
user_id, user_type,              bootstrap; SDK                roles, perms,            actor by the bus
roles, permissions,              accessors answer              identity_auth,           worker; auth
identity_authority,              with this identity            session_id               context derived
session_id                                                                              from it
```

1. **Proc → supervisor.** When the proc prepares an isolated execution, it
   snapshots the bound request context into the portable room. The supervisor
   restores it at bootstrap. This is the same mechanism every context-aware
   tool relies on; the relay adds nothing here.
2. **Supervisor → message.** `relay_named_service_call` reads the restored
   context and copies the user fields into the message's `actor`. It refuses
   to publish without one: a runtime with no bound identity gets
   `named_service_relay_identity_missing` instead of an anonymous call.
3. **Message → handler.** The bus worker builds the request context from the
   actor — the identical code path every Data Bus handler gets (a Telegram
   automation, a canvas patch, and a relay call all bind identity the same
   way) — and binds it, plus an auth context, around the handler invocation.
4. **Handler → provider.** The handler dispatches with an auth context read
   from the bound request context. The provider then does what it always
   does: resolve the user's connected accounts, check consent claims
   (`mail:send`, `slack:files:write`, ...), and refuse with the standard
   consent errors when the user has not approved something. Consent
   escalation from a relayed call looks exactly like consent escalation from
   a direct call.

Identity and admission are related, separate contracts. The actor says who
caused the work. The admission selector says which authority regime owns this
one named-service invocation:

| Mode | Relay selector | Target action |
| --- | --- | --- |
| Application | Audit source; the actor carries a trusted source bundle. | Validate the source bundle and dispatch under application authority. |
| Delegated bearer | Exact `access_id`, delegated `client_id`, grantor/delegate binding, and expiry metadata. | Match it to `identity_authority.delegated_card_binding`, then ask Connection Hub for that exact current card and active catalog. |
| Delegated agent | Trusted source bundle and agent ids plus delegated client and grantor ids. | Match it to the actor, then ask Connection Hub for the current agent card and active catalog. |

The selector contains identifiers and provenance. Tokens, credentials, raw
cards, catalog documents, and account scopes remain in their owning trusted
services. The selector is a platform-owned sibling of the provider request;
`NamedServiceRequest.context` remains provider-visible diagnostic context. The
Hub response supplies a sanitized invocation decision and account scope. The
handler binds that scope while the provider is invoked and restores the prior
task context afterward.

For a delegated-client bearer session,
`identity_authority.delegated_card_binding` retains the exact authenticated
card identity. That is session identity rather than a cached authorization
decision: each new invocation queries current Hub state, so card edits,
revocation, expiry, and current-catalog changes apply to the next call.

The session id rides along in the actor so that anything session-scoped
(conversation lane events, reply routing) keeps working; the relay does not
create sessions and does not elevate anything — the actor's roles are the
roles the original request carried.

**Trust model, stated plainly.** The bus worker trusts the `actor` field of
messages in its stream. That is the Data Bus's existing model: the streams
live in Redis, and only platform components hold Redis credentials — a party
that can write these streams already operates inside the platform's trust
perimeter. The relay inherits this model; it neither weakens nor strengthens
it. If the bus later gains per-message authentication (for example, the
Connection Hub's federated Data Bus tokens, which materialize a verified
external identity as a session plus a signed, Redis-registered, expiring
token), the relay picks it up like every other subject.

## Delivery semantics: at-least-once, exactly-once effect

The bus delivers **at least once** — not because anything publishes twice,
but because delivery and acknowledgment are decoupled:

- a consumer claims a message, the handler runs, and the proc dies before the
  ack → the message stays in the group's pending list → the auto-claim pass
  hands it to another consumer → the handler runs again;
- the worker's retry path re-publishes a message whose handler raised — even
  if the failure happened *after* the side effect (the email left, then the
  result write failed).

For notifications this is harmless; for `send`-class actions a duplicate is
user-visible. The relay therefore makes the *effect* exactly-once:

- the client sets `idempotency_key = message_id` (the handler declares
  `idempotency=required`, so key-less messages are rejected before dispatch);
- the handler's first action is a replay check against the recorded result
  (`...:relay-done:<message_id>`); a redelivered or retried message answers
  from the record without touching the provider;
- the record expires after `NAMED_SERVICE_RELAY_RESULT_TTL_SECONDS`
  (default 900) — far beyond the client's wait window.

## Failure modes

| What happened | What the caller sees | Why |
| --- | --- | --- |
| No provider worker answered in time | `named_service_relay_timeout` (504) | proc restarting, worker busy, or bundle disabled; safe to retry — the idempotency record protects a late duplicate execution's effect |
| Runtime carries no identity | `named_service_relay_identity_missing` (401) | the portable room was built without a bound request context; nothing anonymous is ever relayed |
| Relay carries no typed admission selector | `named_service_relay_admission_missing` (rejected) | every invocation selects its authority positively |
| Selector and restored actor disagree | `named_service_relay_admission_invalid` (rejected) | bearer and agent identity are validated before Hub or provider dispatch |
| Delegated card is missing, expired, revoked, or belongs to another client | structured delegated-card denial (403) | Connection Hub resolves current authority for this exact invocation |
| Current delegated catalog cannot be resolved | structured temporary-unavailability denial (503) | delegated admission requires known current authority before provider dispatch |
| Actor fails bundle/handler role visibility | `bundle_not_visible` / `handler_not_visible` (rejected result) | the standard bus visibility checks, applied to the relayed actor |
| Handler kept failing past the retry budget | error result + message in the DLQ stream | inspect `kdcube:data-bus:<t>:<p>:<bundle>:dlq` |
| Provider refused (consent, claims, account) | the provider's own error envelope, unchanged | the relay is a transport; consent semantics stay the provider's |

## Knobs

| Setting | Default | Meaning |
| --- | --- | --- |
| `NAMED_SERVICE_RELAY_TIMEOUT_MS` | 90000 | how long the client polls the results stream |
| `NAMED_SERVICE_RELAY_RESULT_TTL_SECONDS` | 900 | how long a recorded response answers redeliveries |
| `DATA_BUS_HANDLER_TIMEOUT_SECONDS` | 120 | worker-side cap on one handler invocation |
| `DATA_BUS_MAX_RETRIES` | 5 | worker retry budget before the DLQ |

## What A Provider App Must Do To Serve Relayed Calls

The app must already own and explicitly publish the provider through its
named-service registry. Discovery publication is separate from relay
transport; see
[Discovery Registry](../../namespace-services/discovery-README.md#ownership-and-publication-invariant).
Then declare the relay handler:

```python
from kdcube_ai_app.apps.chat.sdk.solutions.named_services_providers.relay import (
    NAMED_SERVICE_RELAY_SUBJECT,
    handle_named_service_relay,
)

@data_bus_handler(subject=NAMED_SERVICE_RELAY_SUBJECT, idempotency="required")
async def named_service_relay(self, ctx, message):
    return await handle_named_service_relay(ctx, message)
```

`kdcube-services@1-0` ships this handler for its published providers, so mail,
Slack, and conversations are relay-reachable out of the box. An owner app
serving another namespace adds the same method to make its already-published
provider callable from isolated runtimes.

## Related

- `docs/runtime/cross-runtime-context-README.md` — the portable context room:
  what identity/policy travels into isolated runtimes and how it is restored.
- `docs/sdk/bundle/bundle-interfaces-README.md` — the `@data_bus_handler`
  declaration surface.
