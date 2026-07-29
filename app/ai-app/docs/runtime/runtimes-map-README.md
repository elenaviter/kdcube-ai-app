---
id: repo:kdcube-ai-app/app/ai-app/docs/runtime/runtimes-map-README.md
title: "The KDCube Runtimes: Map And Model"
summary: "The narrative map of KDCube runtimes and the explicit boundary contracts that connect them: serving and scheduled work, the Data Bus, app venvs, the agent harness, isolated execution, cross-runtime context, named services and MCP, accounting, subagents, and distributed serving."
tags: ["runtime", "architecture", "fences", "scheduling", "isolation", "distribution", "exec", "cross-runtime", "data-bus", "venv", "named-services", "mcp", "accounting"]
keywords:
  [
    "runtimes map",
    "fence model",
    "what crosses runtimes",
    "direct surface dispatch",
    "conversation event lane",
    "data bus runtime",
    "communicator",
    "events to the initiator",
    "bundle venv",
    "supervisor executor",
    "split execution profile",
    "cross-runtime context",
    "named service discovery",
    "mcp doors",
    "accounting across fences",
    "cluster critical section",
    "advisory locks",
    "observed file lock",
    "subagent fence",
    "distributed serving",
    "horizontal scaling",
  ]
updated_at: 2026-07-29
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/agent-in-the-runtimes-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/tenant-project-user-and-execution-boundaries-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/cross-runtime-context-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/fenced-runtime-bootstrap-and-reduce-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/harness/README.md
  - repo:kdcube-ai-app/app/ai-app/docs/exec/README-iso-runtime.md
  - repo:kdcube-ai-app/app/ai-app/docs/exec/runtime-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/exec/distributed-exec-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/bundle-runtime-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/bundle-transports-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/bundle-scheduled-jobs-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/bundle/bundle-venv-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/service/comm/README-comm.md
  - repo:kdcube-ai-app/app/ai-app/docs/service/comm/comm-recording-event-sinks-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/service/synch-mechanisms/critical-section-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/automations/automations-sdk-solution-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/namespace-services/discovery-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/kdcube-services/named-services-from-isolated-runtime-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/sdk/solutions/timeline/fork-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/economics/economic-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/accounting/accounting-README.md
---
# The KDCube Runtimes: Map And Model

KDCube is not one runtime. It is a set of genuinely different runtimes that
compose into one serving system: a processor that serves direct app surfaces
and schedules many kinds of work, a durable Data Bus, per-app virtual
environments, an agent harness that keeps framework-neutral conversation
state, an isolated execution family for generated code, provider apps reached
through named-service discovery over several bridges, and a portable-context
contract that lets a single request cross supported runtime boundaries without
losing its identity, authority, or accounting subject.

This page is the narrative map: what each runtime is FOR, what runs there,
**what crosses into it**, and what it guarantees. Throughout, **⚙ marks the
main lever** — the exact symbol to search for in the SDK to find the crossing
being drawn. The companion index with
the per-surface context-guarantee tables is
[Runtime Surfaces And Boundaries](README.md). The agent's-eye view of the
same system is [The Agent In The Runtimes Fusion](agent-in-the-runtimes-README.md).

## 1. One Primitive, Repeated: The Fence

KDCube repeats one design move at runtime boundaries: make the crossing
explicit. A **fence** declares what may cross, what the target reconstructs,
and which condition is enforced there. Unrelated process state does not ride
along implicitly.

Different fences enforce different properties. The split execution fence
removes network and credentials from generated code. The credential fence
keeps provider tokens on the trusted side and checks claims per operation. An
economic fence reserves allowance before integrated spend. A runtime-context
crossing carries an already-bound identity projection; it does not repeat the
original login on every hop. Structural isolation belongs specifically to the
split executor, not to every boundary called a fence.

```text
   source runtime                            target runtime
  +--------------------------+     |     +---------------------------+
  | work + bound context     |     |     | reconstructs trusted      |
  | only declared fields     | one typed | services or applies the   |
  | leave this runtime       | crossing  | check declared here       |
  |                          | ========> |                           |
  +--------------------------+     |     +---------------------------+
                                 FENCE
```

| Fence | Source side | Declared crossing or check | Target-side effect | Where in this map |
| --- | --- | --- | --- | --- |
| **Execution fence** | split executor: no network, credentials, or descriptor env | `tool_call(id, params)` over an authenticated socket | supervisor applies the exported catalog and policy | §8 |
| **Economic fence** | a turn that wants to run | a reservation that must clear admission | funding, quotas, the ledger | §11 |
| **Credential fence** | agent side: no provider token | an operation claim and bound actor/agent identity | Connection Hub resolves the credential on the trusted side for that call | §9 (the two gates) |
| **Data fence** (tenant/project/user) | trusted app operation | bound scope applied through storage helpers | scoped storage access; app code must preserve owner keys | every storage-using runtime |
| **Runtime-identity fence** | a supported hop to another trusted process or machine | JSON-safe portable context with already-resolved authority | target runtime reconstructs trusted services | §9 |
| **Subagent fence** | a spawned child agent | a serializable spec in, reduced side files out | the coordinating parent runtime | §12 |

Most fences are boundaries in **space** — two processes, two containers, two
machines. The economic fence is the same move in **time**: the decision is
placed one step ahead of the power, so a turn that would exceed its funding
never starts (§11).

The precise wording for each guarantee — which claim belongs to which
boundary — is fixed in
[Tenant, User, Authority, And Execution Boundaries](tenant-project-user-and-execution-boundaries-README.md)
under *Precise Language For Documentation And Publications*. In short: the
deployment layer isolates by namespace, the shared runtime enforces identity
and grants, and the **structural** isolation claim belongs to the
generated-code boundary.

## 2. The Map At A Glance: What Crosses The Runtimes

The runtimes are all different — different processes, different lifetimes,
different trust. What holds them together is that every arrow below has a
declared contract. Trusted runtime hops restore the full portable context
(identity, routing, resolved authority, accounting subject). Narrower targets
receive a boundary-specific projection: an app-venv helper receives serialized
call data, while the split executor receives reduced, non-secret execution
context and leaves the full portable spec with its trusted supervisor.

```text
                              THE WORLD
   chat clients | REST callers | webhooks (Telegram) | MCP clients |
                widget & scene browsers | other apps
      |                  |                    |
      | chat submit      | HTTP request       | MCP request
      | (SSE/Socket.IO)  | (@api auth/public) | (@mcp door)
      v                  v                    v
 +--------------------------------------------------------------------+
 | SERVING RUNTIME (proc)                                             |
 |                                                                    |
 |  integrations router ----- direct dispatch: binds identity         |
 |    (@api, @mcp, widgets)   (ExternalEventPayload), calls the       |
 |                            app surface in place, answers           |
 |                                                                    |
 |  conversation event lane - conversational work, ordered per        |
 |  schedulers                conversation; automations, @cron,       |
 |                            @on_job claim scheduled/queued work     |
 |                                                                    |
 |   +------------------------------------------------------------+  |
 |   |  THE APP (entrypoint + declared surfaces)                  |  |
 |   +------------------------------------------------------------+  |
 +-----|----------|------------|---------------|----------------|----+
       |          |            |               |                |
       | agent    | plain      | durable JSON  | named-service  | exec launch
       | turn:    | serializ-  | messages      | operation +    | payload;
       | context  | able args  | (streams,     | 4-field        | then only
       | + blocks | and        | claimed by    | discovery      | tool_call
       |          | results    | workers)      | descriptor     | over socket
       v          v            v               v                v
 +-----------+ +---------+ +-----------+ +----------------+ +----------------+
 | AGENT     | | APP     | | DATA BUS  | | PROVIDER APP   | | ISOLATED EXEC  |
 | HARNESS   | | VENV    | | durable   | | (same or other | | supervisor     |
 | timeline, | | cached  | | app-to-   | | process) via   | |  (trusted)     |
 | refs,     | | subproc | | app path  | | local | API |  | |   ⇄ authed     |
 | workspace | | per app | |           | | MCP | Data Bus | |     socket     |
 +-----------+ +---------+ +-----------+ | bridge         | | executor       |
                                         +----------------+ |  (empty)       |
                                                            | docker|Fargate |
                                                            +----------------+

  EACH BOUNDARY RECEIVES ITS DECLARED CONTEXT SHAPE:
    trusted runtime -> full portable context and reconstructed services
    app venv       -> serialized call data, including context passed by caller
    split executor -> reduced EXEC_CONTEXT + work/artifact/socket surfaces
  NEVER SERIALIZED AS LIVE OBJECTS: pools, clients, callbacks, large payloads
  SECRET-BEARING CONFIG: only trusted runtimes that require it; never the
    restricted executor

  ON COMMUNICATOR-ENABLED ARROWS: a comm spec crosses with the work, the far
    runtime rebuilds it, and events (deltas, steps, files, errors,
    measurements) stream back to the INITIATOR's connected surfaces (§5)
```

The full per-boundary "what crosses / what does not" tables are in
[Runtime Surfaces And Boundaries](README.md); the sections below walk the
boxes.

## 3. The Serving Runtime: Direct Surfaces And Scheduled Work

The processor (proc) hosts the apps and serves two different kinds of entry.

**Direct surfaces.** An app's declared `@api` operations (authenticated and
public — webhooks such as Telegram enter here), its `@mcp` doors, and its
widgets/static assets are served by the integrations router
(⚙ `call_bundle_op_public`, ⚙ `_dispatch_bundle_mcp_request` in
`apps/chat/proc/rest/integrations/integrations.py`) straight into the app: the router binds the request to an `ExternalEventPayload` (routing,
actor, user, authority) and calls the app surface in place, request/response.
No lane, no queue, no scheduler — the app answers directly, with full
identity bound around the call.

`public` means the platform route does not require an authenticated platform
session. It does not establish the caller's identity or authorization policy.
The app or integration may still carry or resolve an actor — for example from
a channel identity or a managed authority projection — and enforce its own
requirements before protected work runs.

**Scheduled work.** Conversational and queued work goes through ordering and
scheduling machinery:

| Work kind | Entry | Ordering and exclusivity |
| --- | --- | --- |
| **Chat turns** | `@on_reactive_event` → the shared `run()` entry | The conversation event lane reserves one accepted start batch for one turn; same-ingress siblings may share that batch; same-conversation turns serialize across workers while different conversations run in parallel |
| **Automations** | saved automation records; due-scanner + run-now | Scheduled and manual runs converge on one execution path; each execution is its own agent turn with its own conversation/turn identity |
| **Scheduled jobs** | `@cron(...)` methods, auto-discovered | Redis leases coordinate an active owner by declared span — `system`, `instance`, or `process`; competing ticks are skipped while the lease is valid, and failures are isolated |
| **Background jobs** | `@on_job` handlers | Claimed fairly off a Redis Stream across processors, deduplicated by key — a burst of webhook-triggered work spreads across the fleet instead of hammering one worker |

The split of responsibilities is deliberate: a public `@api` webhook answers
fast and **enqueues**; `@cron` decides **when** scheduled work is due;
`@on_job` handles ready work that has been enqueued; long-running or per-user
work never executes inside the scheduler tick. An automation is the
agent-shaped version of the same idea: due slots become queued executions
become background jobs become fresh agent turns.

**Worked micro-example — one Telegram message crosses both paths:**

```text
Telegram
   |  POST to the app's PUBLIC @api webhook route
   v
direct dispatch: the handler answers Telegram fast (200)
   |  ... and SUBMITS the message as conversational work
   |  ⚙ submit_telegram_turn(...)   (sdk/integrations/telegram)
   |  normalized ExternalEventPayload -> conversation event lane
   |  ⚙ enqueue_chat_task_with_lane_events_atomic
   v
the lane orders it; an agent turn runs on whichever worker takes it
   |  progress and the answer stream through the communicator (§5)
   v
the app's Telegram delivery posts the reply back into the chat
```

The webhook itself is the direct path (answer fast, never block a scheduler
tick); the conversation is the scheduled path; the reply rides the
communicator back out through the channel. One message, three runtimes, one
actor-and-authority lineage end to end: the Telegram actor remains explicit,
and a linked platform projection is carried when a protected boundary needs it.

Direct and scheduled entries do **not** share one payload or lifecycle. They do
share the runtime identity contract: each execution binds its actor, routing,
and any resolved authority projection before trusted work uses them. A direct
surface that submits conversational work creates the lane's
`ExternalEventPayload` at that crossing. The inbound surface matrix is in
[Bundle Transports](../sdk/bundle/bundle-transports-README.md).

**Delivery guarantee, not transactional promise.** The lane guarantees
at-least-once delivery: if a worker dies mid-turn, the reservation expires
and the event is redelivered elsewhere. An operation that already produced an
external side effect before the crash needs an idempotency strategy; the
platform pattern is a response record per message id, answered on redelivery
instead of re-executing. Exactly-once on external effects is the app's job;
the runtime gives it ordered delivery and per-message identity to build on.

App configuration on this runtime is descriptor-only — every knob is an app
property; there is no environment-variable configuration — and secrets
resolve through the settings/secrets APIs on the trusted side
([Content Properties And Secrets](../service/content-properties-secrets-mgmt-README.md)).

## 4. The Data Bus Runtime

The Data Bus is the **durable message runtime**: app-scoped JSON messages on
Redis Streams, surviving process death, ordered where ordering is declared,
and processed by handlers the processor owns.

- **Who publishes:** widgets and frontends (Socket.IO or the HTTP publish
  route), tools, internal services, and other apps — server-side publishing
  included. A browser surface can hand durable work to an app without any
  bespoke backend route.
- **Who consumes:** the app's `@data_bus_handler(...)` methods, claimed by
  processor-owned workers from the app-scoped stream.
- **Ordering:** `serial_per_partition` coordinates one active handler per
  partition while its lease is valid. It does not promise strict FIFO across
  retries, late claims, or dead-letter recovery; partitions can run in
  parallel.
- **At-least-once effect:** the stream can redeliver when a process dies
  between doing the work and acknowledging it. A specific relay may record a
  response per message id and answer redelivery from that record, but every
  mutating handler still needs durable idempotency and optimistic-concurrency
  checks at its storage authority.
- **What it is for:** the durable app-to-app path. It is the leg the
  named-services relay rides when generated code in an isolated runtime calls
  a provider app (§8, §10) — reaching the provider **while preserving the
  original request identity and consent checks**.
- **What it is not:** it is not the conversation timeline (conversation
  events land there only when an explicit bridge writes them) and not the
  live SSE relay (clients see live envelopes through the communicator, not by
  reading streams).

Contracts: [Bundle Runtime](../sdk/bundle/bundle-runtime-README.md) (Data Bus
handlers) and [Bundle Transports](../sdk/bundle/bundle-transports-README.md).

## 5. The Communication Runtime

The Data Bus is the durable path; the **communicator** is the live one — and
it is a runtime concern precisely because the work that needs to speak does
not stay in one runtime. A turn's events must reach the initiator whether
they were emitted in the processor, a venv subprocess's trusted caller, the
ISO supervisor, a subagent fence, or a remote Fargate task.

```text
communicator-enabled runtime doing the work
  (proc | supervisor | subagent fence | remote exec)
        |
        |  the comm SPEC crossed WITH the work (portable, JSON-safe);
        |  the far runtime REBUILDS a communicator from it
        |  ⚙ ChatCommunicator
        v
communicator: deltas | steps | files | errors | measurements
        |
        +--------------------+----------------------+
        |                    |                      |
        v                    v                      v
  PEERED to the        BROADCAST to the        RECORDED as selected
  initiating           user's connected        turn event artifacts;
  conversation/turn    surfaces (an event      reload hydrates the
  stream (live         like a conversation     durable client view
  SSE/Socket.IO        title reaches every     (citations, cost,
  envelopes)           open surface)           timing, panels)
```

- **The spec crosses, the object does not.** A live communicator is never
  serialized; the host exports a comm spec, and the child runtime rebuilds a
  `ChatCommunicator` bound to the same conversation and initiator. Recording
  selectors cross only when JSON-portable.
- **Peered and broadcast are both first-class.** Most events are peered —
  they belong to the initiating conversation and stream to its live clients.
  Some are broadcast to all of the user's connected surfaces, which each
  apply them by relevance (the conversation-title event is the worked
  example).
- **Live and durable meet at selected artifacts.** Recording policy chooses
  which communicator events are exported by
  ⚙ `comm.export_recorded_events()` and persisted with the turn. Reload does
  not replay the live stream byte for byte; it fetches durable timeline and
  stream artifacts and hydrates the same client-state model.
- **The one runtime with no communicator is the ISO executor** — by design.
  Generated code has no channel of its own; progress and results surface
  through the supervisor side, which does hold a rebuilt communicator.

Contracts: [Communication](../service/comm/README-comm.md),
[Comm Recording And Event Sinks](../service/comm/comm-recording-event-sinks-README.md),
[Chat Stream Events](../sdk/solutions/chat/chat-stream-events-README.md).

## 6. The App Venv Runtime

Each app can run selected helpers in its **own cached virtual environment** —
a real runtime with its own interpreter, its own installed dependencies, and
a subprocess boundary:

```text
proc (shared interpreter, shared event loop)
   |
   |  serialized call payload
   |  (plain args, including any context values the caller passes)
   v
@venv helper in the app's cached venv subprocess   ⚙ @venv(requirements=...)
   |  own interpreter, own deps from the app's requirements.txt
   |  blocking library calls cannot stall the shared event loop
   v
plain serializable result
```

- One cached venv per app id; the venv **rebuilds lazily when the referenced
  `requirements.txt` content changes** — no proc restart.
- The decorated callable is the boundary: serializable data in, serializable
  data out. Context needed by the helper must be represented in that call
  payload; a live request-context object is not transported into the child.
  Live proc objects — communicators, DB pools, Redis clients, tool registries —
  never cross; orchestration stays in proc.
- Use it for dependency-heavy leaf work and libraries that do not belong in
  the shared proc interpreter; the app keeps the platform's interpreter clean
  while carrying its own stack.

Reference: [Bundle Venv](../sdk/bundle/bundle-venv-README.md).

## 7. The Agent Harness Runtime

Between the serving runtime and any concrete agent framework sits the
harness: the framework-neutral machinery for **events** (resolving canonical
object refs under trusted context), **timeline** (the durable conversation
record as ordered blocks, with projections for the model, the chat client,
and external readers), and **workspace** (a fresh per-turn materialization
surface where durable refs become local bytes on demand).

The harness is what makes "the same conversation, any agent" true: the native
agent and a ported framework write and read the same block grammar, the same
ref namespaces, and the same turn workspace contract. Its own documentation
tree is [Agent Harness Runtime](harness/README.md). The harness owns the
platform conversation record; an integrated framework may still own a durable
checkpointer or reconstruct its own model-facing history.

## 8. The Isolated Execution Runtimes

Model-generated code gets its own runtime **family** — the most intricate
runtime in the platform, because it is itself composed of coordinated
runtimes: a trusted **supervisor** runtime, a restricted **executor**
runtime, and the transport/orchestration that binds them on one host or
across machines.

| Profile | What it is | What it guarantees |
| --- | --- | --- |
| in-memory | safe tools run in the current process | no isolation; for tools that need none |
| local subprocess | a child Python process on the same host | crash containment only — it inherits host network and environment |
| Docker combined (legacy) | supervisor and UID-dropped executor child in one container | filtered env and no network namespace for the child, but one shared mount namespace |
| **Docker split (reference)** | two sibling containers: trusted supervisor and restricted executor | executor has **no network**, a **read-only root filesystem**, narrow work/artifact/log/socket mounts, **no platform secrets and no descriptor payloads**; every tool call crosses an authenticated per-execution socket with peer-credential checks |
| Fargate / external | supervisor and generated-code child inside one remote task/container | the same logical supervisor/tool-call contract and snapshot transport, but **not** split Docker's separate-container mount boundary; assess task IAM, network, filesystem, child isolation, and snapshot transport |

```text
trusted processor
      |  exec launch payload (work/out surfaces, exported tool catalog)
      |  ⚙ exec_tools.execute_code_python | codegen_tools.codegen_python
      v
+----------------------+   authenticated     +----------------------------+
| supervisor           |  per-exec socket    | executor                   |
| descriptors, tools,  | <--- tool_call ---- | generated code             |
|                      | ⚙ agent_io_tools    |                            |
|                      |   .tool_call(...)   |                            |
| provider access,     | ---- result ------> | no network, read-only fs,  |
| network              |                     | no secrets, narrow mounts  |
+----------------------+                     +----------------------------+
```

The split profile is the execution fence made physical. The executor is the
most capable component in the system — it runs arbitrary code — and it is
deliberately the emptiest: nothing to take, nothing to reach. When generated
code needs a privileged operation, it asks across the socket; the supervisor
applies the exported tool catalog and policy, and provider credentials never
leave the trusted side. A named-service call from generated code continues
from the supervisor over the Data Bus relay (§4) to the provider app with the
original identity intact. Distributed execution changes how code is
transported and run; it does not change the logical result contract the agent
sees.

Mechanics, mounts, environment tables, and operations:
[ISO Runtime](../exec/README-iso-runtime.md),
[Isolated Code Execution Architecture](../exec/runtime-README.md),
[Distributed Execution — Fargate](../exec/distributed-exec-README.md).

## 9. Cross-Runtime Context: Reconstruction, Not Serialization

```text
processor --> async task --> trusted child/supervisor --> Fargate task
    |             |                    |                  |
    +---- full portable context across trusted hops -----+
          identity | routing | resolved authority | accounting

app venv: serialized call context, not live proc objects
split executor: reduced EXEC_CONTEXT; full portable spec stays in supervisor
live pools and clients are never serialized across the boundary
```

A single request may touch the processor, an async task, a worker thread, a
subprocess, a supervisor container, and a remote ECS task. Across supported
trusted-runtime hops, the full JSON-safe **portable context** carries request
identity (tenant, project, app, actor, user, roles), routing (session,
conversation, turn), authority carried **already resolved** (downstream code
reads the projection; it never re-derives who the actor is), the accounting
subject and dimensions, the app call context, and the named-service discovery
descriptor.

The final generated-code boundary deliberately narrows that room rather than
erasing context altogether. The trusted supervisor restores the full portable
spec. The split executor receives `EXEC_CONTEXT` with safe identifiers such as
tenant/project/user, conversation/turn, app, and execution ids, plus its narrow
work, artifact, log, and socket surfaces. It does not receive the full portable
spec, descriptor payloads, platform or provider credentials, secret-provider
material, or live services.

Live database pools, Redis clients, provider objects, callbacks, and large
payloads do not cross as serialized objects. A trusted target runtime
**reconstructs** them from validated configuration. The complete portable spec
can carry model-provider configuration, and a trusted Docker/Fargate supervisor
can receive descriptor payloads or resolve secrets needed by approved tools.
Those secret-bearing inputs do not enter the restricted executor. Identity is
never model-selectable: a model or tool argument can name an object, but it
cannot change the tenant, user, authority, or economics subject.

### The carried identity is what unlocks accounts

The execution context is not only *who is asking* — it is the key that lets
supported trusted runtimes link the work to the user's **connected accounts**
and this agent's **delegated grants** at the moment of use. The context binds
the resolved platform/grantor user and the acting agent's own identity
(`kdcube-agent:<app>:<agent>` for a hosted agent; a `dcr-…` client identity
for an external app), and a guarded operation resolves authority from both:

```text
execution context — bound at entry, carried to supported enforcement points
  user identity | acting agent identity | authority projection
        |
        v   at a guarded operation, trusted code checks TWO gates
Connection Hub
  gate 1: does THIS agent hold a grant for this operation
          (Delegated by KDCube — per user, per agent, revocable)
  gate 2: does a connected account of THIS user authorize the claim,
          with this agent bound to use it on that account
          (Delegated to KDCube — per account, per claim)
        |
        v
the Connection Hub BROKER resolves the credential on the trusted
side of the fence — for ONE call; the provider token never enters
the agent's runtime
⚙ ensure_claim(...)  (solutions/connections/delegated_to_kdcube/broker.py)
```

The resolution itself is the **broker's** job, on the trusted side of the
fence: it selects the account that holds the claim — fanning out across the
user's bound accounts or honoring an explicit account id — refreshes the
stored credential at resolution time, and scopes what the tool receives to
the claim being exercised. A resolution that cannot succeed comes back
structured, never as a bare failure: a reason (`connect_required`,
`claim_upgrade_required`, `reconnect_required`, `account_required`,
`agent_grant_required`), labeled candidates when several accounts match, and
a retry hint — the exact fix, addressed to the user and readable by the
agent.

When a guarded call runs on a supported trusted runtime, the bound identity
projection reaches that enforcement point with the work — a tool in proc, a
named-service call relayed through the isolated runtime's trusted supervisor,
or a provider operation on another app. Because identity is never
model-selectable, no prompt can point the resolution at another user's
accounts. Contracts:
[Tenant, User, Authority, And Execution Boundaries](tenant-project-user-and-execution-boundaries-README.md)
(Connection Hub rows of the enforcement matrix) and the Connection Hub
solution docs under `docs/sdk/solutions/connections/`.

Contract and field-level detail:
[Cross-Runtime Context](cross-runtime-context-README.md).

## 10. Named Services And MCP: Crossings Between Apps

A provider app is a runtime of its own — possibly another process, another
worker, even another machine. Reaching it is therefore a runtime crossing
with the same discipline as every other fence.

```text
agent / tool / generated code needs  ns:<operation>
        |
        v
named-service DISCOVERY — Redis-backed registry per tenant/project
  (what travels between runtimes is a 4-field descriptor:
   schema | backend | tenant | project — the directory itself never moves)
        |
        v
dispatch over the BEST AVAILABLE BRIDGE, auth context preserved:
  local in-proc call   (provider co-hosted in this processor)
  | app API bridge     (another process, same deployment)
  | MCP                (an MCP-speaking consumer or provider)
  | Data Bus relay     (durable; the leg from isolated runtimes)
        |
        v
provider app enforces its own schema, claims, and consent
```

MCP itself runs in **both directions** and each is a crossing:

- **As provider:** an app's `@mcp` doors are served through the integrations
  router (§3). The door's declared visibility, authentication, guards, and
  grants determine the caller context; protected calls arrive with bound
  identity rather than model-supplied authority.
- **As consumer:** remote MCP servers appear as tools in an agent's
  inventory; the MCP subsystem carries the calls out, subject to the agent's
  inventory and connection authorization.

MCP is one transport among the bridges, not the identity of the runtime.
Contracts: [Named Service Discovery](../sdk/namespace-services/discovery-README.md),
[Named Services From The Isolated Runtime](../sdk/solutions/kdcube-services/named-services-from-isolated-runtime-README.md),
[Bundle Transports](../sdk/bundle/bundle-transports-README.md).

## 11. Accounting Across The Fences

Spend control is a runtime concern because the calls that cost money happen
in **different runtimes** — and attribution must survive each supported
crossing on an integrated accounting path.

```text
turn admission — the economic fence, in TIME
  ⚙ EconomicsGuard  (async with EconomicsGuard(...): ...)
  estimate -> reserve against the payer's allowance
  (a turn that would exceed funding never starts)
        |
        v
the turn runs; metered calls report from WHEREVER they execute
  ⚙ track_llm | track_embedding | track_web_search  (infra/accounting):
  a model call in proc | an embedding inside a tool | a provider call
  brokered by the ISO supervisor for generated code | a search in a venv
  helper's trusted caller
        |     the accounting subject rides the portable context (§9),
        |     so every one of them lands on the same turn and payer
        v
settlement — actual usage reconciled against the reservation:
  unused hold released | overage absorbed and attributed to who caused it
```

The guarantee is scoped honestly: accounting follows **integrated** paths —
model, embedding, web-search, and participating custom calls. A trusted app
calling an arbitrary provider library directly is outside the meter; the
runtime enforces the economic events it can observe.

Mechanism and integration points:
[Economics](../economics/economic-README.md),
[Accounting](../accounting/accounting-README.md).

## 12. Subagent Fences And The Timeline Fork

Delegated agent work uses the same fence discipline at agent granularity:

```text
parent runtime
   |  enter scope, bind the CHILD's identity
   |  ⚙ react.delegate  (the native adapter's spawn tool)
   v
build the portable spec INSIDE the bound scope
   |
   v
child runs behind a fence
  (async task | thread | subprocess | docker | fargate)
  own workspace; timeline forked by value
   |
   v
reduce: side files (events, deltas, results) fold back into the parent
```

The generic contract: enter a scope, bind the child's identity **inside**
that scope, build the serializable spec after binding, run the child in its
own workspace behind a fence, then **reduce** the child's side files —
recorded events, deltas, results — back into the coordinator. The ordering
rule matters: a portable spec built before the scope is bound would carry the
parent's identity into the child.

Conversation state forks the same way ownership would predict: a subagent's
conversation is seeded with a **projection copy** of the parent's — working
summaries plus the in-progress turn — copied **by value**, so the two
timelines share no state afterward, with back-references in both directions
for clients that rebuild the thread view.

Contracts:
[Fenced Runtime Bootstrap And Reduce](fenced-runtime-bootstrap-and-reduce-README.md),
[Timeline Fork](../sdk/solutions/timeline/fork-README.md).

## 13. Cluster Critical Sections: When Work Must Not Parallelize

Distribution (§14) means many replicas attempting the same work at the same time:
several processors load the same app and would bootstrap the same schema,
several replicas would materialize the same shared bundle storage, every
worker sees the same due cron tick. Some work must not overlap. KDCube
coordinates it with three substrates, chosen by the resource being protected:

| Substrate | Guards | Mechanism |
| --- | --- | --- |
| **Postgres advisory transaction locks** | schema bootstrap and migrations in `on_bundle_load` | `pg_advisory_xact_lock` on a stable key; releases automatically on commit, rollback, or connection failure |
| **Redis locks** | cluster-wide coordination before shared work | `@cron` leader election by span (§3); the observed Redis lock for shared materialization — the key uses a **shared** segment and the owner identity lives in the **value**, so replicas contend on the same lock instead of each taking its own |
| **Observed file locks** | the shared-filesystem (EFS) mutation itself | `fcntl.flock` plus an in-process lock, with owner metadata written **inside** the lock file — readable during an incident |

The division of labor: Redis coordinates *before* filesystem work starts; the
file lock protects the filesystem mutation *itself*. Both are advisory — they
guard only participants that use the helper. Redis locks are expiring leases,
not proof of an exactly-once effect: if a lease is lost, old work may still be
running. Cron jobs and other side effects therefore remain idempotent, and
durable mutations use a transaction, revision check, idempotency key, or the
resource's own conditional-write primitive.

**Schema bootstrap** (⚙ `pg_advisory_xact_lock`) — the most common case. Several replicas load the same
app; DDL runs once:

```python
async with pg_pool.acquire() as conn:
    async with conn.transaction():
        await conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            f"{tenant}:{project}:{bundle_id}:schema:v1",
        )
        # check schema version, run idempotent CREATE/migrations,
        # record the version — short, bounded, no backfills here
```

**A shared build in bundle storage** (⚙ `observed_file_lock`) — the
canonical guarded-build pattern:
fast-path on a signature, acquire, **re-check under the lock**, build into a
safe location, verify readiness, write the signature **last**:

```python
from kdcube_ai_app.storage.observed_file_locks import observed_file_lock

def ensure_shared_index(self) -> Path:
    root = self.bundle_storage_root()
    signature = self.bundle_prop("knowledge.ref", default="local")
    if _current(root, signature):              # fast path, no lock taken
        return root / "my-index"
    with observed_file_lock(
        lock_path=root / ".my-index.lock",
        resource_id=f"{bundle_id}:my-index",
        operation="my.bundle.index.build",
        wait_seconds=300,
    ):
        if _current(root, signature):          # re-check under the lock
            return root / "my-index"
        build_index(root / "my-index")         # build into a safe location
        _verify_ready(root)
        _write_signature(root, signature)      # signature LAST
    return root / "my-index"
```

Async code that must not block the event loop uses
`observed_file_lock_async(...)`. The full contract — lock shape, acquire
flow, stale-owner recovery, the runtime's own uses (git bundle
materialization, prepared indexes, UI once-builds) — is in
[Synchronization Mechanisms](../service/synch-mechanisms/critical-section-README.md).

## 14. Distribution Is The Forcing Function

Every design above answers one operating condition: **a later turn may run on
another worker or another machine, and users share the serving
infrastructure.**

- Durable state is keyed by **conversation**, never kept in process memory.
  The agent is rebuilt fresh per turn; a worker-local singleton is ephemeral
  reuse, not a durable service. Memory is not authority.
- Users share workers; KDCube binds each execution to the actor and authority
  context established by its entry path. A public route does not require a
  platform session; an app or integration may still carry or resolve an actor,
  and protected boundaries can require and resolve more.
  Platform storage helpers apply tenant/project and
  app/user/conversation/turn scope; trusted app code must keep those owner keys
  intact. This is logical data scoping, while the structural isolation claim
  belongs to the generated-code boundary (§8).
- Horizontal scale follows because nothing required to continue a later turn
  lives only in one process. Adding workers adds capacity; the conversation
  lane keeps per-conversation order while unrelated work parallelizes.
- The at-least-once lane (§3) exists for the same reason: a machine can
  disappear mid-turn, and the work must land somewhere else without losing
  its place in the conversation.

The complete boundary map — deployment scope, shared machinery, identity
continuity, authority resolution, trusted app code, and the reusable
isolation primitives — is
[Tenant, User, Authority, And Execution Boundaries](tenant-project-user-and-execution-boundaries-README.md).

## The Levers At A Glance

Every ⚙ lever in this map, in one place — each is the exact symbol to search
for in the SDK for that crossing:

| Crossing | ⚙ Lever |
| --- | --- |
| Direct surface dispatch (§3) | `call_bundle_op_public`, `_dispatch_bundle_mcp_request` — `apps/chat/proc/rest/integrations/integrations.py` |
| Scheduled-work entries (§3) | `@on_reactive_event` → `run()` / `execute_core` · `@cron(...)` · `@on_job` |
| Telegram submit → lane (§3) | `submit_telegram_turn` · `enqueue_chat_task_with_lane_events_atomic` |
| Data Bus handling (§4) | `@data_bus_handler(...)` |
| Comm rebuild / recorded events (§5) | `ChatCommunicator` · `comm.export_recorded_events()` |
| App venv boundary (§6) | `@venv(requirements=...)` |
| Exec entry / the execution fence (§8) | `exec_tools.execute_code_python` \| `codegen_tools.codegen_python` · `agent_io_tools.tool_call(...)` |
| Credential resolution, the two gates (§9) | `ensure_claim(...)` — `solutions/connections/delegated_to_kdcube/broker.py` |
| Economic fence / metering (§11) | `EconomicsGuard` · `track_llm` \| `track_embedding` \| `track_web_search` |
| Subagent spawn (§12) | `react.delegate` |
| Cluster critical sections (§13) | `pg_advisory_xact_lock` · `observed_file_lock` / `observed_file_lock_async` |

## Reading Order

| Need | Read |
| --- | --- |
| The index and per-surface context guarantees | [Runtime Surfaces And Boundaries](README.md) |
| The agent's-eye view: one agent fed from any surface, two layers of state, native vs integrated | [The Agent In The Runtimes Fusion](agent-in-the-runtimes-README.md) |
| The full boundary and enforcement map | [Tenant, User, Authority, And Execution Boundaries](tenant-project-user-and-execution-boundaries-README.md) |
| What crosses a runtime hop | [Cross-Runtime Context](cross-runtime-context-README.md) |
| Generated-code isolation mechanics | [ISO Runtime](../exec/README-iso-runtime.md) and [Distributed Execution](../exec/distributed-exec-README.md) |
| Data Bus handlers and inbound surfaces in app code | [Bundle Runtime](../sdk/bundle/bundle-runtime-README.md) and [Bundle Transports](../sdk/bundle/bundle-transports-README.md) |
| Streaming events back to the initiator across communicator-enabled runtimes | [Communication](../service/comm/README-comm.md) and [Comm Recording And Event Sinks](../service/comm/comm-recording-event-sinks-README.md) |
| Scheduling entries in app code | [Bundle Scheduled Jobs](../sdk/bundle/bundle-scheduled-jobs-README.md) |
| The app's own dependency environment | [Bundle Venv](../sdk/bundle/bundle-venv-README.md) |
| Provider discovery and the bridges | [Named Service Discovery](../sdk/namespace-services/discovery-README.md) |
| Spend control and attribution | [Economics](../economics/economic-README.md) and [Accounting](../accounting/accounting-README.md) |
| Guarding once-only work across replicas | [Synchronization Mechanisms](../service/synch-mechanisms/critical-section-README.md) |
| Delegated child agents | [Fenced Runtime Bootstrap And Reduce](fenced-runtime-bootstrap-and-reduce-README.md) |
