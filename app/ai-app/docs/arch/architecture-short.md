---
id: repo:kdcube-ai-app/app/ai-app/docs/arch/architecture-short.md
title: "Architecture Short"
summary: "Concise current architecture of KDCube: one tenant/project deployment, directional app surfaces, ordered conversation eventing, app Data Bus, authority, storage, isolation, and accounting."
status: current
tags: ["arch", "architecture", "overview", "apps", "runtime"]
updated_at: 2026-08-13
keywords: ["KDCube architecture", "app surfaces", "as provider", "as consumer", "conversation event lane", "isolated execution"]
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/arch/security-and-trust-model-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/arch/control-plane-web-app-README.md
  - repo:kdcube/app/ai-app/docs/arch/application-hosted-websites-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/arch/architecture-of-what-we-built-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/arch/architecture-of-what-you-build-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/arch/architecture-long.md
  - repo:kdcube-ai-app/app/ai-app/docs/runtime/tenant-project-user-and-execution-boundaries-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/service/comm/conversation-event-bus-and-data-bus-README.md
---
# KDCube Architecture: Short Map

KDCube is an open-source AI application framework with an integrated,
self-hosted production runtime. Builders declare apps, surfaces, dependencies,
identity rules, execution policy, and storage contracts. The runtime serves and
enforces those declarations under concurrency, failure, security, and cost
constraints.

Application backend code is trusted deployment code; generated-code isolation
is a separate, profile-dependent boundary. Read
[Security And Trust Model](security-and-trust-model-README.md) before treating
this architecture map as a security claim.

## Scope

One running deployment is bound to one effective `tenant/project` and may serve
many users and apps concurrently. PostgreSQL, Redis, object storage, and shared
filesystem infrastructure may be dedicated or shared; shared deployments keep
tenant/project data separated through schemas, namespaces, and prefixes.

```text
deployment scope       tenant + project
request scope          actor + user + authority
product-work scope     app + conversation + agent + turn
```

## System Map

```text
browser | channel | webhook | REST client | MCP client
                          |
                          v
                proxy and public routing
                          |
                          v
              ingress and request context
       resolve route | verify surface-required proof
 bind tenant/project | actor/user/authority when applicable
 validate payload/visibility/limits/budget | admit/upload/queue
                          |
          +---------------+----------------+
          |               |                |
          v               v                v
 conversation lanes   Data Bus        direct app/API
 + wake queues        streams         operations
          |               |                |
          +---------------+----------------+
                          |
                          v
                  proc / processors
             load app and invoke surface
                          |
       +------------------+-------------------+
       |                  |                   |
       v                  v                   v
 app/agent code     trusted SDK/tools   isolated execution
 UI/API/jobs        identity/storage    supervisor + executor
 providers          hosting/economics
       |                  |                   |
       +------------------+-------------------+
                          |
                          v
        configured stores and external providers
```

## Apps Have Two Directions

An app can provide surfaces, consume surfaces, or do both:

```text
surfaces.as_provider                 surfaces.as_consumer
--------------------                 --------------------
what this app exposes                what this app may call

API and operations                   per-agent Python tools
widgets and main view                MCP server connections
MCP endpoints                        MCP tool allow-lists
default chat intent                  named-service namespaces
visibility and managed auth          UI resolvers/materialization
```

Provider policy does not grant the app consumer access. Consumer wiring does
not republish another service as this app's own surface. Each agent can have a
different inventory under `surfaces.as_consumer.agents.<agent_id>`.

Apps may expose any honest combination of REST, UI, chat, MCP, named services,
Data Bus handlers, integrations, jobs, websites, or ReAct. None is universally
required.

## Browser And Sites

The browser reads `/api/cp-frontend-config` and follows its provider-neutral
auth and app-routing contract. `/profile` is the source of logged-in state.

An app may serve widgets, a main view, the reserved default chat, or a complete
website. Website declarations compile into a validated
`ApplicationSiteCatalog`; Redis distributes catalog generations and each proc
routes from an immutable in-memory snapshot. One app contributes at most its
one main-view site; one deployment can serve many such app-owned sites.

## Conversation Work

One ordered event lane is keyed by:

```text
tenant + project + user + conversation + agent
```

Reactive ingress atomically writes one prepared batch to the lane and admits one
bodyless wake to the processor queue. Non-reactive events enter only the lane.
The lane sequence defines order; the queue schedules work. One accepted start
batch, possibly containing same-ingress sibling events, starts one app turn.
Native ReAct can fold eligible later events through owner-fenced lane handling.
The current run-to-completion adapter consumes only its start batch, so later
events wait for another turn. A custom adapter must wire live folding
explicitly. Unconsumed `event.user.steer` expires and never becomes future work.

Conversation Event Bus events carry prompts, attachments, followups, steers,
approvals, callbacks, and future-turn context. The Data Bus carries app-owned
domain work independent of chat. An app explicitly submits conversation events
when Data Bus work should enter or wake a conversation.

Streaming delivery is a third mechanism. SSE relay state is scoped by
`tenant + project + session_id`; `stream_id` selects one connection within that
set.

## Identity And Authority

Ingress resolves the route, verifies the proof required by its surface policy,
and binds the applicable tenant/project, actor/user, authority, routing, and
lineage context. Public routes need no normal platform session, though they may
carry identity or require app-specific proof.
Scheduled and internal work starts from system or already-bound context.
Connection Hub owns the platform authority/provider registry, connection
edges, authority projection, connected external accounts, and delegated
credentials.

```text
external account -> KDCube
  delegated to KDCube: trusted tools may use approved provider claims

KDCube -> external operator
  delegated by KDCube: automation receives bounded resource/operation grants
```

These directions are separate. Raw channel proof is not platform login, and a
bearer token is a handle to server-side grant/session state rather than the
authority source itself.

## Cross-Runtime And Isolated Execution

Request identity and authority travel through async, thread, subprocess,
supervisor, app-call, and Data Bus boundaries in the JSON-safe portable
`comm_ctx`. It carries situation and policy, not credentials, blobs, or live
handles. A trusted runtime bootstrap is separate and may contain descriptors,
model configuration, and selected provider keys needed to rebuild trusted
services. The restricted split executor receives a reduced execution ticket,
not that full trusted payload or platform/provider credentials.

For generated code, model-proposed refs and paths are untrusted locators. A
trusted resolver binds tenant/project/user/authority before materializing bytes
into a sparse workspace. In split Docker execution, the executor has no network
and receives no platform storage root, app storage root, descriptors, or
provider credentials. Approved tools execute in the trusted supervisor under
the carried request identity.

The same workspace and ISO-runtime contracts can support KDCube ReAct or an
externally authored agent such as LangGraph.

Hosted framework graphs are also turn-scoped. A graph is built for the current
bound user/conversation turn and discarded afterward; shared checkpointer and
domain stores carry continuity across workers. This is separate from the ISO
executor, which bounds generated code invoked by the graph or ReAct harness.

## Storage And Economics

KDCube does not pretend all state belongs in one store. Descriptors and secret
references, PostgreSQL records, Redis lanes/queues/catalogs/auth records, app
filesystem storage, artifact storage, and user-scoped provider secrets have
different owners and durability contracts.

Economics-aware work follows `verify -> reserve -> run -> settle`. Tracked calls
are attributed to request lineage such as user, app, conversation, turn, flow,
provider, and model.

## Deployment

The contracts run in local, Docker Compose, process/container, and ECS/Fargate
profiles. Docker split is the reference production profile for untrusted
generated code: supervisor and executor are separate containers. The current
Fargate generated-code profile instead runs a trusted supervisor and a
UID-dropped, network-isolated child in one remote task/container, using filtered
state and snapshot transport; it is not the split Docker mount boundary.
Deployment-specific ports, task sizes, IAM, DNS, TLS, and storage choices
belong to deployment descriptors and operations docs, not this stable map.

## Read Next

- [Control Plane Web App](control-plane-web-app-README.md)
- [Architecture Of What We Built](architecture-of-what-we-built-README.md)
- [Architecture Of What You Build](architecture-of-what-you-build-README.md)
- [Architecture Long](architecture-long.md)
- [Application-Hosted Website Architecture](application-hosted-websites-README.md)
- [Tenant, Project, User, Authority, And Execution Boundaries](../runtime/tenant-project-user-and-execution-boundaries-README.md)
- [Conversation Event Bus And Data Bus](../service/comm/conversation-event-bus-and-data-bus-README.md)
