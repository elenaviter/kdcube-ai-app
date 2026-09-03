---
id: repo:kdcube-ai-app/app/ai-app/docs/service/cicd/delegated-management-service-README.md
title: "Delegated KDCube Management Service"
summary: "Documents the public, deployment-scoped service for delegated inspection, application-surface discovery, and exact application reload with Connection Hub admission and request-bound approval."
status: current-source; live-deployment-acceptance-pending
tags: ["service", "cicd", "management", "connection-hub", "delegated-authority", "idempotency"]
keywords: ["KDCube management resource", "application reload", "request-bound permit", "signed approval ticket", "effect ledger", "OAuth protected resource"]
updated_at: 2026-09-03
see_also:
  - repo:kdcube-ai-app/app/ai-app/docs/service/cicd/deployment-target-control-api-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/arch/delegated-authority-and-admission-README.md
  - repo:kdcube-ai-app/app/ai-app/docs/arch/security-and-trust-model-README.md
  - https://github.com/elenaviter/app-ecosystem/blob/main/docs/connection-hub/package/delegated-authority-and-admission.md
  - https://github.com/elenaviter/app-ecosystem/blob/main/docs/connection-hub/recipes/direct-protected-service.md
---
# Delegated KDCube Management Service

The delegated management service exposes a bounded public API for one running
KDCube tenant/project deployment. A caller authenticated through Connection
Hub can inspect deployment state, discover one application's declared
surfaces, or reload one exact declared application when its live delegated
card grants the corresponding operation.

These routes operate inside the running chat processor. Starting or recovering
a stopped deployment remains a local Docker, orchestrator, or infrastructure
operation because the stopped deployment cannot serve login, Connection Hub,
card resolution, or its management API.

## Protected Resource And Operations

The service constructs its resource from descriptor-owned deployment
coordinates:

```text
urn:kdcube:management:deployment:<RFC3986 tenant>:<RFC3986 project>
```

No operation accepts a tenant or project override.

| Operation | Method and route | Result |
| --- | --- | --- |
| `kdcube.management.deployment.inspect` | `GET /api/integrations/management/v1/deployment` | Bounded platform release, aggregate readiness, and declared-application preparation state. |
| `kdcube.management.application.surfaces.read` | `GET /api/integrations/management/v1/applications/{application_id}/surfaces` | Enabled API, MCP, widget, job, and messaging declarations for one exact application. |
| `kdcube.management.application.reload` | `POST /api/integrations/management/v1/applications/{application_id}/reload` | Reload evidence for one exact application from current descriptor authority. |

Every call requires an opaque Connection Hub bearer in `Authorization` and a
printable 1..256-character `Idempotency-Key`. Reload accepts the exact JSON
body `{}`. Inspect and surface discovery accept no body. Application IDs are
single exact identifiers; wildcard, path, URL, and reload-all forms are
rejected.

Protected-resource discovery is available at:

```text
GET /api/integrations/management/v1/.well-known/oauth-protected-resource
```

The document publishes the concrete resource, the deployment's Connection Hub
OAuth issuer, and the three operation routes. A missing bearer receives `401`
with that metadata URL in `WWW-Authenticate`.

## End-To-End Authority Flow

```text
 Connection Hub CLI                 running KDCube                  Connection Hub
 ------------------                 --------------                  --------------
 select endpoint
 discover protected resource ------> metadata route
 OAuth Authorization Code + PKCE ---------------------------------> OAuth/card flow
 receive opaque card-bound bearer <-------------------------------- user approval

 create exact request
 application_id + {}
 Idempotency-Key
        |
        v
 public management route
        |
        +-- contracts.py
        |   validate exact input
        |   construct deployment resource
        |   hash canonical request document
        |
        +-- admission.py
        |   keep bearer opaque
        |   sign resource + operation + invocation + digest
        |   with the management service HMAC secret
        |---------------------------------------------------------> direct admission
        |                                                           verify service proof
        |                                                           resolve current bearer
        |                                                           resolve live card/catalog
        |                                                           resolve invocation policy
        |<--------------------------------------------------------- allow or bounded denial
        |
        +-- service.py
        |   admission must allow before any runtime effect
        |
        +-- effect_ledger.py
        |   atomically reserve resource + operation + invocation
        |   same digest: replay/pending/unknown
        |   changed digest: conflict
        |
        +-- runtime.py
            inspect registry/readiness
            OR project enabled manifest surfaces
            OR call exact internal reload authority for application_id
        |
        +-- settle terminal result in shared Redis ledger
        v
 bounded result + authority revisions + replay evidence
```

Connection Hub resolves the current card and active catalog on every call.
Card edits, catalog changes, expiry, and revocation therefore affect the next
request without restarting either service.

The management runtime returns declared public coordinates only. It omits
source paths, repository credentials, secret references, environment values,
container identities, internal hostnames, and raw exceptions. Widget
coordinates use the authenticated `/widgets/{alias}` route.

## Request-Bound Reload Approval

Reload is configured as request-bound. `Always` remains reusable while the
live card and catalog authorize the operation. `Once` is valid only for the
exact browser-approved invocation and request digest.

```text
 unchanged reload request                     Connection Hub browser surface
 ------------------------                     ------------------------------
 direct admission denies
        |
        +-- Connection Hub signs a short-lived approval ticket containing:
        |     service, caller/card, resource, operation,
        |     invocation id, request digest, application id,
        |     card/catalog revisions, issued_at, expires_at
        |
        +-- management response returns consent_required,
            exact request fields, absolute ticket expiry, and authorization URL
                                                   |
 CLI opens the URL -------------------------------> user signs in as card owner
                                                   UI carries ticket opaquely
                                                   server verifies signature,
                                                   expiry, owner, displayed fields,
                                                   request and live revisions
                                                   before changing card/policy
                                                   |
                                                   +-- allow once:
                                                   |   issue exact request permit
                                                   +-- allow always:
                                                       commit reusable policy
 unchanged path/body/bearer/Idempotency-Key
 retries ----------------------------------------> exact authority is consumed
```

The approval ticket authenticates the browser handoff. It is not the permit
used at operation time. The request permit is durable Connection Hub authority
bound to the card and exact request. The KDCube effect ledger separately
prevents an admitted retry from performing the reload twice.

For a browser-resolvable reload denial, the public recovery has stable reason
`delegated_request_permit_required`. The management admission client extracts
the ticket from the returned URL, verifies it with the registered service
secret, compares it with the exact request and recovery fields, and derives
`expires_at` from that authenticated ticket. The UI treats the ticket as
opaque, and the Connection Hub server verifies it again before any card or
invocation-policy mutation.

## Idempotency And Failure States

The canonical request digest covers schema, concrete deployment resource,
exact operation, exact application ID, and body. The shared Redis effect
ledger is reserved only after current admission succeeds and before the
runtime adapter executes.

| Ledger state | Same invocation and digest | Runtime effect |
| --- | --- | --- |
| absent | reserve `effect_started` | execute once |
| `effect_started`, still within pending interval | `409 effect_outcome_pending` | do not execute |
| `effect_started`, older than pending interval | `409 effect_outcome_unknown` | do not execute automatically |
| `effect_completed` or `effect_failed` | return stored response with replay evidence | do not execute |
| any state with another digest | `409 invocation_id_conflict` | do not execute |

Terminal records have no time-based expiry in this first contract. This keeps
an accepted invocation replayable across processor restarts. Operators should
monitor ledger growth as part of the deployment's Redis capacity policy.

Admission and ledger failures stop before the runtime operation. A runtime
failure is settled as a terminal failed outcome so an uncertain retry cannot
repeat the effect. If the effect completed but settlement failed, the caller
receives `effect_outcome_unrecorded`; a later attempt resolves the retained
started record as pending or unknown rather than applying the effect again.

## Descriptor Configuration

The processor owns the management route and direct-admission client:

```yaml
management:
  delegated:
    enabled: true
    effect_pending_seconds: 120
    connection_hub:
      app_id: connection-hub@1-0
      service_id: kdcube-management
      service_secret_ref: connections.delegated_credentials.admission.services.kdcube-management.signing_secret
      admission_url: ""
      timeout_seconds: 10
```

An empty `admission_url` resolves to the Connection Hub app operation on the
same descriptor-owned processor port. A different URL is an explicit
descriptor choice.

The Connection Hub app descriptor registers the workload and marks reload as
request-bound:

```yaml
connections:
  delegated_credentials:
    admission:
      enabled: true
      services:
        kdcube-management:
          secret_ref: connections.delegated_credentials.admission.services.kdcube-management.signing_secret
          resources: ["urn:kdcube:management:deployment:*:*"]
          request_bound_operations:
            - kdcube.management.application.reload
          request_permit_ttl_seconds: 600
```

The active delegated catalog declares the same resource selector and exact
operations. The maintained default marks the resource admin-only. The shared
HMAC value exists only in the Connection Hub app secret provider under the
referenced path and contains at least 32 random bytes.

## Module Ownership

| Module | Responsibility |
| --- | --- |
| `management/contracts.py` | Resource grammar, exact input validation, canonical digest, result/error envelopes. |
| `management/routes.py` | Public HTTP methods, bearer/idempotency requirements, discovery, body rules, dependency assembly. |
| `management/admission.py` | Fresh signed direct-admission call to Connection Hub, plus signed recovery-ticket verification; bearer remains opaque. |
| `management/service.py` | Admission-first orchestration, recovery projection, audit evidence, replay behavior. |
| `management/effect_ledger.py` | Shared atomic reservation and guarded terminal settlement. |
| `management/runtime.py` | Bounded registry/readiness projection, enabled surface discovery, and exact application reload adapter. |
| Connection Hub `delegated_admission.py` | Workload verification, live card/catalog evaluation, signed browser recovery, request-permit consumption. |
| Connection Hub `invocation_policy` | Durable `Once`, `Always`, request-bound permit, and invocation replay authority. |

## Audit And Secret Boundary

Effect audit evidence includes decision ID, pairwise caller profile,
`access_id`, card/catalog/policy revisions, target, operation, exact
application, invocation ID, digest, and runtime instance. These identifiers
support diagnosis without carrying credentials.

Responses, logs, errors, consent URLs, and ledger records exclude bearer and
refresh tokens, authorization codes, PKCE verifiers, HMAC secrets and
signatures, provider credentials, session cookies, descriptor secret values,
and internal localhost authorization values. The signed approval ticket is a
short-lived authorization handoff and must not be logged.

The source contract is covered by focused route, runtime, service, ledger,
Connection Hub admission, request-ticket, request-permit, and widget tests.
Live acceptance still requires a staged runtime that loaded the exact tested
KDCube and Connection Hub sources, followed by one real approved reload and
its no-second-effect replay proof.
