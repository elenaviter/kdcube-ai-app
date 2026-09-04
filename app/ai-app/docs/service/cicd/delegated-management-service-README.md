---
id: repo:kdcube-ai-app/app/ai-app/docs/service/cicd/delegated-management-service-README.md
title: "Delegated KDCube Management Service"
summary: "Documents KDCube's deployment-scoped management service: Card-governed automation, exact secret operations through the selected provider, and browser-approved one-use descriptor export."
status: current-source; live-deployment-acceptance-pending
tags: ["service", "cicd", "management", "connection-hub", "delegated-authority", "idempotency"]
keywords: ["KDCube management resource", "application reload", "request-bound permit", "secret provider", "secret descriptor export", "human approval", "PKCE", "effect ledger", "OAuth protected resource"]
updated_at: 2026-09-04
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
surfaces, reload one exact declared application, and manage one exact secret
through the deployment-selected provider when its live delegated card grants
the corresponding operation.

The same service exposes a separate owner-performed export ceremony. It reads
an explicit list of secret keys after browser confirmation and returns the
values once to a PKCE-bound loopback CLI. This ceremony creates no delegated
card grant.

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

Each secret operation constructs a concrete resource from the server-owned
coordinates and validated target:

```text
urn:kdcube:management:secret:<tenant>:<project>:<scope>:<bundle-or-_>:<key>
```

No operation accepts a tenant or project override.

| Operation | Method and route | Result |
| --- | --- | --- |
| `kdcube.management.deployment.inspect` | `GET /api/integrations/management/v1/deployment` | Bounded platform release, aggregate readiness, and declared-application preparation state. |
| `kdcube.management.application.surfaces.read` | `GET /api/integrations/management/v1/applications/{application_id}/surfaces` | Enabled API, MCP, widget, job, and messaging declarations for one exact application. |
| `kdcube.management.application.reload` | `POST /api/integrations/management/v1/applications/{application_id}/reload` | Reload evidence for one exact application from current descriptor authority. |
| `kdcube.management.secret.metadata.read` | `POST /api/integrations/management/v1/secrets/metadata/read` | Existence, provider, and write capability for one exact platform or bundle key. |
| `kdcube.management.secret.value.read` | `POST /api/integrations/management/v1/secrets/value/read` | One exact plaintext value for an admitted caller. Responses use `Cache-Control: no-store`. |
| `kdcube.management.secret.value.write` | `POST /api/integrations/management/v1/secrets/value/write` | Set one exact value through the selected provider. |
| `kdcube.management.secret.delete` | `POST /api/integrations/management/v1/secrets/delete` | Delete one exact value through the selected provider. |

Every call requires an opaque Connection Hub bearer in `Authorization` and a
printable 1..256-character `Idempotency-Key`. Reload accepts the exact JSON
body `{}`. Inspect and surface discovery accept no body. Application IDs are
single exact identifiers; wildcard, path, URL, and reload-all forms are
rejected.

Secret operations accept one exact `platform` or `bundle` target. Bundle
targets include one exact application id. Wildcards, metadata keys, user
scope, path escapes, and platform-to-bundle namespace escapes are rejected
before admission. A secret-operation request digest is a keyed HMAC over the
full validated request, including a write value. It binds approval and replay
to the exact value without exposing a reusable plaintext hash. Secret values
stay outside recovery URLs, logs, Card records, and effect-ledger records.

Protected-resource discovery is available at:

```text
GET /api/integrations/management/v1/.well-known/oauth-protected-resource
```

The document publishes the concrete resources, the deployment's Connection Hub
OAuth issuer, and the delegated operation routes. A missing bearer receives `401`
with that metadata URL in `WWW-Authenticate`.

## Two Authority Journeys

```text
delegated automation

agent or operator CLI
  -> card-bound bearer
  -> exact resource + operation + invocation policy
  -> Connection Hub admission on every call
  -> KDCube selected secret provider
  -> bounded result

owner-performed descriptor export

Connection Hub CLI
  -> exact non-secret key manifest + PKCE challenge + loopback callback
  -> KDCube browser page
  -> current platform admin session + explicit Export once decision
  -> one-use authorization code bound to manifest digest and PKCE verifier
  -> KDCube selected secret provider
  -> new local secrets.yaml + bundles.secrets.yaml directory
```

For delegated automation, choosing `Once` updates the caller's Card and its
invocation policy. The Card continues to identify who may attempt the
operation, while the policy consumes the one admitted invocation. Choosing
`Always` records reusable Card authority.

For human export, the person performs the action directly. The transaction
therefore stands alone: it is short-lived, exact-manifest-bound, and consumed
once, with no Card mutation and no reusable export credential.

## Human Secret Export

The CLI names every requested key explicitly. This works uniformly with file,
host-vault, and cloud secret providers, including providers whose durable
storage intentionally retains only digests of key names and cannot enumerate
the original names.

```bash
connection-hub host secret export \
  --platform-key services.brave.api_key \
  --bundle-key connection-hub@1-0=connections.oauth_state_secret \
  --output-directory ./kdcube-secret-export-20260904
```

The output directory must be new. It contains canonical `secrets.yaml` and
`bundles.secrets.yaml` documents. POSIX hosts create the directory with mode
`0700` and both files with mode `0600`; Windows uses the ACL inherited from
the selected parent directory. The CLI prints paths, counts, digest, and
approval evidence while keeping values out of terminal output.

The protocol has three server routes:

| Route | Purpose |
| --- | --- |
| `POST /secrets/export/start` | Validate and persist the exact manifest, loopback callback, state, and S256 challenge with a short TTL. |
| `GET/POST /secrets/export/authorize` | Resolve a human platform session, display every exact target, and record one approve or deny decision. |
| `POST /secrets/export/exchange` | Atomically consume the code and verifier, then read exactly the approved keys. |

Redis stores a hash-addressed transaction containing non-secret targets,
digests, state, status, approval evidence, and the assurance, evidence-age,
and result-size policy captured when the request starts. Later descriptor
changes cannot weaken those pinned limits; disabling export still stops an
in-flight request. Redis stores the authorization code only as a SHA-256
digest and never stores exported values. Atomic compare and swap permits one
approval transition and one exchange. A failed provider read consumes that
attempt; the operator starts a new visible approval rather than replaying a
partly completed export.

### Human assurance

`management.secret_export.required_assurance` selects the minimum evidence:

| Value | Evidence contract | Built-in state |
| --- | --- | --- |
| `session_confirmation` | Current KDCube platform admin browser cookie plus the exact form decision. | Implemented through the deployment's configured KDCube browser authority. |
| `fresh_authentication` | A recent authentication event whose time and subject are verified. | Fails closed with `human_approval_step_up_unavailable` until the deployment installs a capable verifier. |
| `user_verification` | A fresh user-verifying authenticator ceremony, such as WebAuthn/passkey. | Fails closed until the deployment installs a capable verifier. |

Explicit `Authorization` and ID-token headers are rejected at this boundary,
and Connection Hub delegated authentication is disabled in the verifier. A
delegated admin bearer therefore cannot become human approval.

The verifier receives a `HumanApprovalContext` containing the exact action,
deployment, transaction id, request digest, required assurance, maximum
evidence age, and a validated relative return URL. Each evaluation also names
one explicit phase:

- `present` authorizes showing the exact decision form. An adapter may redirect
  to its challenge; after that challenge succeeds, it may return evidence while
  retaining the request-bound proof for the decision step.
- `commit` authorizes recording the submitted decision. An adapter must return
  evidence directly and should atomically consume its completed proof. A new
  challenge at this phase fails closed and the user starts review again.

For either phase, the verifier returns:

- `HumanApprovalEvidence`, bound to that digest with the verified subject,
  assurance method, and verification time; or
- `HumanApprovalChallenge`, containing a validated relative or HTTPS URL for
  an authority-owned browser ceremony.

The authorize `GET` may follow that challenge. The final decision `POST` never
turns a newly returned challenge into approval; it fails with
`human_approval_restart_required`, and the browser must complete the challenge
before presenting the decision again. KDCube independently checks the evidence
type, digest, age, clock skew, and assurance level before recording approval.

An IdP reauthentication or WebAuthn adapter implements
`HumanApprovalVerifier` and is installed as
`request.app.state.human_approval_verifier` during processor assembly. The
adapter owns provider state and nonce validation, binds the authenticated
subject and authentication time to the supplied context, and verifies the
required platform administrator authority. The export transaction, callback,
and exchange contracts remain provider-neutral.

A production adapter uses this sequence:

1. Persist one short-lived adapter transaction keyed by a random
   `state`, bound to the supplied action, transaction id, request digest,
   subject, return URL, nonce, and required assurance.
2. Redirect to the authority configured by this KDCube deployment. A Cognito
   managed-login adapter requests `prompt=login`; a direct Google OIDC adapter
   requests and validates `auth_time` plus `nonce`. Google does not expose a
   provider-neutral promise that every such redirect displays a password
   challenge, so evidence is accepted only when the returned `auth_time` is
   within the configured age.
3. Exchange the authorization code server-side and validate signature,
   issuer, audience, nonce, state, exact subject continuity, authentication
   time, and current KDCube administrator role.
4. During `present`, return typed evidence for the original request digest but
   retain the completed proof until the decision form is submitted. During
   `commit`, atomically consume that proof and return the same evidence. A
   WebAuthn adapter instead verifies RP id, origin, challenge, signature
   counter, and the user-verification flag before returning
   `user_verification` evidence.

Until one of those adapters is assembled and live-tested for the deployment's
authority, selecting its stronger assurance level remains an intentional
fail-closed configuration. The built-in cookie verifier is never promoted to
fresh-authentication evidence.

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
exact operation, exact application ID, and body. Ordinary operations use
SHA-256; secret operations use HMAC-SHA-256 with the protected-service secret
so a low-entropy value cannot be guessed from the digest offline. The shared
Redis effect ledger is reserved only after current admission succeeds and
before the runtime adapter executes.

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
  secret_export:
    enabled: true
    required_assurance: session_confirmation
    max_evidence_age_seconds: 300
    transaction_ttl_seconds: 180
    consumed_tombstone_seconds: 600
    max_targets: 64
    max_total_value_bytes: 1048576
```

An empty `admission_url` resolves to the Connection Hub app operation on the
same descriptor-owned processor port. A different URL is an explicit
descriptor choice.

The Connection Hub app descriptor registers the workload, both protected
resource families, and every request-bound operation:

```yaml
connections:
  delegated_credentials:
    admission:
      enabled: true
      services:
        kdcube-management:
          secret_ref: connections.delegated_credentials.admission.services.kdcube-management.signing_secret
          resources:
            - "urn:kdcube:management:deployment:*:*"
            - "urn:kdcube:management:secret:*:*:*:*:*"
          request_bound_operations:
            - kdcube.management.application.reload
            - kdcube.management.secret.metadata.read
            - kdcube.management.secret.value.read
            - kdcube.management.secret.value.write
            - kdcube.management.secret.delete
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
| `management/routes.py` | Delegated HTTP methods, bearer/idempotency requirements, discovery, body rules, dependency assembly, and export-router composition. |
| `management/admission.py` | Fresh signed direct-admission call to Connection Hub, plus signed recovery-ticket verification; bearer remains opaque. |
| `management/service.py` | Admission-first orchestration, recovery projection, audit evidence, replay behavior. |
| `management/effect_ledger.py` | Shared atomic reservation and guarded terminal settlement. |
| `management/runtime.py` | Bounded registry/readiness projection, enabled surface discovery, and exact application reload adapter. |
| `management/secret_contracts.py` | Exact platform/bundle target grammar, secret resources, and operation constants. |
| `management/secret_runtime.py` | Provider-neutral metadata, read, write, and delete through the configured `ISecretsManager`. |
| `management/human_approval.py` | Request-bound evidence/challenge interface, independent assurance validation, and built-in platform-browser-session verifier. |
| `management/secret_export.py` | Exact export request digest, Redis transaction state, PKCE binding, and atomic one-use transitions. |
| `management/secret_export_routes.py` | Start, browser approval, and one-use exchange HTTP ceremony. |
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
